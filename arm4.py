from __future__ import annotations

import argparse
import ast
import asyncio
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False


class Arm4Agent:
    """Prototype for generating validated pure-Python candidate tools."""

    TOOL_NAME_RE = re.compile(r"^solve_[a-z0-9_]+$")
    ALLOWED_IMPORTS = {"math", "itertools", "collections", "heapq", "functools", "statistics", "random"}
    BLOCKED_IMPORTS = {"os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "urllib", "importlib", "ctypes"}
    BLOCKED_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars"}
    BLOCKED_ATTRS = {"__class__", "__subclasses__", "__globals__", "__dict__"}

    GENERATION_PROMPT = """
You are an Operations Research tool prototyping assistant.
First decide whether the existing tools can solve the user's problem. If one can, return JSON with:
{"status":"EXISTING_TOOL_AVAILABLE","recommended_tool":"tool_name","reason":"..."}

If no existing tool covers the problem, generate a small pure-Python candidate tool.
Return only valid JSON with:
{
 "tool_name":"solve_example_problem",
 "description":"...",
 "problem_family":"...",
 "algorithm":"...",
 "limitations":["..."],
 "code":"...",
 "tests":[
   {"name":"basic_case","payload":{},"expected_status":"SUCCESS","expected_keys":["status"]}
 ]
}

The code must define exactly: def solve(payload: dict) -> dict:
It must return {"status":"SUCCESS", ...} or {"status":"ERROR","message":"..."}.
Use only Python standard library modules allowed by the validator.
Do not create a general LP/MILP solver, do not use network, files, subprocess, eval, exec, compile, or dynamic imports.
Prefer small optimization methods such as dynamic programming, greedy, enumeration with pruning, or simple heuristics.
Provide at least two tests.
"""

    def __init__(
        self,
        *,
        project_dir: str | os.PathLike | None = None,
        generated_dir: str = "generated_tools",
        timeout_seconds: float = 5.0,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ):
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent)
        load_dotenv(self.project_dir / ".env")
        load_dotenv()
        self.generated_dir = self.project_dir / generated_dir
        self.candidates_dir = self.generated_dir / "_candidates"
        self.registry_path = self.generated_dir / "registry.json"
        self.timeout_seconds = timeout_seconds
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "deepseek-chat")

    async def list_existing_mcp_tools(self) -> list[dict]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=sys.executable,
                args=[str(self.project_dir / "mcp_server.py")],
                cwd=str(self.project_dir),
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    return [
                        {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                        for tool in response.tools
                    ]
        except Exception:
            return self._list_tools_from_config()

    async def determine_tool_gap(self, user_prompt: str) -> dict:
        tools = await self.list_existing_mcp_tools()
        response = await self.request_tool_candidate(user_prompt, tools)
        parsed = self.parse_generation_response(response)
        if parsed.get("status") == "EXISTING_TOOL_AVAILABLE":
            return parsed
        return {"status": "TOOL_GAP", "candidate": parsed, "existing_tools": tools}

    async def request_tool_candidate(self, user_prompt: str, existing_tools: list[dict]) -> str:
        if not self.llm_api_key:
            raise RuntimeError("LLM_API_KEY is required for tool generation.")
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        messages = [
            {"role": "system", "content": self.GENERATION_PROMPT},
            {
                "role": "user",
                "content": (
                    "Existing MCP tools:\n"
                    f"{json.dumps(existing_tools, ensure_ascii=False, indent=2)}\n\n"
                    "User problem:\n"
                    f"{user_prompt}"
                ),
            },
        ]
        kwargs = {"model": self.llm_model, "messages": messages, "temperature": 0.0}
        try:
            response = await client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
        except Exception as exc:
            if not self._is_response_format_error(exc):
                raise
            response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or "{}"

    def parse_generation_response(self, content: str) -> dict:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Generation response is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Generation response must be a JSON object.")
        if data.get("status") == "EXISTING_TOOL_AVAILABLE":
            return {
                "status": "EXISTING_TOOL_AVAILABLE",
                "recommended_tool": str(data.get("recommended_tool", "")),
                "reason": str(data.get("reason", "")),
            }
        return data

    def validate_tool_name(self, tool_name: str) -> None:
        if not isinstance(tool_name, str) or not self.TOOL_NAME_RE.match(tool_name):
            raise ValueError("tool_name must match ^solve_[a-z0-9_]+$")

    def validate_ast(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ValueError(f"Candidate code has invalid syntax: {exc}") from exc

        solve_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"]
        if len(solve_defs) != 1:
            raise ValueError("Candidate code must define exactly one top-level solve(payload: dict) function.")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".")[0]
                    if root_name in self.BLOCKED_IMPORTS or root_name not in self.ALLOWED_IMPORTS:
                        raise ValueError(f"Import is not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root_name = (node.module or "").split(".")[0]
                if root_name in self.BLOCKED_IMPORTS or root_name not in self.ALLOWED_IMPORTS:
                    raise ValueError(f"Import is not allowed: {node.module}")
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.BLOCKED_CALLS:
                    raise ValueError(f"Call is not allowed: {func.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRS:
                    raise ValueError(f"Attribute access is not allowed: {node.attr}")

    def validate_test_specs(self, tests: Any) -> list[dict]:
        if not isinstance(tests, list) or len(tests) < 2:
            raise ValueError("At least two test specs are required.")
        clean_tests = []
        for test in tests:
            if not isinstance(test, dict):
                raise ValueError("Each test spec must be an object.")
            payload = test.get("payload")
            expected_status = test.get("expected_status", "SUCCESS")
            expected_keys = test.get("expected_keys", ["status"])
            if not isinstance(payload, dict):
                raise ValueError("Test payload must be an object.")
            if expected_status not in {"SUCCESS", "ERROR"}:
                raise ValueError("expected_status must be SUCCESS or ERROR.")
            if not isinstance(expected_keys, list):
                raise ValueError("expected_keys must be a list.")
            clean_tests.append(
                {
                    "name": str(test.get("name", f"test_{len(clean_tests) + 1}")),
                    "payload": payload,
                    "expected_status": expected_status,
                    "expected_keys": [str(key) for key in expected_keys],
                }
            )
        return clean_tests

    def run_candidate_tests(self, candidate_path: Path, tests: list[dict]) -> dict:
        runner = (
            "import importlib.util,json,sys\n"
            "path=sys.argv[1]\n"
            "payload=json.loads(sys.stdin.read())\n"
            "spec=importlib.util.spec_from_file_location('candidate_tool', path)\n"
            "mod=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "result=mod.solve(payload)\n"
            "print(json.dumps(result, ensure_ascii=False))\n"
        )
        passed = []
        failed = []
        for test in tests:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", runner, str(candidate_path)],
                    input=json.dumps(test["payload"], ensure_ascii=False),
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    cwd=str(self.project_dir),
                )
                if proc.returncode != 0:
                    failed.append({"name": test["name"], "error": proc.stderr.strip()})
                    continue
                result = json.loads(proc.stdout)
                missing_keys = [key for key in test["expected_keys"] if key not in result]
                if result.get("status") != test["expected_status"] or missing_keys:
                    failed.append({"name": test["name"], "result": result, "missing_keys": missing_keys})
                else:
                    passed.append(test["name"])
            except subprocess.TimeoutExpired:
                failed.append({"name": test["name"], "error": "timeout"})
            except Exception as exc:
                failed.append({"name": test["name"], "error": f"{type(exc).__name__}: {exc}"})
        return {
            "status": "SUCCESS" if len(passed) >= 2 and not failed else "ERROR",
            "passed": passed,
            "failed": failed,
        }

    def approve_candidate(self, candidate: dict, candidate_path: Path, test_result: dict) -> dict:
        tool_name = candidate["tool_name"]
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        approved_path = self.generated_dir / f"{tool_name}.py"
        shutil.copyfile(candidate_path, approved_path)
        code = approved_path.read_text(encoding="utf-8")
        entry = {
            "tool_name": tool_name,
            "description": str(candidate.get("description", "")),
            "algorithm": str(candidate.get("algorithm", "")),
            "limitations": [str(item) for item in candidate.get("limitations", []) if isinstance(candidate.get("limitations", []), list)],
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": self.llm_model,
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "tests_passed": len(test_result.get("passed", [])),
            "status": "validated candidate tool; MCP registration pending",
        }
        registry = self._load_registry()
        registry["tools"][tool_name] = entry
        self._save_registry(registry)
        return {"status": "APPROVED", "path": str(approved_path), "registry_entry": entry}

    def list_generated_tools(self) -> list[dict]:
        registry = self._load_registry()
        return list(registry.get("tools", {}).values())

    def run_generated_tool(self, tool_name: str, payload: dict) -> dict:
        registry = self._load_registry()
        if tool_name not in registry.get("tools", {}):
            raise ValueError("Generated tool is not approved in registry.")
        tool_path = self.generated_dir / f"{tool_name}.py"
        if not tool_path.exists():
            raise FileNotFoundError(f"Generated tool file not found: {tool_path}")
        return self._run_single_payload(tool_path, payload)

    async def generate_tool(self, user_prompt: str) -> dict:
        tools = await self.list_existing_mcp_tools()
        raw = await self.request_tool_candidate(user_prompt, tools)
        candidate = self.parse_generation_response(raw)
        if candidate.get("status") == "EXISTING_TOOL_AVAILABLE":
            return candidate

        self.validate_tool_name(candidate.get("tool_name", ""))
        code = str(candidate.get("code", ""))
        tests = self.validate_test_specs(candidate.get("tests"))
        self.validate_ast(code)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = self.candidates_dir / f"{candidate['tool_name']}.py"
        candidate_path.write_text(code, encoding="utf-8")
        test_result = self.run_candidate_tests(candidate_path, tests)
        if test_result["status"] != "SUCCESS":
            return {"status": "VALIDATION_FAILED", "test_result": test_result}
        approval = self.approve_candidate(candidate, candidate_path, test_result)
        return {"status": "VALIDATED_CANDIDATE", "approval": approval, "test_result": test_result}

    def self_test(self) -> dict:
        safe_code = (
            "def solve(payload: dict) -> dict:\n"
            "    items = payload.get('items', [])\n"
            "    return {'status': 'SUCCESS', 'count': len(items)}\n"
        )
        timeout_code = "def solve(payload: dict) -> dict:\n    while True:\n        pass\n"
        checks = {}
        with tempfile.TemporaryDirectory() as tmp:
            agent = Arm4Agent(project_dir=tmp, timeout_seconds=0.5, llm_api_key="dummy")
            agent.generated_dir = Path(tmp) / "generated_tools"
            agent.candidates_dir = agent.generated_dir / "_candidates"
            agent.registry_path = agent.generated_dir / "registry.json"
            candidate = agent.candidates_dir / "solve_count_items.py"
            agent.candidates_dir.mkdir(parents=True)
            candidate.write_text(safe_code, encoding="utf-8")

            checks["safe_code_passes_ast"] = self._check_ok(lambda: agent.validate_ast(safe_code))
            checks["import_os_rejected"] = self._check_raises(lambda: agent.validate_ast("import os\ndef solve(payload: dict) -> dict:\n    return {'status':'SUCCESS'}\n"))
            checks["subprocess_rejected"] = self._check_raises(lambda: agent.validate_ast("import subprocess\ndef solve(payload: dict) -> dict:\n    return {'status':'SUCCESS'}\n"))
            checks["open_rejected"] = self._check_raises(lambda: agent.validate_ast("def solve(payload: dict) -> dict:\n    open('x')\n    return {'status':'SUCCESS'}\n"))
            tests = [
                {"name": "a", "payload": {"items": [1]}, "expected_status": "SUCCESS", "expected_keys": ["status", "count"]},
                {"name": "b", "payload": {"items": []}, "expected_status": "SUCCESS", "expected_keys": ["status", "count"]},
            ]
            checks["json_return_ok"] = agent.run_candidate_tests(candidate, tests)["status"] == "SUCCESS"
            bad_tests = [{"name": "bad", "payload": {}, "expected_status": "ERROR", "expected_keys": ["status"]}, tests[0]]
            checks["wrong_objective_test_fails"] = agent.run_candidate_tests(candidate, bad_tests)["status"] == "ERROR"
            timeout_path = agent.candidates_dir / "solve_timeout.py"
            timeout_path.write_text(timeout_code, encoding="utf-8")
            checks["timeout_terminates"] = agent.run_candidate_tests(timeout_path, tests)["status"] == "ERROR"
            approved = agent.approve_candidate(
                {"tool_name": "solve_count_items", "description": "count", "algorithm": "count", "limitations": []},
                candidate,
                {"passed": ["a", "b"]},
            )
            checks["registry_save_read"] = bool(approved["registry_entry"]) and len(agent.list_generated_tools()) == 1
            checks["unapproved_tool_cannot_run"] = self._check_raises(lambda: agent.run_generated_tool("solve_missing", {}))

        failed = [name for name, ok in checks.items() if not ok]
        return {"status": "SUCCESS" if not failed else "ERROR", "checks": checks, "failed": failed}

    def _list_tools_from_config(self) -> list[dict]:
        path = self.project_dir / "tools_config.jsonl"
        tools = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cfg = json.loads(line)
                tools.append(
                    {
                        "name": cfg.get("name"),
                        "description": cfg.get("description", ""),
                        "inputSchema": cfg.get("inputSchema", {}),
                    }
                )
        return tools

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"version": 1, "tools": {}}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("tools"), dict):
                return data
        except Exception:
            pass
        return {"version": 1, "tools": {}}

    def _save_registry(self, registry: dict) -> None:
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.registry_path)

    def _run_single_payload(self, tool_path: Path, payload: dict) -> dict:
        runner = (
            "import importlib.util,json,sys\n"
            "spec=importlib.util.spec_from_file_location('tool', sys.argv[1])\n"
            "mod=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "print(json.dumps(mod.solve(json.loads(sys.stdin.read())), ensure_ascii=False))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", runner, str(tool_path)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=str(self.project_dir),
        )
        if proc.returncode != 0:
            return {"status": "ERROR", "message": proc.stderr.strip()}
        return json.loads(proc.stdout)

    def _is_response_format_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "response_format" in text or "json_object" in text or "unsupported" in text or "not support" in text

    def _check_ok(self, func) -> bool:
        try:
            func()
            return True
        except Exception:
            return False

    def _check_raises(self, func) -> bool:
        try:
            func()
            return False
        except Exception:
            return True


async def async_main(args) -> None:
    agent = Arm4Agent()
    if args.generate:
        result = await agent.generate_tool(args.generate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.list:
        print(json.dumps(agent.list_generated_tools(), ensure_ascii=False, indent=2))
    else:
        print("Use --self-test, --generate 'problem', or --list")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm4 LLM Generated Tool Prototype")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--generate", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(Arm4Agent().self_test(), ensure_ascii=False, indent=2))
    else:
        asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
