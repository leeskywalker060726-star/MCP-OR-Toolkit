import asyncio
import datetime
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv


# ============================================================================
# Arm 2: Analysis Guided Agent
# Arm1 baseline plus a lightweight Problem Analyzer before tool selection.
# This keeps the solver flow unchanged after analysis:
# User prompt -> Problem Analyzer -> LLM tool selection -> MCP solver call -> LLM final explanation.
# ============================================================================


class Arm2Agent:
    """Analysis-guided agent preserving Arm1's MCP tool-calling behavior."""

    SYSTEM_PROMPT = """
            You are a senior Operations Research (OR) optimization expert. You need to analyze the user's problem and select the most appropriate optimization tool (e.g., LP, Pure ILP, KM Matching) to solve it.
            Carefully read the tool descriptions and strictly pass parameters according to the JSON Schema.
            
            [CRITICAL Algorithm Selection & Parameter Rules]:
            1. Algorithm Selection Rule: For ANY programming problem involving integer variables (whether some or all variables are integers), you MUST default to using `solve_mixed_integer_programming` because its underlying Branch and Bound method is the most robust. Unless the user explicitly requests "Cutting Plane method" or "Pure Integer Programming", NEVER select `solve_pure_integer_programming`!
            2. Operator Formatting Rule: When constructing constraints, the 'op' field MUST AND ONLY USE "<=", ">=", or "="! The use of English abbreviations like "LE" or "GE" is STRICTLY PROHIBITED!
            3. Variable Naming & Coefficient Rule (FATAL ERROR WARNING): In the 'terms' dictionary of both objective and constraints, the Key MUST be a pure variable name (e.g., "x1", "y"), and the Value MUST be a numerical coefficient!
               - NEVER put the coefficient inside the variable name (e.g., {"50x": 50} is WRONG; it must be extracted as {"x": 50}).
               - NEVER reverse the key-value pair (e.g., {"500": "x1"} is WRONG; it must be {"x1": 500}).
            
            After obtaining the tool's result, if the result contains "ERROR", analyze the error message and call the tool again to correct it.
            If the calculation is successful, explain the final assignment plan or optimal solution to the user in natural language.
            """

    ANALYZER_PROMPT = """
            You are a Problem Analyzer for Operations Research tasks.
            Extract structured problem features from the user's natural-language problem.
            Do not solve the optimization problem and do not choose a tool.
            Return only one valid JSON object with these fields:
            {
              "problem_type": "linear_programming | mixed_integer_programming | pure_integer_programming | unweighted_bipartite_matching | maximum_weight_matching | unknown",
              "variable_type": "continuous | integer | mixed | binary | assignment | unknown",
              "scale": "small | medium | large | unknown",
              "objective": "short natural language summary",
              "constraints": ["short constraint feature"],
              "signals": ["keywords or facts that influenced the analysis"]
            }
            """

    UNKNOWN_PROBLEM_ANALYSIS = {
        "problem_type": "unknown",
        "variable_type": "unknown",
        "scale": "unknown",
        "objective": "",
        "constraints": [],
        "signals": [],
    }

    def __init__(
        self,
        *,
        project_dir: str | os.PathLike | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        analyzer_llm_api_key: str | None = None,
        analyzer_llm_base_url: str | None = None,
        analyzer_llm_model: str | None = None,
        max_turns: int = 3,
        request_delay: float = 2.0,
        test_file: str = "test_cases.jsonl",
        report_file: str = "test_errors_report.md",
    ):
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0.")
        if request_delay < 0:
            raise ValueError("request_delay cannot be negative.")

        self.project_dir = Path(project_dir or Path(__file__).resolve().parent)
        load_dotenv(self.project_dir / ".env")
        load_dotenv()

        self.selector_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.selector_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.selector_model = llm_model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.analyzer_api_key = (
            analyzer_llm_api_key
            or os.getenv("ANALYZER_LLM_API_KEY")
            or self.selector_api_key
        )
        self.analyzer_base_url = (
            analyzer_llm_base_url
            or os.getenv("ANALYZER_LLM_BASE_URL")
            or self.selector_base_url
        )
        self.analyzer_model = (
            analyzer_llm_model
            or os.getenv("ANALYZER_LLM_MODEL")
            or self.selector_model
        )
        self.analyzer_mode = self._detect_analyzer_mode(
            analyzer_llm_api_key,
            analyzer_llm_base_url,
            analyzer_llm_model,
        )
        self.max_turns = max_turns
        self.request_delay = request_delay
        self.test_file = test_file
        self.report_file = report_file

        if not self.selector_api_key:
            raise RuntimeError(
                "[FATAL ERROR] LLM_API_KEY not found. Please create a .env file in the root directory and add your key!"
            )
        if not self.selector_base_url:
            raise RuntimeError("[FATAL ERROR] LLM_BASE_URL is empty.")
        if not self.selector_model:
            raise RuntimeError("[FATAL ERROR] LLM_MODEL is empty.")
        if not self.analyzer_api_key:
            raise RuntimeError("[FATAL ERROR] Analyzer API key is empty after fallback.")
        if not self.analyzer_base_url:
            raise RuntimeError("[FATAL ERROR] Analyzer base URL is empty after fallback.")
        if not self.analyzer_model:
            raise RuntimeError("[FATAL ERROR] Analyzer model is empty after fallback.")

        self.selector_client = AsyncOpenAI(
            api_key=self.selector_api_key,
            base_url=self.selector_base_url,
        )
        self.analyzer_client = AsyncOpenAI(
            api_key=self.analyzer_api_key,
            base_url=self.analyzer_base_url,
        )

        print(f"[Selector LLM] model={self.selector_model}, base_url={self.selector_base_url}")
        print(f"[Analyzer LLM] model={self.analyzer_model}, base_url={self.analyzer_base_url}")
        print(f"[Analyzer Mode] {self.analyzer_mode}")

    async def solve(
        self,
        user_prompt: str,
        case_id: str = "MANUAL_CASE",
        expected_algo: str | None = None,
    ) -> dict:
        """Unified public entry point for solving one natural-language OR problem."""
        return await self.run_case(
            case_id=case_id,
            expected_algo=expected_algo,
            user_prompt=user_prompt,
        )

    async def analyze_problem(self, user_prompt: str) -> dict:
        """Extract structured problem features for Arm2 tool selection guidance."""
        analysis, _ = await self._analyze_problem_with_call_count(user_prompt)
        return analysis

    async def _analyze_problem_with_call_count(self, user_prompt: str) -> tuple[dict, int]:
        call_count = 0
        self._last_analyzer_call_count = 0
        try:
            call_count += 1
            self._last_analyzer_call_count = call_count
            response = await self._create_analyzer_completion(user_prompt, use_response_format=True)
        except Exception as exc:
            if not self._is_response_format_compatibility_error(exc):
                raise
            call_count += 1
            self._last_analyzer_call_count = call_count
            response = await self._create_analyzer_completion(user_prompt, use_response_format=False)

        content = response.choices[0].message.content or "{}"
        analysis = json.loads(content)
        if not isinstance(analysis, dict):
            raise ValueError("Problem Analyzer response is not a JSON object.")
        self._last_analyzer_call_count = call_count
        return self._normalize_problem_analysis(analysis), call_count

    async def _create_analyzer_completion(self, user_prompt: str, *, use_response_format: bool):
        kwargs = {
            "model": self.analyzer_model,
            "messages": [
                {"role": "system", "content": self.ANALYZER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        return await self.analyzer_client.chat.completions.create(**kwargs)

    def _is_response_format_compatibility_error(self, exc: Exception) -> bool:
        error_text = str(exc).lower()
        return (
            "response_format" in error_text
            or "json_object" in error_text
            or "unsupported" in error_text
            or "not support" in error_text
        )

    async def run_case(
        self,
        case_id: str,
        expected_algo: str | None,
        user_prompt: str,
    ) -> dict:
        """Run one test case or manual prompt and return structured results."""
        print(f"\n{'=' * 80}")
        print(f"[Test ID] {case_id}")
        print(f"[Expected Algorithm] {expected_algo}")
        print(f"[User Prompt] {user_prompt}")
        print(f"{'-' * 80}")

        total_start = time.perf_counter()
        test_result = self._create_result_record(case_id, expected_algo)
        try:
            analysis_start = time.perf_counter()
            try:
                problem_analysis, analyzer_call_count = await self._analyze_problem_with_call_count(user_prompt)
                test_result["analyzer_llm_call_count"] += analyzer_call_count
                test_result["llm_call_count"] = (
                    test_result["analyzer_llm_call_count"]
                    + test_result["selector_llm_call_count"]
                )
                test_result["analysis_success"] = True
                test_result["problem_analysis"] = problem_analysis
                print(f"[Problem Analysis] {json.dumps(problem_analysis, ensure_ascii=False)}")
            except Exception as exc:
                test_result["analyzer_llm_call_count"] += getattr(self, "_last_analyzer_call_count", 0)
                test_result["llm_call_count"] = (
                    test_result["analyzer_llm_call_count"]
                    + test_result["selector_llm_call_count"]
                )
                problem_analysis = self._unknown_problem_analysis()
                test_result["analysis_success"] = False
                test_result["problem_analysis"] = problem_analysis
                test_result["is_perfect"] = False
                test_result["errors"].append(f"Problem analysis failed: {type(exc).__name__}: {exc}")
                print(f"[Error] Problem analysis failed: {type(exc).__name__}: {exc}")
            finally:
                test_result["analysis_time_seconds"] = round(time.perf_counter() - analysis_start, 6)

            guided_user_prompt = self._build_guided_user_prompt(user_prompt, problem_analysis)
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[str(self.project_dir / "mcp_server.py")],
                cwd=str(self.project_dir),
            )

            try:
                async with stdio_client(server_params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        mcp_tools_response = await session.list_tools()
                        openai_tools = self._convert_mcp_tools_to_openai(mcp_tools_response.tools)

                        messages = [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": guided_user_prompt},
                        ]

                        actual_algo_set = False

                        for turn in range(self.max_turns):
                            if turn > 0:
                                print(f"\n  [{self.selector_model} Self-Correction Analysis (Turn {turn + 1})] ...")

                            test_result["selector_llm_call_count"] += 1
                            test_result["llm_call_count"] = (
                                test_result["analyzer_llm_call_count"]
                                + test_result["selector_llm_call_count"]
                            )
                            response = await self.selector_client.chat.completions.create(
                                model=self.selector_model,
                                messages=messages,
                                tools=openai_tools,
                                temperature=0.0,
                            )

                            message = response.choices[0].message

                            if message.tool_calls:
                                messages.append(message)

                                tool_call = message.tool_calls[0]
                                tool_name = tool_call.function.name
                                tool_args = self._parse_tool_arguments(
                                    tool_call.function.arguments,
                                    test_result,
                                    turn,
                                )
                                if tool_args is None:
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "content": (
                                                "ERROR: Tool arguments are not valid JSON. "
                                                "Please correct the JSON arguments and call the tool again."
                                            ),
                                        }
                                    )
                                    continue

                                if not actual_algo_set:
                                    test_result["actual_algo"] = tool_name
                                    test_result["llm_payload"] = json.dumps(tool_args, ensure_ascii=False)
                                    actual_algo_set = True

                                    print(f"  [{self.selector_model} Decision] Selected OR Tool: >>> {tool_name} <<<")
                                    self._validate_algorithm_selection(test_result, tool_name, expected_algo)
                                    print(f"  [{self.selector_model} Parsed Payload] {test_result['llm_payload']}")
                                else:
                                    print(f"  [{self.selector_model} Self-Correction Strategy] Retrying tool: >>> {tool_name} <<<")
                                    print(f"  [{self.selector_model} Corrected Payload] {json.dumps(tool_args, ensure_ascii=False)}")

                                try:
                                    mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                                    result_text = self._extract_mcp_result_text(mcp_result, test_result, turn)
                                except Exception as exc:
                                    test_result["is_perfect"] = False
                                    test_result["errors"].append(f"Turn {turn + 1} MCP tool call failed: {exc}")
                                    result_text = f"ERROR: MCP tool call failed: {exc}"

                                test_result["engine_output"] = result_text
                                print(f"  [Engine Output] {result_text}")
                                self._capture_engine_error(test_result, result_text, turn)

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "content": result_text,
                                    }
                                )
                            else:
                                print(f"{'-' * 80}")
                                if turn == 0:
                                    print(f"[{self.selector_model} Replied directly without calling tools]\n{message.content}")
                                    test_result["is_perfect"] = False
                                    test_result["errors"].append("LLM refused or failed to call any tools.")
                                else:
                                    print(f"[{self.selector_model} Final Response]\n{message.content}")

                                test_result["llm_final_response"] = message.content or ""

                                if "<｜｜DSML｜｜" in test_result["llm_final_response"]:
                                    test_result["is_perfect"] = False
                                    test_result["errors"].append(
                                        "Model output contains raw Tool Call tags. Self-correction loop might have failed due to tool config or prompt issues."
                                    )
                                break
                        else:
                            test_result["is_perfect"] = False
                            test_result["errors"].append(
                                f"Reached maximum turns ({self.max_turns}). Model failed to reach a valid conclusion."
                            )
                            print(f"{'-' * 80}")
                            print(
                                f"[{self.selector_model} Aborted] Consecutive errors reached {self.max_turns} turns. Forced stop."
                            )
            except Exception as exc:
                error_message = f"Agent execution failed: {type(exc).__name__}: {exc}"
                test_result["is_perfect"] = False
                test_result["errors"].append(error_message)
                print(f"[Error] {error_message}")
        except Exception as exc:
            error_message = f"Agent execution failed: {type(exc).__name__}: {exc}"
            test_result["is_perfect"] = False
            test_result["errors"].append(error_message)
            print(f"[Error] {error_message}")
        finally:
            test_result["total_time_seconds"] = round(time.perf_counter() - total_start, 6)

        return test_result

    async def run_all_tests(self, test_file: str | None = None) -> list[dict]:
        """Batch-run JSONL test cases and generate the Markdown error report."""
        test_path = self.project_dir / (test_file or self.test_file)
        if not test_path.exists():
            print(f"[FATAL ERROR] Cannot find test case file {test_path}. Please create it first!")
            return []

        print(">>> Starting ExactOR Agent Batch Testing Engine <<<")

        test_results = []
        with test_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                case = {}
                try:
                    case = json.loads(line)
                    res = await self.run_case(case["id"], case["expected_algo"], case["prompt"])
                    test_results.append(res)
                    if self.request_delay > 0:
                        await asyncio.sleep(self.request_delay)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSONL line {line_num}: {line}")
                    test_results.append(
                        self._create_failed_record(
                            case_id=f"INVALID_JSON_LINE_{line_num}",
                            expected_algo=None,
                            error=f"Invalid JSONL line: {line}",
                        )
                    )
                except Exception as exc:
                    print(f"[Error] Test case {case.get('id', 'Unknown')} execution failed: {exc}")
                    test_results.append(
                        self._create_failed_record(
                            case_id=case.get("id", "Unknown"),
                            expected_algo=case.get("expected_algo"),
                            error=f"Python runtime fatal crash: {exc}",
                            user_prompt=case.get("prompt"),
                        )
                    )

        self.generate_error_report(test_results)
        return test_results

    def generate_error_report(self, results: list[dict]) -> None:
        """Summarize errors during testing into a Markdown report."""
        report_path = self.project_dir / self.report_file
        failed_results = [res for res in results if not res.get("is_perfect")]

        with report_path.open("w", encoding="utf-8") as f:
            f.write("# ExactOR Arm2 Analysis Guided Test Report\n\n")
            f.write("**Arm**: `ARM_2_ANALYSIS_GUIDED`\n")
            f.write(f"**Generated At**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analyzer Model**: `{self.analyzer_model}`\n")
            f.write(f"**Analyzer Base URL**: `{self.analyzer_base_url}`\n")
            f.write(f"**Selector Model**: `{self.selector_model}`\n")
            f.write(f"**Selector Base URL**: `{self.selector_base_url}`\n")
            f.write(
                f"**Total Cases**: {len(results)} | **Passed**: {len(results) - len(failed_results)} | **Failed**: {len(failed_results)}\n\n"
            )
            f.write("---\n\n")

            if not failed_results:
                f.write("**Awesome! All test cases passed perfectly. No errors found.**\n")
                print(
                    f"\n>>> Batch testing completed! Detected 0 error(s). Detailed report written to `{report_path.name}` <<<"
                )
                return

            for res in failed_results:
                f.write(f"## Failed Case: {res.get('case_id')}\n\n")
                f.write("### Identified Issues:\n")
                for err in res.get("errors", []):
                    f.write(f"- **{err}**\n")

                f.write("\n### Execution Context:\n")
                f.write(f"- **Expected Algorithm**: `{res.get('expected_algo')}`\n")
                f.write(f"- **Actual Algorithm**: `{res.get('actual_algo')}`\n")
                f.write(f"- **Analysis Success**: `{res.get('analysis_success')}`\n")
                f.write(f"- **Analysis Time**: `{res.get('analysis_time_seconds')} seconds`\n")
                f.write(f"- **Total Time**: `{res.get('total_time_seconds')} seconds`\n")
                f.write(f"- **Analyzer Model**: `{res.get('analyzer_model')}`\n")
                f.write(f"- **Analyzer Base URL**: `{res.get('analyzer_base_url')}`\n")
                f.write(f"- **Selector Model**: `{res.get('selector_model')}`\n")
                f.write(f"- **Selector Base URL**: `{res.get('selector_base_url')}`\n")
                f.write(f"- **Analyzer LLM Call Count**: `{res.get('analyzer_llm_call_count')}`\n")
                f.write(f"- **Selector LLM Call Count**: `{res.get('selector_llm_call_count')}`\n")
                f.write(f"- **Total LLM Call Count**: `{res.get('llm_call_count')}`\n")
                f.write(
                    "- **Problem Analysis**:\n```json\n"
                    f"{json.dumps(res.get('problem_analysis', {}), ensure_ascii=False, indent=2)}\n"
                    "```\n"
                )
                f.write(f"- **LLM Payload**:\n```json\n{res.get('llm_payload')}\n```\n")
                f.write(f"- **Engine Output**:\n```json\n{res.get('engine_output')}\n```\n")
                f.write("\n---\n\n")

        print(
            f"\n>>> Batch testing completed! Detected {len(failed_results)} error(s). Detailed report written to `{report_path.name}` <<<"
        )

    def _convert_mcp_tools_to_openai(self, mcp_tools: list) -> list[dict]:
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return openai_tools

    def _validate_algorithm_selection(
        self,
        test_result: dict,
        tool_name: str,
        expected_algo: str | None,
    ) -> None:
        if expected_algo is None:
            print("  [Validation] SKIP: No expected algorithm provided.")
            return

        if tool_name == expected_algo:
            print("  [Validation] PASS: Algorithm selection correct.")
        elif expected_algo == "solve_pure_integer_programming" and tool_name == "solve_mixed_integer_programming":
            print("  [Validation] PASS (Relaxed): Using MILP for Pure ILP is valid in engineering.")
        else:
            print(f"  [Validation] FAIL: Algorithm selection incorrect (Expected: {expected_algo})")
            test_result["is_perfect"] = False
            test_result["errors"].append(
                f"Algorithm selection error: Expected `{expected_algo}`, but model chose `{tool_name}`."
            )

    def _create_result_record(self, case_id: str, expected_algo: str | None) -> dict:
        return {
            "case_id": case_id,
            "expected_algo": expected_algo,
            "is_perfect": True,
            "errors": [],
            "actual_algo": "None",
            "llm_payload": "None",
            "engine_output": "None",
            "llm_final_response": "None",
            "problem_analysis": {},
            "analysis_success": True,
            "analysis_time_seconds": 0.0,
            "total_time_seconds": 0.0,
            "analyzer_model": self.analyzer_model,
            "selector_model": self.selector_model,
            "analyzer_base_url": self.analyzer_base_url,
            "selector_base_url": self.selector_base_url,
            "analyzer_llm_call_count": 0,
            "selector_llm_call_count": 0,
            "llm_call_count": 0,
        }

    def _build_guided_user_prompt(self, user_prompt: str, problem_analysis: dict) -> str:
        return (
            "User Problem:\n\n"
            f"{user_prompt}\n\n\n"
            "Problem Analysis:\n\n"
            f"{json.dumps(problem_analysis, ensure_ascii=False, indent=2)}\n\n\n"
            "Please select the most appropriate MCP tool.\n"
            "The problem analysis is auxiliary and may be imperfect.\n"
            "If it conflicts with the original user problem or MCP tool descriptions,\n"
            "prioritize the original problem and the tool schemas."
        )

    def get_llm_config_summary(self) -> dict:
        return {
            "analyzer_model": self.analyzer_model,
            "analyzer_base_url": self.analyzer_base_url,
            "selector_model": self.selector_model,
            "selector_base_url": self.selector_base_url,
            "analyzer_mode": self.analyzer_mode,
        }

    def _detect_analyzer_mode(
        self,
        analyzer_llm_api_key: str | None,
        analyzer_llm_base_url: str | None,
        analyzer_llm_model: str | None,
    ) -> str:
        has_independent_config = all(
            [
                analyzer_llm_api_key or os.getenv("ANALYZER_LLM_API_KEY"),
                analyzer_llm_base_url or os.getenv("ANALYZER_LLM_BASE_URL"),
                analyzer_llm_model or os.getenv("ANALYZER_LLM_MODEL"),
            ]
        )
        return "independent" if has_independent_config else "fallback-to-selector"

    def _normalize_problem_analysis(self, analysis: dict) -> dict:
        allowed_problem_types = {
            "linear_programming",
            "mixed_integer_programming",
            "pure_integer_programming",
            "unweighted_bipartite_matching",
            "maximum_weight_matching",
            "unknown",
        }
        allowed_variable_types = {
            "continuous",
            "integer",
            "mixed",
            "binary",
            "assignment",
            "unknown",
        }
        allowed_scales = {"small", "medium", "large", "unknown"}

        problem_type = analysis.get("problem_type")
        variable_type = analysis.get("variable_type")
        scale = analysis.get("scale")
        objective = analysis.get("objective")
        constraints = analysis.get("constraints")
        signals = analysis.get("signals")

        if problem_type not in allowed_problem_types:
            problem_type = "unknown"
        if variable_type not in allowed_variable_types:
            variable_type = "unknown"
        if scale not in allowed_scales:
            scale = "unknown"
        if not isinstance(objective, str):
            objective = ""
        if not isinstance(constraints, list):
            constraints = []
        if not isinstance(signals, list):
            signals = []

        return {
            "problem_type": problem_type,
            "variable_type": variable_type,
            "scale": scale,
            "objective": objective,
            "constraints": [str(item) for item in constraints],
            "signals": [str(item) for item in signals],
        }

    def _unknown_problem_analysis(self) -> dict:
        return dict(self.UNKNOWN_PROBLEM_ANALYSIS)

    def _create_failed_record(
        self,
        *,
        case_id: str,
        expected_algo: str | None,
        error: str,
        user_prompt: str | None = None,
    ) -> dict:
        result = self._create_result_record(case_id, expected_algo)
        result["is_perfect"] = False
        result["errors"].append(error)
        if user_prompt is not None:
            result["user_prompt"] = user_prompt
        return result

    def _parse_tool_arguments(self, raw_arguments: str, test_result: dict, turn: int) -> dict | None:
        try:
            return json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            test_result["is_perfect"] = False
            test_result["llm_payload"] = raw_arguments
            test_result["errors"].append(
                f"Turn {turn + 1} LLM tool arguments are not valid JSON: {exc}"
            )
            print(f"  [Error] Invalid tool JSON arguments: {exc}")
            return None

    def _extract_mcp_result_text(self, mcp_result, test_result: dict, turn: int) -> str:
        if not getattr(mcp_result, "content", None):
            test_result["is_perfect"] = False
            test_result["errors"].append(f"Turn {turn + 1} MCP returned empty content.")
            return "ERROR: MCP returned empty content."

        first_content = mcp_result.content[0]
        result_text = getattr(first_content, "text", None)
        if result_text is None or result_text == "":
            test_result["is_perfect"] = False
            test_result["errors"].append(f"Turn {turn + 1} MCP returned empty text content.")
            return "ERROR: MCP returned empty text content."

        return result_text

    def _capture_engine_error(self, test_result: dict, result_text: str, turn: int) -> None:
        try:
            res_json = json.loads(result_text)
            if res_json.get("status") == "ERROR":
                test_result["is_perfect"] = False
                test_result["errors"].append(
                    f"Turn {turn + 1} C++ Engine Execution Error: {res_json.get('message')}"
                )
        except Exception:
            if "ERROR" in result_text:
                test_result["is_perfect"] = False
                test_result["errors"].append(
                    f"Turn {turn + 1} C++ Engine Execution Error: {result_text}"
                )


async def main():
    agent = Arm2Agent()
    await agent.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
