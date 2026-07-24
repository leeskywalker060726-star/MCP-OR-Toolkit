import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv


# ============================================================================
# Arm 1: Direct Baseline
# Class-based wrapper for the original ExactOR Agent Client.
# This keeps the direct flow unchanged:
# User prompt -> LLM tool selection -> MCP solver call -> LLM final explanation.
# ============================================================================


class Arm1Agent:
    """Direct baseline agent preserving the original agent_client.py behavior."""

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

    def __init__(
        self,
        *,
        project_dir: str | os.PathLike | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
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

        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY")
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.max_turns = max_turns
        self.request_delay = request_delay
        self.test_file = test_file
        self.report_file = report_file

        if not self.llm_api_key:
            raise RuntimeError(
                "[FATAL ERROR] LLM_API_KEY not found. Please create a .env file in the root directory and add your key!"
            )

        self.llm_client = AsyncOpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )

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

        test_result = self._create_result_record(case_id, expected_algo)
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
                        {"role": "user", "content": user_prompt},
                    ]

                    actual_algo_set = False

                    for turn in range(self.max_turns):
                        if turn > 0:
                            print(f"\n  [{self.llm_model} Self-Correction Analysis (Turn {turn + 1})] ...")

                        response = await self.llm_client.chat.completions.create(
                            model=self.llm_model,
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

                                print(f"  [{self.llm_model} Decision] Selected OR Tool: >>> {tool_name} <<<")
                                self._validate_algorithm_selection(test_result, tool_name, expected_algo)
                                print(f"  [{self.llm_model} Parsed Payload] {test_result['llm_payload']}")
                            else:
                                print(f"  [{self.llm_model} Self-Correction Strategy] Retrying tool: >>> {tool_name} <<<")
                                print(f"  [{self.llm_model} Corrected Payload] {json.dumps(tool_args, ensure_ascii=False)}")

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
                                print(f"[{self.llm_model} Replied directly without calling tools]\n{message.content}")
                                test_result["is_perfect"] = False
                                test_result["errors"].append("LLM refused or failed to call any tools.")
                            else:
                                print(f"[{self.llm_model} Final Response]\n{message.content}")

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
                            f"[{self.llm_model} Aborted] Consecutive errors reached {self.max_turns} turns. Forced stop."
                        )
        except Exception as exc:
            error_message = f"Agent execution failed: {type(exc).__name__}: {exc}"
            test_result["is_perfect"] = False
            test_result["errors"].append(error_message)
            print(f"[Error] {error_message}")

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
            f.write("# ExactOR Arm1 Direct Baseline Test Report\n\n")
            f.write("**Arm**: `ARM_1_DIRECT_BASELINE`\n")
            f.write(f"**Generated At**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model**: {self.llm_model}\n")
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
        }

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
    agent = Arm1Agent()
    await agent.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
