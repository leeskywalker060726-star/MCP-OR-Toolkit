import asyncio
import json
import os
import sys
import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ============================================================================
# ExactOR Agent Client (LLM Agent Brain with Automated Error Reporting)
# Responsibilities: Connect to DeepSeek API, read test cases in bulk, validate 
# Natural Language <-> Math Engine interaction, and summarize errors.
# Upgrades: Introduced an Agentic Loop for multi-turn self-correction if the engine fails.
# ============================================================================

# 1. Load environment variables from .env file
load_dotenv()

# 2. Dynamically load LLM configurations
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com") 
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat") 

if not LLM_API_KEY:
    print("[FATAL ERROR] LLM_API_KEY not found. Please create a .env file in the root directory and add your key!")
    sys.exit(1)

# Initialize LLM Client
llm_client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

async def run_agent(case_id: str, expected_algo: str, user_prompt: str) -> dict:
    """Run a single test case and return structured results for the report."""
    print(f"\n{'=' * 80}")
    print(f"[Test ID] {case_id}")
    print(f"[Expected Algorithm] {expected_algo}")
    print(f"[User Prompt] {user_prompt}")
    print(f"{'-' * 80}")
    
    test_result = {
        "case_id": case_id,
        "expected_algo": expected_algo,
        "is_perfect": True,
        "errors": [],
        "actual_algo": "None",
        "llm_payload": "None",
        "engine_output": "None",
        "llm_final_response": "None"
    }
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_tools_response = await session.list_tools()
            
            openai_tools = []
            for tool in mcp_tools_response.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

            # [Ultimate System Prompt: Strict rules for algorithm selection and parameter formatting]
            system_prompt = """
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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            MAX_TURNS = 3 # Maximum allowed turns for LLM self-correction
            actual_algo_set = False
            
            for turn in range(MAX_TURNS):
                if turn > 0:
                    print(f"\n  [{LLM_MODEL} Self-Correction Analysis (Turn {turn + 1})] ...")
                    
                response = await llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=openai_tools,
                    temperature=0.0 # Strict math modeling, temperature set to 0 for maximum stability
                )

                message = response.choices[0].message
                
                if message.tool_calls:
                    messages.append(message)
                    
                    # For simplicity, we only record the first tool call
                    tool_call = message.tool_calls[0]
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if not actual_algo_set:
                        test_result["actual_algo"] = tool_name
                        test_result["llm_payload"] = json.dumps(tool_args, ensure_ascii=False)
                        actual_algo_set = True
                        
                        print(f"  [{LLM_MODEL} Decision] Selected OR Tool: >>> {tool_name} <<<")
                        
                        # Error Capture 1: Incorrect algorithm selection (with relaxed compatibility)
                        if tool_name == expected_algo:
                            print(f"  [Validation] PASS: Algorithm selection correct.")
                        elif expected_algo == "solve_pure_integer_programming" and tool_name == "solve_mixed_integer_programming":
                            print(f"  [Validation] PASS (Relaxed): Using MILP for Pure ILP is valid in engineering.")
                        else:
                            print(f"  [Validation] FAIL: Algorithm selection incorrect (Expected: {expected_algo})")
                            test_result["is_perfect"] = False
                            test_result["errors"].append(f"Algorithm selection error: Expected `{expected_algo}`, but model chose `{tool_name}`.")
                            
                        print(f"  [{LLM_MODEL} Parsed Payload] {test_result['llm_payload']}")
                    else:
                        print(f"  [{LLM_MODEL} Self-Correction Strategy] Retrying tool: >>> {tool_name} <<<")
                        print(f"  [{LLM_MODEL} Corrected Payload] {json.dumps(tool_args, ensure_ascii=False)}")
                    
                    mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                    result_text = mcp_result.content[0].text
                    
                    # Always record the latest computation result
                    test_result["engine_output"] = result_text
                    print(f"  [Engine Output] {result_text}")
                    
                    # Error Capture 2: Underlying C++ engine error
                    try:
                        res_json = json.loads(result_text)
                        if res_json.get("status") == "ERROR":
                            test_result["is_perfect"] = False
                            test_result["errors"].append(f"Turn {turn + 1} C++ Engine Execution Error: {res_json.get('message')}")
                    except:
                        pass
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })
                    
                else:
                    # If no tool is called, the LLM has concluded and is generating the business report
                    print(f"{'-' * 80}")
                    if turn == 0:
                        print(f"[{LLM_MODEL} Replied directly without calling tools]\n{message.content}")
                        test_result["is_perfect"] = False
                        test_result["errors"].append("LLM refused or failed to call any tools.")
                    else:
                        print(f"[{LLM_MODEL} Final Response]\n{message.content}")
                        
                    test_result["llm_final_response"] = message.content
                    
                    # Error Capture 3: Check for self-correction loop failure (raw internal tags leaked)
                    if "<｜｜DSML｜｜" in test_result["llm_final_response"]:
                        test_result["is_perfect"] = False
                        test_result["errors"].append("Model output contains raw Tool Call tags. Self-correction loop might have failed due to tool config or prompt issues.")
                        
                    # Successfully got the final text, break the multi-turn retry loop
                    break
            else:
                # Triggered for-else block: tried MAX_TURNS but failed to give a normal response
                test_result["is_perfect"] = False
                test_result["errors"].append(f"Reached maximum turns ({MAX_TURNS}). Model failed to reach a valid conclusion.")
                print(f"{'-' * 80}")
                print(f"[{LLM_MODEL} Aborted] Consecutive errors reached {MAX_TURNS} turns. Forced stop.")
                
    return test_result

def generate_error_report(results: list):
    """Summarize errors during testing into a Markdown report."""
    report_file = "test_errors_report.md"
    failed_results = [res for res in results if not res["is_perfect"]]
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# ExactOR Agent Automated Test Error Report\n\n")
        f.write(f"**Generated At**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {LLM_MODEL}\n")
        f.write(f"**Total Cases**: {len(results)} | **Passed**: {len(results) - len(failed_results)} | **Failed**: {len(failed_results)}\n\n")
        f.write(f"---\n\n")
        
        if not failed_results:
            f.write("**Awesome! All test cases passed perfectly. No errors found.**\n")
            return
            
        for res in failed_results:
            f.write(f"## Failed Case: {res['case_id']}\n\n")
            f.write("### Identified Issues:\n")
            for err in res['errors']:
                f.write(f"- **{err}**\n")
                
            f.write("\n### Execution Context:\n")
            f.write(f"- **Expected Algorithm**: `{res['expected_algo']}`\n")
            f.write(f"- **Actual Algorithm**: `{res['actual_algo']}`\n")
            f.write(f"- **LLM Payload**:\n```json\n{res['llm_payload']}\n```\n")
            f.write(f"- **Engine Output**:\n```json\n{res['engine_output']}\n```\n")
            
            f.write("\n---\n\n")
            
    print(f"\n>>> Batch testing completed! Detected {len(failed_results)} error(s). Detailed report written to `{report_file}` <<<")


async def run_all_tests():
    test_file = "test_cases.jsonl"
    if not os.path.exists(test_file):
        print(f"[FATAL ERROR] Cannot find test case file {test_file}. Please create it first!")
        return
        
    print(">>> Starting ExactOR Agent Batch Testing Engine <<<")
    
    test_results = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            try:
                case = json.loads(line)
                res = await run_agent(case["id"], case["expected_algo"], case["prompt"])
                test_results.append(res)
                # Wait for 2 seconds to avoid triggering API Rate Limits
                await asyncio.sleep(2) 
            except json.JSONDecodeError:
                print(f"[Warning] Skipping invalid JSONL line: {line}")
            except Exception as e:
                print(f"[Error] Test case {case.get('id', 'Unknown')} execution failed: {e}")
                test_results.append({
                    "case_id": case.get('id', 'Unknown'),
                    "is_perfect": False,
                    "errors": [f"Python runtime fatal crash: {str(e)}"]
                })
                
    # Generate report after all tests are finished
    generate_error_report(test_results)

if __name__ == "__main__":
    asyncio.run(run_all_tests())