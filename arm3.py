from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


class Arm3Agent:
    """Bandit strategy selector over the existing Arm1 and Arm2 agents."""

    STRATEGIES = ("arm1_direct", "arm2_analysis_guided")

    def __init__(
        self,
        *,
        project_dir: str | os.PathLike | None = None,
        state_file: str = "arm3_bandit_state.json",
        ucb_c: float | None = None,
        update_state: bool = True,
    ):
        self.project_dir = Path(project_dir or Path(__file__).resolve().parent)
        self.state_path = self.project_dir / state_file
        self.ucb_c = float(ucb_c if ucb_c is not None else os.getenv("ARM3_UCB_C", "1.414"))
        self.update_state_enabled = update_state
        self.strategy_factories = {
            "arm1_direct": self._create_arm1_agent,
            "arm2_analysis_guided": self._create_arm2_agent,
        }

    async def analyze_context(self, user_prompt: str) -> dict:
        try:
            from arm2 import Arm2Agent

            agent = self._instantiate_agent(Arm2Agent, "Arm2Agent")
            analysis = await agent.analyze_problem(user_prompt)
        except Exception as exc:
            return {
                "problem_type": "unknown",
                "variable_type": "unknown",
                "scale": "unknown",
                "context_key": "unknown|unknown|unknown",
                "analysis_success": False,
                "analysis_error": f"{type(exc).__name__}: {exc}",
            }

        problem_type = str(analysis.get("problem_type") or "unknown")
        variable_type = str(analysis.get("variable_type") or "unknown")
        scale = str(analysis.get("scale") or "unknown")
        return {
            "problem_type": problem_type,
            "variable_type": variable_type,
            "scale": scale,
            "context_key": f"{problem_type}|{variable_type}|{scale}",
            "analysis_success": True,
            "problem_analysis": analysis,
        }

    def load_bandit_state(self) -> dict:
        if not self.state_path.exists():
            return self._empty_state()
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                raw_state = json.load(f)
            return self._sanitize_state(raw_state)
        except Exception:
            return self._empty_state()

    def save_bandit_state(self, state: dict) -> None:
        clean_state = self._sanitize_state(state)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.state_path.name}.",
            suffix=".tmp",
            dir=str(self.state_path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(clean_state, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.state_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    async def select_strategy(self, context_key: str, state: dict) -> dict:
        arms = self._ensure_context(state, context_key)
        try:
            selection = await self._select_strategy_with_mcp_ucb(arms)
            selection["selection_backend"] = "mcp_ucb"
            return selection
        except Exception as exc:
            selection = self._select_strategy_local_ucb(arms)
            selection["selection_backend"] = "local_ucb_fallback"
            selection["selection_error"] = f"{type(exc).__name__}: {exc}"
            return selection

    async def execute_strategy(
        self,
        strategy_name: str,
        user_prompt: str,
        case_id: str = "MANUAL_CASE",
        expected_algo: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        try:
            if strategy_name not in self.strategy_factories:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            agent = self.strategy_factories[strategy_name]()
            self._validate_solve_interface(agent, strategy_name)
            result = await agent.solve(
                user_prompt=user_prompt,
                case_id=case_id,
                expected_algo=expected_algo,
            )
            return {
                "status": "SUCCESS" if result.get("is_perfect", False) else "ERROR",
                "strategy_result": result,
                "execution_time_seconds": round(time.perf_counter() - started, 6),
                "errors": list(result.get("errors", [])),
            }
        except Exception as exc:
            return {
                "status": "ERROR",
                "strategy_result": {},
                "execution_time_seconds": round(time.perf_counter() - started, 6),
                "errors": [f"Strategy execution failed: {type(exc).__name__}: {exc}"],
            }

    def compute_reward(
        self,
        execution_result: dict,
        *,
        expected_algo: str | None = None,
        reward_override: Optional[float] = None,
    ) -> tuple[float, dict]:
        """Process reward for bandit routing; it is not a proof of solution quality."""
        if reward_override is not None:
            reward = self._clamp_reward(float(reward_override))
            return reward, {"override": reward}

        strategy_result = execution_result.get("strategy_result") or {}
        engine_output = str(strategy_result.get("engine_output") or "")
        final_response = str(strategy_result.get("llm_final_response") or "")
        errors = execution_result.get("errors") or []
        actual_algo = strategy_result.get("actual_algo")
        elapsed = float(execution_result.get("execution_time_seconds") or 0.0)

        execution_success = 0.2 if execution_result.get("status") == "SUCCESS" and not errors else 0.0
        engine_success = 0.3 if engine_output and "ERROR" not in engine_output else 0.0
        final_answer = 0.2 if final_response else 0.0
        if expected_algo is None:
            algorithm_match = 0.2 if actual_algo and actual_algo != "None" else 0.0
        elif actual_algo == expected_algo:
            algorithm_match = 0.2
        elif expected_algo == "solve_pure_integer_programming" and actual_algo == "solve_mixed_integer_programming":
            algorithm_match = 0.2
        else:
            algorithm_match = 0.0
        efficiency = 0.1 if elapsed <= 10 else max(0.0, 0.1 * (30.0 - min(elapsed, 30.0)) / 20.0)
        if not engine_output and not final_response:
            efficiency = 0.0

        breakdown = {
            "execution_success": execution_success,
            "engine_success": engine_success,
            "final_answer": final_answer,
            "algorithm_match": algorithm_match,
            "efficiency": efficiency,
        }
        return self._clamp_reward(sum(breakdown.values())), breakdown

    def update_bandit_state(self, state: dict, context_key: str, strategy_name: str, reward: float) -> dict:
        clean_reward = self._clamp_reward(reward)
        arms = self._ensure_context(state, context_key)
        arm = arms[strategy_name]
        arm["pulls"] = int(arm.get("pulls", 0)) + 1
        arm["total_reward"] = float(arm.get("total_reward", 0.0)) + clean_reward
        return state

    async def solve(
        self,
        user_prompt: str,
        case_id: str = "MANUAL_CASE",
        expected_algo: str | None = None,
        reward_override: Optional[float] = None,
    ) -> dict:
        started = time.perf_counter()
        errors = []
        context = await self.analyze_context(user_prompt)
        context_key = context.get("context_key", "unknown|unknown|unknown")
        state = self.load_bandit_state()
        selection = await self.select_strategy(context_key, state)
        selected_strategy = selection["selected_strategy"]
        execution = await self.execute_strategy(selected_strategy, user_prompt, case_id, expected_algo)
        reward, reward_breakdown = self.compute_reward(
            execution,
            expected_algo=expected_algo,
            reward_override=reward_override,
        )
        if execution.get("errors"):
            errors.extend(execution["errors"])
        if self.update_state_enabled:
            state = self.update_bandit_state(state, context_key, selected_strategy, reward)
            self.save_bandit_state(state)
        return {
            "status": "SUCCESS" if execution.get("status") == "SUCCESS" else "ERROR",
            "case_id": case_id,
            "expected_algo": expected_algo,
            "selected_strategy": selected_strategy,
            "selection_backend": selection.get("selection_backend"),
            "context": context,
            "bandit_scores": selection.get("bandit_scores", {}),
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "execution_time_seconds": execution.get("execution_time_seconds", 0.0),
            "total_time_seconds": round(time.perf_counter() - started, 6),
            "strategy_result": execution.get("strategy_result", {}),
            "errors": errors,
        }

    async def run_batch_tests(self, test_file: str = "test_cases.jsonl") -> list[dict]:
        path = self.project_dir / test_file
        results = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    case = json.loads(line)
                    results.append(
                        await self.solve(
                            case["prompt"],
                            case_id=case["id"],
                            expected_algo=case.get("expected_algo"),
                        )
                    )
                except Exception as exc:
                    results.append(
                        {
                            "status": "ERROR",
                            "case_id": f"LINE_{line_num}",
                            "errors": [f"Batch case failed: {type(exc).__name__}: {exc}"],
                        }
                    )
        return results

    def self_test(self) -> dict:
        checks = []
        state = self._empty_state()
        arms = self._ensure_context(state, "unknown|unknown|unknown")
        selected = self._select_strategy_local_ucb(arms)
        checks.append(("ucb_cold_start", selected["selected_strategy"] == "arm1_direct"))

        with tempfile.TemporaryDirectory() as tmp:
            agent = Arm3Agent(project_dir=tmp, state_file="state.json", update_state=False)
            state = agent._empty_state()
            agent.update_bandit_state(state, "ctx", "arm1_direct", 0.7)
            agent.save_bandit_state(state)
            loaded = agent.load_bandit_state()
            checks.append(("state_roundtrip", loaded["contexts"]["ctx"]["arm1_direct"]["pulls"] == 1))
            (Path(tmp) / "state.json").write_text("{bad json", encoding="utf-8")
            checks.append(("corrupt_json_recovery", agent.load_bandit_state() == agent._empty_state()))

        reward, _ = self.compute_reward({"status": "SUCCESS"}, reward_override=4.0)
        checks.append(("reward_clamp_high", reward == 1.0))
        reward, _ = self.compute_reward({"status": "SUCCESS"}, reward_override=-1.0)
        checks.append(("reward_clamp_low", reward == 0.0))

        async def failing_solve(**kwargs):
            raise RuntimeError("boom")

        class FailingAgent:
            async def solve(self, user_prompt: str, case_id: str = "x", expected_algo: str | None = None):
                return await failing_solve()

        self.strategy_factories["arm1_direct"] = lambda: FailingAgent()
        result = asyncio.run(self.execute_strategy("arm1_direct", "x"))
        checks.append(("strategy_exception_capture", result["status"] == "ERROR"))

        failed = [name for name, ok in checks if not ok]
        return {"status": "SUCCESS" if not failed else "ERROR", "checks": dict(checks), "failed": failed}

    async def _select_strategy_with_mcp_ucb(self, arms: dict) -> dict:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.project_dir / "mcp_server.py")],
            cwd=str(self.project_dir),
        )
        payload = {
            "arms": arms,
            "hyperparameters": {"c": self.ucb_c, "seed": 42},
        }
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("solve_ucb_bandit", arguments={"payload": payload})
        text = result.content[0].text
        data = json.loads(text)
        if data.get("status") != "SUCCESS":
            raise RuntimeError(data.get("message", text))
        chosen = data.get("chosen_arm")
        if chosen not in self.STRATEGIES:
            raise RuntimeError(f"MCP UCB returned invalid arm: {chosen}")
        return {"selected_strategy": chosen, "bandit_scores": data.get("scores", {})}

    def _select_strategy_local_ucb(self, arms: dict) -> dict:
        for strategy in self.STRATEGIES:
            if int(arms[strategy].get("pulls", 0)) == 0:
                return {
                    "selected_strategy": strategy,
                    "bandit_scores": {name: (float("inf") if name == strategy else 0.0) for name in self.STRATEGIES},
                }
        total_pulls = sum(int(arms[name]["pulls"]) for name in self.STRATEGIES)
        scores = {}
        for name in self.STRATEGIES:
            pulls = int(arms[name]["pulls"])
            mean = float(arms[name]["total_reward"]) / pulls
            scores[name] = mean + self.ucb_c * math.sqrt(math.log(total_pulls) / pulls)
        selected = sorted(self.STRATEGIES, key=lambda name: (-scores[name], name))[0]
        return {"selected_strategy": selected, "bandit_scores": scores}

    def _ensure_context(self, state: dict, context_key: str) -> dict:
        contexts = state.setdefault("contexts", {})
        arms = contexts.setdefault(context_key, {})
        for strategy in self.STRATEGIES:
            arm = arms.setdefault(strategy, {"pulls": 0, "total_reward": 0.0})
            arm["pulls"] = int(arm.get("pulls", 0))
            arm["total_reward"] = float(arm.get("total_reward", 0.0))
        return arms

    def _sanitize_state(self, state: Any) -> dict:
        clean = self._empty_state()
        if not isinstance(state, dict):
            return clean
        contexts = state.get("contexts")
        if not isinstance(contexts, dict):
            return clean
        for context_key, arms in contexts.items():
            if not isinstance(context_key, str) or not isinstance(arms, dict):
                continue
            clean_arms = self._ensure_context(clean, context_key)
            for strategy in self.STRATEGIES:
                raw = arms.get(strategy, {})
                if not isinstance(raw, dict):
                    continue
                pulls = raw.get("pulls", 0)
                reward = raw.get("total_reward", 0.0)
                if isinstance(pulls, int) and pulls >= 0 and isinstance(reward, (int, float)) and math.isfinite(float(reward)):
                    clean_arms[strategy] = {"pulls": pulls, "total_reward": float(reward)}
        return clean

    def _empty_state(self) -> dict:
        return {"version": 1, "contexts": {}}

    def _clamp_reward(self, reward: float) -> float:
        if not math.isfinite(reward):
            return 0.0
        return max(0.0, min(1.0, reward))

    def _create_arm1_agent(self):
        from arm1 import Arm1Agent

        return self._instantiate_agent(Arm1Agent, "Arm1Agent")

    def _create_arm2_agent(self):
        from arm2 import Arm2Agent

        return self._instantiate_agent(Arm2Agent, "Arm2Agent")

    def _instantiate_agent(self, agent_cls, agent_name: str):
        signature = inspect.signature(agent_cls)
        if "project_dir" in signature.parameters:
            return agent_cls(project_dir=self.project_dir)
        return agent_cls()

    def _validate_solve_interface(self, agent, strategy_name: str) -> None:
        solve = getattr(agent, "solve", None)
        if solve is None:
            raise TypeError(f"{strategy_name} agent does not expose solve().")
        if not inspect.iscoroutinefunction(solve):
            raise TypeError(f"{strategy_name} solve() must be an async method.")
        parameters = inspect.signature(solve).parameters
        required = {"user_prompt", "case_id", "expected_algo"}
        missing = sorted(required - set(parameters))
        if missing:
            raise TypeError(f"{strategy_name} solve() missing parameter(s): {', '.join(missing)}")


async def async_main(args) -> None:
    agent = Arm3Agent()
    if args.batch:
        results = await agent.run_batch_tests(args.batch)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("Use --self-test or --batch test_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm3 Bandit Strategy Selector")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--batch", default=None)
    args = parser.parse_args()
    if args.self_test:
        agent = Arm3Agent()
        print(json.dumps(agent.self_test(), ensure_ascii=False, indent=2))
    else:
        asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
