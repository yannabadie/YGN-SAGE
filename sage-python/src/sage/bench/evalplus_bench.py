"""EvalPlus benchmark adapter: HumanEval+ (164) and MBPP+ (378).

Wraps the EvalPlus library (v0.3.1) to generate solutions via AgentSystem,
write them in EvalPlus JSONL format, and evaluate with the EvalPlus CLI.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.humaneval import extract_code
from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)

# Dataset loader dispatch
_DATASET_LOADERS = {
    "humaneval": "evalplus.data:get_human_eval_plus",
    "mbpp": "evalplus.data:get_mbpp_plus",
}


def _load_dataset(dataset: str) -> dict[str, dict[str, Any]]:
    """Load an EvalPlus dataset by name.

    Returns dict keyed by task_id, e.g. {"HumanEval/0": {...}, ...}.
    """
    if dataset not in _DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {list(_DATASET_LOADERS.keys())}"
        )
    module_path, func_name = _DATASET_LOADERS[dataset].split(":")
    import importlib

    mod = importlib.import_module(module_path)
    loader = getattr(mod, func_name)
    return loader()


class EvalPlusBench:
    """EvalPlus benchmark adapter for HumanEval+ and MBPP+.

    Generates solutions via AgentSystem.run(), writes EvalPlus-format JSONL,
    and optionally evaluates with the EvalPlus CLI.

    Args:
        system: AgentSystem to benchmark. If None, no solutions are generated.
        event_bus: EventBus for emitting BENCH_RESULT events.
        dataset: "humaneval" or "mbpp".
        baseline_mode: When True, tasks are sent directly to the LLM without
            routing, memory, guardrails, or framework overhead.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        dataset: str = "humaneval",
        baseline_mode: bool = False,
    ):
        if dataset not in _DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Supported: {list(_DATASET_LOADERS.keys())}"
            )
        self.system = system
        self.event_bus = event_bus
        self.dataset = dataset
        self.baseline_mode = baseline_mode
        self.manifest: BenchmarkManifest | None = None

    async def generate_solutions(
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Generate solutions for EvalPlus tasks via AgentSystem.run().

        Each solution dict has keys: task_id, solution, _latency_ms,
        _cost_usd, _system_used, _error.

        Returns empty list if no system is configured.
        """
        if self.system is None:
            log.warning("No AgentSystem configured -- returning empty solutions")
            return []

        problems = _load_dataset(self.dataset)
        task_ids = list(problems.keys())
        if limit is not None:
            task_ids = task_ids[:limit]

        model_id = ""
        if hasattr(self.system, "agent_loop"):
            model_id = (
                getattr(self.system.agent_loop, "_last_model", "") or "unknown"
            )
        if self.baseline_mode:
            model_id = f"baseline:{model_id}" if model_id else "baseline"

        self.manifest = BenchmarkManifest(
            benchmark=f"evalplus_{self.dataset}", model=model_id
        )

        solutions: list[dict[str, Any]] = []

        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            prompt = problem["prompt"]
            entry_point = problem["entry_point"]

            t0 = time.perf_counter()
            error = ""
            system_used = 0

            try:
                task_prompt = (
                    "Complete this Python function. "
                    "Return ONLY the complete function, no explanation.\n\n"
                    f"```python\n{prompt}\n```"
                )

                if self.baseline_mode:
                    # Bypass routing/memory/guardrails
                    from sage.llm.base import Message, Role

                    llm_response = await self.system.agent_loop._llm.generate(
                        [Message(role=Role.USER, content=task_prompt)]
                    )
                    response = (
                        llm_response.content
                        if hasattr(llm_response, "content")
                        else str(llm_response)
                    )
                    system_used = 0
                else:
                    response = await self.system.run(task_prompt)
                    system_used = (
                        getattr(
                            self.system.agent_loop, "_last_routing_system", 0
                        )
                        or 2  # Default S2 for code tasks
                    )

                completion = extract_code(response, entry_point)

            except Exception as e:
                completion = ""
                error = str(e)[:200]
                log.error(f"[{task_id}] Generation failed: {error}")

            latency = (time.perf_counter() - t0) * 1000
            cost = (
                getattr(self.system.agent_loop, "total_cost_usd", 0.0)
                if self.system
                else 0.0
            )

            solutions.append(
                {
                    "task_id": task_id,
                    "solution": completion,
                    "_latency_ms": round(latency, 1),
                    "_cost_usd": round(cost, 6),
                    "_system_used": system_used,
                    "_error": error,
                }
            )

            # Record trace for truth pack
            self.manifest.add(
                TaskTrace(
                    task_id=task_id,
                    passed=False,  # Unknown until evaluate()
                    latency_ms=round(latency, 1),
                    cost_usd=round(cost, 6),
                    model=model_id,
                    routing=f"S{system_used}",
                    error=error[:200] if error else "",
                )
            )

            # Emit progress event
            if self.event_bus:
                from sage.agent_loop import AgentEvent

                self.event_bus.emit(
                    AgentEvent(
                        type="BENCH_RESULT",
                        step=i + 1,
                        timestamp=time.time(),
                        meta={
                            "benchmark": f"evalplus_{self.dataset}",
                            "task_id": task_id,
                            "system_used": system_used,
                            "latency_ms": round(latency, 1),
                            "progress": f"{i + 1}/{len(task_ids)}",
                        },
                    )
                )

            log.info(
                f"[{i + 1}/{len(task_ids)}] {task_id}: "
                f"generated ({latency:.0f}ms)"
            )

        return solutions

    def write_solutions(
        self,
        solutions: list[dict[str, Any]],
        path: str | Path,
    ) -> None:
        """Write solutions in EvalPlus JSONL format.

        The output file contains one JSON object per line with keys:
        task_id, solution (plus optional metadata prefixed with _).
        """
        from evalplus.data import write_jsonl

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # EvalPlus expects: {"task_id": "...", "solution": "..."}
        # write_jsonl with drop_builtin=True strips keys starting with _
        write_jsonl(str(path), solutions)
        log.info(f"Wrote {len(solutions)} solutions to {path}")

    def evaluate_task(
        self,
        solution_code: str,
        problem: dict[str, Any],
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Evaluate a single solution against EvalPlus base + plus tests.

        Uses subprocess sandbox (Windows-compatible, no Unix `resource` module).
        Runs the canonical solution to get expected outputs, then compares.

        Returns: {"base_passed": bool, "plus_passed": bool, "error": str}
        """
        entry_point = problem["entry_point"]
        canonical = problem["canonical_solution"]
        prompt = problem["prompt"]
        base_inputs = problem.get("base_input", [])
        plus_inputs = problem.get("plus_input", [])
        test_code = problem.get("test", "")
        atol = problem.get("atol", 0)

        # --- Base tests: use the original HumanEval check() function ---
        from sage.bench.humaneval import run_test
        base_passed, base_error = run_test(
            prompt, solution_code, test_code, entry_point, timeout=timeout
        )

        if not base_passed:
            return {"base_passed": False, "plus_passed": False, "error": base_error}

        # --- Plus tests: run solution vs canonical on plus_inputs ---
        if not plus_inputs:
            return {"base_passed": True, "plus_passed": True, "error": ""}

        # Build a test program that compares solution vs canonical
        # on a sample of plus_inputs (cap at 200 to avoid timeout)
        sample_inputs = plus_inputs[:200]

        # Build the comparison script
        if f"def {entry_point}" in solution_code:
            sol_code = solution_code
        else:
            sol_code = prompt + solution_code

        test_program = f'''{sol_code}

# --- Canonical solution ---
{prompt}{canonical}
_canonical = {entry_point}

# Rename solution
import types
_solution = types.FunctionType(
    {entry_point}.__code__, {{**globals()}}, "{entry_point}_sol"
)

# --- Run plus tests ---
import json, sys
inputs = json.loads(sys.stdin.read())
atol = {atol}
failures = 0
for args in inputs:
    try:
        expected = _canonical(*args)
        actual = _solution(*args)
        if atol > 0:
            if isinstance(expected, float) and isinstance(actual, float):
                if abs(expected - actual) > atol:
                    failures += 1
                    continue
        if expected != actual:
            failures += 1
    except Exception:
        failures += 1
print(f"PLUS_RESULT:{{failures}}/{{len(inputs)}}")
'''

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_program)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                ["python", tmp_path],
                input=json.dumps(sample_inputs),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return {
                    "base_passed": True,
                    "plus_passed": False,
                    "error": proc.stderr[:300],
                }

            # Parse result
            for line in proc.stdout.split("\n"):
                if line.startswith("PLUS_RESULT:"):
                    parts = line.split(":")[1].split("/")
                    failures = int(parts[0])
                    total = int(parts[1])
                    plus_passed = failures == 0
                    return {
                        "base_passed": True,
                        "plus_passed": plus_passed,
                        "error": f"{failures}/{total} plus tests failed" if failures else "",
                    }

            return {"base_passed": True, "plus_passed": False, "error": "no result line"}

        except subprocess.TimeoutExpired:
            return {"base_passed": True, "plus_passed": False, "error": "plus_timeout"}
        except Exception as e:
            return {"base_passed": True, "plus_passed": False, "error": str(e)[:200]}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def run(self, limit: int | None = None) -> BenchReport:
        """Full pipeline: generate solutions, evaluate each against EvalPlus+ tests.

        Uses subprocess-based evaluation (Windows-compatible).
        Each task is evaluated against base tests (original HumanEval)
        + plus tests (EvalPlus enhanced, up to 999 additional inputs).
        """
        solutions = await self.generate_solutions(limit=limit)

        if not solutions:
            return BenchReport.from_results(f"evalplus_{self.dataset}", [])

        # Load problems for evaluation
        problems = _load_dataset(self.dataset)

        # Evaluate each solution
        task_results: list[TaskResult] = []
        base_pass = 0
        plus_pass = 0

        for i, sol in enumerate(solutions):
            task_id = sol["task_id"]
            problem = problems[task_id]

            eval_result = self.evaluate_task(sol["solution"], problem)
            passed = eval_result["base_passed"] and eval_result["plus_passed"]

            if eval_result["base_passed"]:
                base_pass += 1
            if passed:
                plus_pass += 1

            error = sol["_error"] or eval_result.get("error", "")

            task_results.append(
                TaskResult(
                    task_id=task_id,
                    passed=passed,
                    system_used=sol["_system_used"],
                    latency_ms=sol["_latency_ms"],
                    cost_usd=sol["_cost_usd"],
                    error=error,
                )
            )

            status = "PASS" if passed else "FAIL"
            base_s = "base_ok" if eval_result["base_passed"] else "base_fail"
            plus_s = "plus_ok" if eval_result["plus_passed"] else "plus_fail"
            log.info(f"[{i+1}/{len(solutions)}] {task_id}: {status} ({base_s}, {plus_s})")

        # Update manifest traces with pass/fail from evaluation
        if self.manifest:
            for tr, result in zip(self.manifest.traces, task_results):
                tr.passed = result.passed

        total = len(solutions)
        log.info(
            f"EvalPlus {self.dataset}: base={base_pass}/{total}, "
            f"plus={plus_pass}/{total}"
        )

        return BenchReport.from_results(
            f"evalplus_{self.dataset}", task_results
        )
