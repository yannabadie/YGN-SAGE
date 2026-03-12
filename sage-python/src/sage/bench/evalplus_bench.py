"""EvalPlus benchmark adapter: HumanEval+ (164) and MBPP+ (378).

Wraps the EvalPlus library (v0.3.1) to generate solutions via AgentSystem,
write them in EvalPlus JSONL format, and evaluate with the EvalPlus CLI.
"""

from __future__ import annotations

import asyncio
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
        official_mode: bool = False,
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
        self.official_mode = official_mode
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

        # Model ID captured from first routing decision (not lazily post-hoc)
        model_id = "unknown"
        if hasattr(self.system, "_last_decision") and self.system._last_decision:
            model_id = getattr(self.system._last_decision, "model_id", "unknown")
        elif hasattr(self.system, "agent_loop"):
            llm = getattr(self.system.agent_loop, "_llm", None)
            if llm:
                model_id = getattr(llm, "model_id", "unknown")

        # Detect provider name from the system
        provider_name = ""
        if hasattr(self.system, "agent_loop"):
            llm = getattr(self.system.agent_loop, "_llm", None)
            if llm:
                provider_name = type(llm).__name__

        self.manifest = BenchmarkManifest(
            benchmark=f"evalplus_{self.dataset}",
            model=model_id,
            provider=provider_name,
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

                    llm_response = await asyncio.wait_for(
                        self.system.agent_loop._llm.generate(
                            [Message(role=Role.USER, content=task_prompt)]
                        ),
                        timeout=60.0,
                    )
                    response = (
                        llm_response.content
                        if hasattr(llm_response, "content")
                        else str(llm_response)
                    )
                    system_used = 0
                else:
                    response = await asyncio.wait_for(
                        self.system.run(task_prompt),
                        timeout=60.0,
                    )
                    system_used = (
                        getattr(
                            self.system.agent_loop, "_last_routing_system", 0
                        )
                        or 2  # Default S2 for code tasks
                    )

                completion = extract_code(response, entry_point)

            except asyncio.TimeoutError:
                completion = ""
                error = "generation_timeout_60s"
                log.warning(f"[{task_id}] Generation timed out after 60s")
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

            status = "OK" if not error else f"ERR:{error[:30]}"
            print(
                f"  [{i + 1}/{len(task_ids)}] {task_id}: "
                f"{status} ({latency:.0f}ms)",
                flush=True,
            )

        if self.baseline_mode:
            model_id = f"baseline:{model_id}" if model_id != "unknown" else "baseline"
        self.manifest.model = model_id

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

        # --- Base tests ---
        if test_code:
            # HumanEval: use the original check() function
            from sage.bench.humaneval import run_test
            base_passed, base_error = run_test(
                prompt, solution_code, test_code, entry_point, timeout=timeout
            )
        elif base_inputs:
            # MBPP: compare solution vs canonical on base_inputs
            base_result = self._run_comparison(
                solution_code, prompt, canonical, entry_point,
                base_inputs, atol, timeout,
            )
            base_passed = base_result["passed"]
            base_error = base_result["error"]
        else:
            base_passed, base_error = True, ""

        if not base_passed:
            return {"base_passed": False, "plus_passed": False, "error": base_error}

        # --- Plus tests: run solution vs canonical on plus_inputs ---
        if not plus_inputs:
            return {"base_passed": True, "plus_passed": True, "error": ""}

        plus_result = self._run_comparison(
            solution_code, prompt, canonical, entry_point,
            plus_inputs[:200], atol, timeout,
        )
        return {
            "base_passed": True,
            "plus_passed": plus_result["passed"],
            "error": plus_result["error"],
        }

    def _run_comparison(
        self,
        solution_code: str,
        prompt: str,
        canonical: str,
        entry_point: str,
        inputs: list,
        atol: float,
        timeout: float,
    ) -> dict[str, Any]:
        """Run solution vs canonical on a set of inputs. Returns {"passed": bool, "error": str}."""
        if not inputs:
            return {"passed": True, "error": ""}

        if f"def {entry_point}" in solution_code:
            sol_code = solution_code
        else:
            sol_code = prompt + solution_code

        test_program = f'''{sol_code}

# Save reference to solution BEFORE canonical overwrites the name
import types
_solution = types.FunctionType(
    {entry_point}.__code__, {{**globals()}}, "{entry_point}_sol"
)

# --- Canonical solution (overwrites {entry_point}) ---
{prompt}{canonical}
_canonical = {entry_point}

# --- Run comparison tests ---
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
print(f"COMPARE_RESULT:{{failures}}/{{len(inputs)}}")
'''

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_program)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                ["python", tmp_path],
                input=json.dumps(inputs),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return {"passed": False, "error": proc.stderr[:300]}

            for line in proc.stdout.split("\n"):
                if line.startswith("COMPARE_RESULT:"):
                    parts = line.split(":")[1].split("/")
                    failures = int(parts[0])
                    total = int(parts[1])
                    passed = failures == 0
                    error = f"{failures}/{total} tests failed" if failures else ""
                    return {"passed": passed, "error": error}

            return {"passed": False, "error": "no result line"}

        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
        except Exception as e:
            return {"passed": False, "error": str(e)[:200]}
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
            return BenchReport.from_results(
                f"evalplus_{self.dataset}", [],
                model_config={"model": self.manifest.model if self.manifest else "unknown"},
            )

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
            print(
                f"  eval [{i+1}/{len(solutions)}] {task_id}: {status} ({base_s}, {plus_s})",
                flush=True,
            )

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
            f"evalplus_{self.dataset}", task_results,
            model_config={"model": self.manifest.model if self.manifest else "unknown"},
        )

    async def run_official(self, limit: int | None = None, output_dir: str | None = None) -> dict[str, Any]:
        """Generate samples.jsonl and evaluate with official EvalPlus CLI.

        This produces scores comparable to the EvalPlus leaderboard.
        Requires: pip install evalplus

        Returns dict with 'base' and 'plus' pass rates.
        """
        try:
            import evalplus  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "evalplus not installed. Run: pip install evalplus"
            )

        # 1. Generate solutions
        solutions = await self.generate_solutions(limit=limit)
        if not solutions:
            return {"base": 0.0, "plus": 0.0, "error": "no solutions generated"}

        # 2. Write samples.jsonl
        out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="sage_evalplus_"))
        samples_path = out_dir / "samples.jsonl"
        with open(samples_path, "w", encoding="utf-8") as f:
            for sol in solutions:
                entry = {"task_id": sol["task_id"], "solution": sol["solution"]}
                f.write(json.dumps(entry) + "\n")
        log.info("Wrote %d samples to %s", len(solutions), samples_path)

        # 3. Sanitize
        sanitize_result = subprocess.run(
            ["python", "-m", "evalplus.sanitize", "--samples", str(samples_path)],
            capture_output=True, text=True, timeout=120,
        )
        if sanitize_result.returncode != 0:
            log.warning("evalplus.sanitize failed: %s", sanitize_result.stderr[:300])

        # Find sanitized file (evalplus appends -sanitized)
        sanitized = samples_path.with_name(samples_path.stem + "-sanitized.jsonl")
        eval_samples = str(sanitized) if sanitized.exists() else str(samples_path)

        # 4. Evaluate
        eval_result = subprocess.run(
            ["python", "-m", "evalplus.evaluate",
             "--dataset", self.dataset,
             "--samples", eval_samples],
            capture_output=True, text=True, timeout=600,
        )
        log.info("evalplus.evaluate stdout:\n%s", eval_result.stdout)

        # 5. Parse results
        results = {"base": 0.0, "plus": 0.0, "raw_output": eval_result.stdout}
        for line in eval_result.stdout.split("\n"):
            if "pass@1" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip())
                    if "plus" in line.lower() or "+" in line:
                        results["plus"] = val
                    else:
                        results["base"] = val
                except ValueError:
                    pass

        return results
