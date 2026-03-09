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

    def evaluate(self, samples_path: str | Path) -> dict[str, Any]:
        """Run EvalPlus CLI evaluator on a solutions JSONL file.

        Returns parsed results dict with keys: pass@1, plus_pass@1, etc.
        Raises RuntimeError if the CLI fails.
        """
        samples_path = Path(samples_path)
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")

        # EvalPlus writes results to <samples>_eval_results.json
        result_path = str(samples_path).replace(".jsonl", "_eval_results.json")

        cmd = [
            "python",
            "-m",
            "evalplus.evaluate",
            self.dataset,
            "--samples",
            str(samples_path),
            "--i-just-wanna-run",
        ]

        log.info(f"Running EvalPlus evaluator: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("EvalPlus evaluation timed out after 600s")

        if proc.returncode != 0:
            raise RuntimeError(
                f"EvalPlus evaluation failed (rc={proc.returncode}):\n"
                f"{proc.stderr[:1000]}"
            )

        # Parse output for pass@k scores
        results: dict[str, Any] = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

        # Extract pass@k from CLI output lines like "pass@1:\t0.850"
        import re

        for line in proc.stdout.split("\n"):
            match = re.match(r"\s*(pass@\d+):\s+([0-9.]+)", line)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                results[key] = value

        # Also load the detailed eval_results.json if it exists
        if Path(result_path).exists():
            with open(result_path, encoding="utf-8") as f:
                results["eval_details"] = json.load(f)

        return results

    async def run(self, limit: int | None = None) -> BenchReport:
        """Full pipeline: generate -> write -> evaluate -> report.

        If no system is configured, returns an empty report.
        """
        solutions = await self.generate_solutions(limit=limit)

        if not solutions:
            return BenchReport.from_results(f"evalplus_{self.dataset}", [])

        # Write solutions to temp file and evaluate
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / f"{self.dataset}_solutions.jsonl"
            self.write_solutions(solutions, samples_path)

            eval_results: dict[str, Any] = {}
            try:
                eval_results = self.evaluate(samples_path)
            except Exception as e:
                log.error(f"EvalPlus evaluation failed: {e}")

        # Build TaskResult list from solutions + eval results
        task_results: list[TaskResult] = []
        eval_details = eval_results.get("eval_details", {}).get("eval", {})

        for sol in solutions:
            task_id = sol["task_id"]

            # Check if this task passed in eval_details
            # eval_details[task_id] is a list of completion results
            passed = False
            if task_id in eval_details:
                completions = eval_details[task_id]
                if completions:
                    # For pass@1 with single completion, check first result
                    first = completions[0]
                    if isinstance(first, dict):
                        passed = (
                            first.get("base_status", "") == "pass"
                            and first.get("plus_status", "") == "pass"
                        )
                    elif isinstance(first, list):
                        # Older format: list of [base_status, plus_status]
                        passed = all(s == "pass" for s in first)

            task_results.append(
                TaskResult(
                    task_id=task_id,
                    passed=passed,
                    system_used=sol["_system_used"],
                    latency_ms=sol["_latency_ms"],
                    cost_usd=sol["_cost_usd"],
                    error=sol["_error"],
                )
            )

        # Update manifest traces with pass/fail from evaluation
        if self.manifest:
            for tr, result in zip(self.manifest.traces, task_results):
                tr.passed = result.passed

        return BenchReport.from_results(
            f"evalplus_{self.dataset}", task_results
        )
