"""BigCodeBench adapter: 1140 real-world coding tasks (ICLR '25).

Wraps the bigcodebench package to generate solutions via AgentSystem,
evaluate locally with unittest subprocess, and optionally run official CLI.

Install: pip install bigcodebench
Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sage.bench.humaneval import extract_code
from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)


def _load_dataset(subset: str = "full") -> dict[str, dict[str, Any]]:
    """Load BigCodeBench dataset.

    Args:
        subset: "full" (1140 tasks) or "hard" (~150 tasks).
    """
    from bigcodebench.data import get_bigcodebench
    return get_bigcodebench(subset=subset)


class BigCodeBenchBench:
    """BigCodeBench adapter for SAGE.

    Args:
        system: AgentSystem to benchmark.
        event_bus: EventBus for BENCH_RESULT events.
        subset: "full" (1140) or "hard" (~150).
        split: "instruct" (NL) or "complete" (docstring).
        task_timeout: Max seconds for LLM generation.
        eval_timeout: Max seconds for test evaluation.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        subset: str = "full",
        split: str = "instruct",
        task_timeout: float = 120.0,
        eval_timeout: float = 30.0,
    ):
        self.system = system
        self.event_bus = event_bus
        self.subset = subset
        self.split = split
        self.task_timeout = task_timeout
        self.eval_timeout = eval_timeout

    async def run(self, limit: int | None = None) -> BenchReport:
        """Run BigCodeBench benchmark."""
        problems = _load_dataset(self.subset)
        task_ids = list(problems.keys())
        if limit:
            task_ids = task_ids[:limit]

        results: list[TaskResult] = []
        passed_count = 0

        for i, task_id in enumerate(task_ids):
            task = problems[task_id]
            prompt_key = "instruct_prompt" if self.split == "instruct" else "complete_prompt"
            prompt = task.get(prompt_key, task.get("instruct_prompt", ""))

            t0 = time.time()
            solution = ""
            error = ""

            if self.system:
                try:
                    raw = await asyncio.wait_for(
                        self.system.run(prompt),
                        timeout=self.task_timeout,
                    )
                    solution = extract_code(raw, task["entry_point"])
                except asyncio.TimeoutError:
                    error = "TIMEOUT"
                except Exception as exc:
                    error = str(exc)[:200]

            latency_ms = (time.time() - t0) * 1000

            if solution and not error:
                task_passed = self._evaluate_solution(
                    solution=solution,
                    test_code=task["test"],
                    entry_point=task["entry_point"],
                    task_id=task_id,
                    timeout=self.eval_timeout,
                )
            else:
                task_passed = False
                if not error:
                    error = "no solution generated"

            if task_passed:
                passed_count += 1

            results.append(TaskResult(
                task_id=task_id,
                passed=task_passed,
                latency_ms=latency_ms,
                error=error,
            ))

            status = "PASS" if task_passed else "FAIL"
            log.info("[%d/%d] %s %s (%.0fms)", i + 1, len(task_ids), status, task_id, latency_ms)

        total = len(results)
        return BenchReport(
            benchmark=f"bigcodebench-{self.subset}-{self.split}",
            total=total,
            passed=passed_count,
            failed=total - passed_count,
            errors=sum(1 for r in results if r.error),
            pass_rate=passed_count / total if total else 0.0,
            avg_latency_ms=sum(r.latency_ms for r in results) / total if total else 0.0,
            avg_cost_usd=0.0,
            routing_breakdown={},
            results=results,
        )

    @staticmethod
    def _evaluate_solution(
        solution: str,
        test_code: str,
        entry_point: str,
        task_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Evaluate by running unittest test cases in subprocess."""
        script = f"""{solution}

{test_code}

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=0)
"""
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                f.flush()
                tmp_path = f.name

            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            log.debug("Eval timeout for %s", task_id)
            return False
        except Exception as exc:
            log.debug("Eval error for %s: %s", task_id, exc)
            return False
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
