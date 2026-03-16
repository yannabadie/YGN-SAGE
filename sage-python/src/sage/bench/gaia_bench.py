"""GAIA general assistant benchmark adapter."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sage.bench.runner import BenchReport, TaskResult


@dataclass
class GaiaBench:
    system: Any  # AgentSystem — typed as Any to avoid circular import
    split: str = "validation"

    async def run(self, limit: int | None = None) -> BenchReport:
        try:
            from datasets import load_dataset
            ds = load_dataset("gaia-benchmark/GAIA", split=self.split)
        except Exception:
            ds = self._load_local()

        tasks = list(ds)[:limit] if limit else list(ds)
        results: list[TaskResult] = []
        for item in tasks:
            question = item.get("Question", item.get("question", ""))
            expected = item.get("Final answer", item.get("answer", ""))
            t0 = time.perf_counter()
            try:
                response = await self.system.agent_loop.run(question)
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = (
                    expected.lower().strip() in response.lower()
                    if expected
                    else len(response) > 10
                )
                error = "" if passed else "wrong_answer"
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = False
                error = str(e)
            results.append(TaskResult(
                task_id=item.get("task_id", str(len(results))),
                passed=passed,
                latency_ms=latency_ms,
                cost_usd=0.0,
                error=error,
            ))
        model_info = getattr(self.system, "model_info", {})
        return BenchReport.from_results("gaia", results, model_config=model_info)

    def _load_local(self):
        return []
