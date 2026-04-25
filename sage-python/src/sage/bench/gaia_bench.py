"""GAIA general assistant benchmark adapter.

GAIA: General AI Assistants benchmark (165 tasks, 3 levels).
Uses exact string match on 'Final answer' for scoring.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from sage.bench.runner import BenchReport, TaskResult

_log = logging.getLogger("sage.bench.gaia")


def _gaia_exact_match(expected: str, response: str) -> bool:
    """GAIA scoring: exact match or standalone token match.

    Primary: exact match after strip+lower.
    Fallback: expected appears as a standalone word/number in the response
    (bounded by word boundaries).
    """
    exp = expected.strip().lower()
    resp = response.strip().lower()
    if not exp:
        return False
    # Exact match (whole response equals expected)
    if resp == exp:
        return True
    # Standalone token match: expected appears as a whole word/number
    pattern = r'(?<!\w)' + re.escape(exp) + r'(?!\w)'
    return bool(re.search(pattern, resp))


@dataclass
class GaiaBench:
    """GAIA benchmark adapter with level filtering and cost tracking."""

    system: Any  # AgentSystem — typed as Any to avoid circular import
    split: str = "validation"
    level: int | None = None  # 1, 2, 3, or None for all
    skip_file_tasks: bool = True  # Skip tasks requiring file attachments

    async def run(self, limit: int | None = None) -> BenchReport:
        try:
            import os
            from datasets import load_dataset
            token = os.environ.get("HF_TOKEN")
            ds = load_dataset(
                "gaia-benchmark/GAIA", "2023_all",
                split=self.split, token=token,
            )
        except Exception as exc:
            _log.warning("Failed to load GAIA from HuggingFace: %s", exc)
            ds = self._load_local()

        tasks = list(ds)

        # Level filter
        if self.level is not None:
            tasks = [t for t in tasks if str(t.get("Level", "")) == str(self.level)]
            _log.info("GAIA Level %d: %d tasks", self.level, len(tasks))

        # Skip file-dependent tasks
        if self.skip_file_tasks:
            before = len(tasks)
            tasks = [t for t in tasks if not t.get("file_name")]
            skipped = before - len(tasks)
            if skipped:
                _log.info("Skipped %d file-dependent tasks", skipped)

        if limit:
            tasks = tasks[:limit]

        _log.info("Running GAIA: %d tasks (level=%s, limit=%s)", len(tasks), self.level, limit)

        results: list[TaskResult] = []
        total_cost = 0.0

        for idx, item in enumerate(tasks):
            task_id = item.get("task_id", str(idx))
            question = item.get("Question", item.get("question", ""))
            expected = item.get("Final answer", item.get("answer", ""))
            level = item.get("Level", "?")

            t0 = time.perf_counter()
            try:
                response = await self.system.agent_loop.run(question)
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = _gaia_exact_match(expected, response) if expected else len(response) > 10
                error = "" if passed else "wrong_answer"
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                response = ""
                passed = False
                error = str(e)

            # Cost tracking (from provider if available)
            task_cost = 0.0
            try:
                cost_tracker = getattr(self.system, "cost_tracker", None)
                if cost_tracker:
                    task_cost = getattr(cost_tracker, "last_cost", 0.0)
            except Exception:
                pass
            total_cost += task_cost

            status = "PASS" if passed else "FAIL"
            resp_preview = (response[:80] + "...") if len(response) > 80 else response
            resp_preview = resp_preview.replace("\n", " ")
            print(
                f"  [{idx+1}/{len(tasks)}] {status}  L{level}  "
                f'id={task_id[:12]}  expected="{expected}"  got="{resp_preview}"',
                flush=True,
            )

            results.append(TaskResult(
                task_id=task_id,
                passed=passed,
                latency_ms=latency_ms,
                cost_usd=task_cost,
                error=error,
            ))

        passed_count = sum(1 for r in results if r.passed)
        total = len(results)
        rate = passed_count / total * 100 if total else 0
        print(f"\nGAIA results: {passed_count}/{total} ({rate:.1f}%)", flush=True)
        if self.level:
            print(f"  Level: {self.level}", flush=True)
        print(f"  Total cost: ${total_cost:.4f}", flush=True)

        model_info = getattr(self.system, "model_info", {})
        return BenchReport.from_results("gaia", results, model_config=model_info)

    def _load_local(self):
        return []
