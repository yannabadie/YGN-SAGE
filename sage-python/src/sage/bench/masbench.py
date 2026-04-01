"""MASBENCH evaluation — measures WHERE multi-agent systems add value.

Salesforce MASBench (2026) characterizes tasks along 5 axes:
  - breadth: parallel sub-tasks
  - depth: chain reasoning depth
  - horizon: multi-step planning
  - parallel: concurrent independent work
  - robustness: error tolerance

Each axis tests a different topology advantage. By comparing
bare model vs SAGE-sequential vs SAGE-full-engine, we measure
the REAL topology delta on a recognized benchmark.

Usage:
    python -m sage.bench --type masbench --axis depth --limit 20
    python -m sage.bench --type masbench_ablation --axis depth --limit 10
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sage.bench.runner import BenchReport, TaskResult

log = logging.getLogger(__name__)

AXES = ["breadth", "depth", "horizon", "parallel", "robustness"]


def _load_masbench(axis: str, split: str = "test", limit: int | None = None):
    """Load MASBENCH dataset for a given axis."""
    from datasets import load_dataset

    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}. Choose from {AXES}")

    ds = load_dataset("Salesforce/MASBench", axis, split=split)
    tasks = list(ds)[:limit] if limit else list(ds)
    log.info("Loaded %d MASBENCH tasks (axis=%s, split=%s)", len(tasks), axis, split)
    return tasks


def _parse_task(item: dict, axis: str = "") -> tuple[str, str]:
    """Extract prompt and ground truth from MASBENCH item.

    Applies axis-specific prompt engineering to guide the model.
    """
    prompt_data = json.loads(item["prompt_json"])
    reward_data = json.loads(item["reward_model_json"])

    # Prompt is a list of messages (chat format)
    if isinstance(prompt_data, list):
        question = "\n".join(
            msg.get("content", "") for msg in prompt_data if msg.get("role") == "user"
        )
    elif isinstance(prompt_data, dict):
        question = prompt_data.get("content", str(prompt_data))
    else:
        question = str(prompt_data)

    ground_truth = reward_data.get("ground_truth", "")

    # Axis-specific prompt engineering
    if axis == "robustness":
        question = (
            "Ignore all irrelevant information. Focus only on the mathematical problem.\n\n"
            + question
            + "\n\nGive your final answer as a number only."
        )
    elif axis == "horizon":
        question = (
            "Solve each sub-problem step by step. Separate answers with <<horizon>>.\n\n"
            + question
            + "\n\nGive each final answer as a number only."
        )
    elif axis in ("breadth", "parallel"):
        question += "\n\nSolve each part independently. Give your final answer as a number only."
    elif axis == "depth":
        question = (
            "Think step by step carefully. Verify each intermediate result.\n\n"
            + question
            + "\n\nGive your final answer as a number only."
        )
    else:
        question += "\n\nGive your final answer as a number only."

    return question, ground_truth


def _check_answer(response: str, ground_truth: str, axis: str = "") -> bool:
    """Check if response contains the ground truth answer.

    MASBENCH uses exact match but some tasks have multi-part answers
    separated by <<horizon>>. We check each part.
    """
    import re

    if not ground_truth:
        return False

    # Robustness: extract ALL numbers and check all GT numbers present
    if axis == "robustness":
        gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth)
        resp_nums = re.findall(r'-?\d+\.?\d*', response)
        if gt_nums:
            return all(
                any(abs(float(g) - float(r)) < 0.01 for r in resp_nums)
                for g in gt_nums
            )

    # Multi-part answers (horizon format)
    if "<<horizon>>" in ground_truth:
        parts = ground_truth.split("<<horizon>>")
        return all(part.strip() in response for part in parts if part.strip())

    # Single answer — check if it appears in the response
    gt = ground_truth.strip()
    # Try exact match first
    if gt in response:
        return True
    # Try numeric match
    try:
        gt_num = float(gt)
        numbers = re.findall(r'-?\d+\.?\d*', response)
        return any(abs(float(n) - gt_num) < 0.01 for n in numbers)
    except (ValueError, TypeError):
        return gt.lower() in response.lower()


@dataclass
class MASBenchBench:
    """Run MASBENCH evaluation on SAGE."""

    system: Any  # AgentSystem
    axis: str = "depth"

    async def run(self, limit: int | None = None) -> BenchReport:
        tasks = _load_masbench(self.axis, limit=limit)
        results: list[TaskResult] = []

        for i, item in enumerate(tasks):
            question, ground_truth = _parse_task(item, axis=self.axis)
            extra = json.loads(item.get("extra_info_json", "{}"))
            task_depth = extra.get("depth", extra.get("breadth", "?"))

            t0 = time.perf_counter()
            try:
                response = await self.system.run(question)
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = _check_answer(response, ground_truth, axis=self.axis)
                error = "" if passed else f"expected={ground_truth}"
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                response = ""
                passed = False
                error = str(e)[:200]

            log.info(
                "[%d/%d] axis=%s %s=%s passed=%s (%.1fs) gt=%s resp=%s",
                i + 1, len(tasks), self.axis,
                "depth" if "depth" in extra else "value",
                task_depth, passed, latency_ms / 1000,
                ground_truth[:50], response[:80],
            )

            results.append(TaskResult(
                task_id=f"masbench_{self.axis}_{i}",
                passed=passed,
                latency_ms=latency_ms,
                cost_usd=0.0,
                error=error,
            ))

        model_info = getattr(self.system, "model_info", {})
        return BenchReport.from_results(
            f"masbench_{self.axis}", results, model_config=model_info,
        )


@dataclass
class MASBenchAblation:
    """Ablation: bare model vs SAGE-sequential vs SAGE-full on MASBENCH."""

    system: Any
    axis: str = "depth"

    async def run(self, limit: int | None = None) -> dict[str, BenchReport]:
        tasks = _load_masbench(self.axis, limit=limit)
        reports = {}

        # Condition 1: Bare model (no pipeline, direct LLM call)
        log.info("=== Condition 1: Bare Model ===")
        bare_results = await self._run_bare(tasks)
        reports["bare"] = BenchReport.from_results(
            f"masbench_{self.axis}_bare", bare_results,
        )

        # Condition 2: SAGE with full engine
        log.info("=== Condition 2: SAGE Full Engine ===")
        full_results = await self._run_sage(tasks)
        reports["sage_full"] = BenchReport.from_results(
            f"masbench_{self.axis}_sage_full", full_results,
        )

        # Print comparison
        self._print_comparison(reports)
        return reports

    async def _run_bare(self, tasks: list) -> list[TaskResult]:
        """Run tasks with bare LLM (no SAGE pipeline)."""
        from sage.providers.connector import get_available_providers
        from sage.providers.openai_compat import OpenAICompatProvider
        import os

        # Use first available provider (connector.py = source of truth)
        available = get_available_providers()
        if not available:
            raise RuntimeError("No provider available for bare benchmark")
        cfg = available[0]
        provider = OpenAICompatProvider(
            api_key=os.environ.get(cfg["api_key_env"], ""),
            base_url=cfg["base_url"],
            provider_name=cfg["provider"],
        )
        bare_model = cfg.get("default_model", "deepseek-chat")
        log.info("Bare model: %s via %s", bare_model, cfg["provider"])

        results = []
        for i, item in enumerate(tasks):
            question, ground_truth = _parse_task(item, axis=self.axis)
            t0 = time.perf_counter()
            try:
                from sage.llm.base import Message, Role, LLMConfig
                bare_config = LLMConfig(provider=cfg["provider"], model=bare_model, max_tokens=256)
                response = await provider.generate(
                    messages=[Message(role=Role.USER, content=question)],
                    config=bare_config,
                )
                content = response.content or ""
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = _check_answer(content, ground_truth, axis=self.axis)
                error = "" if passed else f"expected={ground_truth}"
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                content = ""
                passed = False
                error = str(e)[:200]

            log.info("[bare %d/%d] passed=%s gt=%s", i + 1, len(tasks), passed, ground_truth[:30])
            results.append(TaskResult(
                task_id=f"bare_{i}", passed=passed,
                latency_ms=latency_ms, cost_usd=0.0, error=error,
            ))
        return results

    async def _run_sage(self, tasks: list) -> list[TaskResult]:
        """Run tasks with full SAGE pipeline."""
        results = []
        for i, item in enumerate(tasks):
            question, ground_truth = _parse_task(item, axis=self.axis)
            t0 = time.perf_counter()
            try:
                response = await self.system.run(question)
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = _check_answer(response, ground_truth, axis=self.axis)
                error = "" if passed else f"expected={ground_truth}"
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                passed = False
                error = str(e)[:200]

            log.info("[sage %d/%d] passed=%s gt=%s", i + 1, len(tasks), passed, ground_truth[:30])
            results.append(TaskResult(
                task_id=f"sage_{i}", passed=passed,
                latency_ms=latency_ms, cost_usd=0.0, error=error,
            ))
        return results

    def _print_comparison(self, reports: dict[str, BenchReport]):
        print(f"\n{'='*60}")
        print(f"  MASBENCH Ablation — axis={self.axis}")
        print(f"{'='*60}")
        for name, report in reports.items():
            print(f"  {name:20s}: {report.pass_rate*100:5.1f}% ({report.passed}/{report.total})")
        print(f"{'='*60}")

        # Delta
        if "bare" in reports and "sage_full" in reports:
            delta = reports["sage_full"].pass_rate - reports["bare"].pass_rate
            print(f"  Topology delta: {delta*100:+.1f}pp")
            if delta > 0:
                print(f"  >>> SAGE HELPS on {self.axis} tasks")
            elif delta < 0:
                print(f"  >>> SAGE HURTS on {self.axis} tasks")
            else:
                print(f"  >>> No difference")
        print()
