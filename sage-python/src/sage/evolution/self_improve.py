"""Self-improvement loop: benchmark -> diagnose -> evolve -> re-benchmark."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

log = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    """Record of one improvement iteration."""

    cycle: int
    before_score: float
    after_score: float
    changes: list[str]
    improved: bool


class SelfImprovementLoop:
    """Runs benchmark -> diagnose failures -> propose changes -> re-benchmark.

    Accepts three async callables:
      * ``benchmark_fn``  -- returns an object with ``.pass_rate`` and ``.results``
      * ``diagnose_fn``   -- accepts list of failed results, returns list of diagnosis strings
      * ``evolve_fn``     -- accepts diagnosis list, returns list of change descriptions

    After each cycle an :class:`ImprovementCycle` is recorded in ``history``.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        registry: Any = None,
        evolution_engine: Any = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.registry = registry
        self.evolution = evolution_engine
        self.history: list[ImprovementCycle] = []

    async def run_cycle(
        self,
        benchmark_fn: Callable[..., Awaitable[Any]],
        diagnose_fn: Callable[..., Awaitable[Any]],
        evolve_fn: Callable[..., Awaitable[Any]],
    ) -> ImprovementCycle:
        """Execute one improvement cycle."""
        cycle_num = len(self.history) + 1
        log.info("Self-improvement cycle %d starting", cycle_num)

        # 1. Benchmark current state
        before = await benchmark_fn()
        before_score = before.pass_rate if hasattr(before, "pass_rate") else 0.0

        # 2. Diagnose failures
        failures = [
            r
            for r in (before.results if hasattr(before, "results") else [])
            if not r.passed
        ]
        diagnosis = await diagnose_fn(failures) if failures else []

        # 3. Evolve (apply changes)
        changes = await evolve_fn(diagnosis) if diagnosis else []

        # 4. Re-benchmark
        after = await benchmark_fn()
        after_score = after.pass_rate if hasattr(after, "pass_rate") else 0.0

        cycle = ImprovementCycle(
            cycle=cycle_num,
            before_score=before_score,
            after_score=after_score,
            changes=changes,
            improved=after_score > before_score,
        )
        self.history.append(cycle)
        log.info(
            "Cycle %d: %.1f%% -> %.1f%% (%s)",
            cycle_num,
            before_score * 100,
            after_score * 100,
            "improved" if cycle.improved else "no change",
        )
        return cycle

    def improvement_rate(self) -> float:
        """Proportion of cycles that improved (0.0 if no history)."""
        if not self.history:
            return 0.0
        improved = sum(1 for c in self.history if c.improved)
        return improved / len(self.history)
