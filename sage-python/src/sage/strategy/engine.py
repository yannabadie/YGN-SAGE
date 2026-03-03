"""Strategy engine: combines solvers with resource allocation.

Orchestrates the selection and weighting of different agent approaches
using game-theoretic principles, then allocates resources accordingly.
"""
from __future__ import annotations

from typing import Any

from sage.strategy.solvers import RegretMatcher, PRDSolver
from sage.strategy.allocator import ResourceAllocator, Allocation


class StrategyEngine:
    """High-level strategy engine for multi-approach optimization."""

    def __init__(
        self,
        strategy_names: list[str],
        solver_type: str = "regret",
        allocator: ResourceAllocator | None = None,
    ):
        self.strategy_names = strategy_names
        n = len(strategy_names)

        if solver_type == "regret":
            self._solver = RegretMatcher(n)
        elif solver_type == "prd":
            self._solver = PRDSolver(n)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        self._allocator = allocator or ResourceAllocator()
        self._history: list[dict[str, Any]] = []

    def get_allocations(self) -> list[Allocation]:
        """Get current resource allocations based on strategy weights."""
        weights = self._solver.get_strategy()
        return self._allocator.allocate(self.strategy_names, weights)

    def report_outcome(self, strategy_index: int, outcomes: list[float]) -> None:
        """Report the outcome of a round.

        Args:
            strategy_index: Which strategy was actually used
            outcomes: Performance score for each strategy this round
        """
        if isinstance(self._solver, RegretMatcher):
            self._solver.update(outcomes, strategy_index)
        elif isinstance(self._solver, PRDSolver):
            self._solver.update(outcomes)

        self._history.append({
            "chosen": strategy_index,
            "outcomes": outcomes,
            "strategy": self._solver.get_strategy(),
        })

    def get_strategy(self) -> list[float]:
        """Get current strategy weights."""
        return self._solver.get_strategy()

    def stats(self) -> dict[str, Any]:
        """Get strategy engine statistics."""
        strategy = self._solver.get_strategy()
        return {
            "strategy_names": self.strategy_names,
            "weights": strategy,
            "rounds": len(self._history),
            "dominant": self.strategy_names[strategy.index(max(strategy))],
        }
