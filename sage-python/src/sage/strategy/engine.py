"""Strategy engine: combines solvers with resource allocation.

Orchestrates the selection and weighting of different agent approaches
using game-theoretic principles, then allocates resources accordingly.
"""
from __future__ import annotations

from typing import Any, List
import numpy as np

from sage.strategy.solvers import RegretMatcher, SAMPOSolver, VolatilityAdaptiveSolver, SHORPSROSolver
from sage.strategy.allocator import ResourceAllocator, Allocation


class StrategyEngine:
    """High-level strategy engine for multi-approach optimization."""

    def __init__(
        self,
        strategy_names: list[str],
        solver_type: str = "regret",
        allocator: ResourceAllocator | None = None,
        **solver_kwargs: Any,
    ):
        self.strategy_names = strategy_names
        n = len(strategy_names)

        if solver_type == "regret":
            self._solver = RegretMatcher(n)
        elif solver_type == "sampo":
            self._solver = SAMPOSolver(n, **solver_kwargs)
        elif solver_type == "vad_cfr":
            self._solver = VolatilityAdaptiveSolver(n, **solver_kwargs)
        elif solver_type == "shor_psro":
            self._solver = SHORPSROSolver(n, **solver_kwargs)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        self._allocator = allocator or ResourceAllocator()
        self._history: list[dict[str, Any]] = []

    def get_allocations(self, payoffs: List[float] | None = None) -> list[Allocation]:
        """Get current resource allocations based on strategy weights."""
        weights = self.get_strategy(payoffs)
        return self._allocator.allocate(self.strategy_names, weights.tolist())

    def report_outcome(self, strategy_index: int, outcomes: List[float]) -> None:
        """Report the outcome of a round.

        Args:
            strategy_index: Which strategy was actually used
            outcomes: Performance score for each strategy this round (payoffs)
        """
        if isinstance(self._solver, (RegretMatcher, VolatilityAdaptiveSolver, SHORPSROSolver)):
            self._solver.update(outcomes, strategy_index)
        elif isinstance(self._solver, SAMPOSolver):
            # Convert single round to a trajectory for SAMPO
            traj = [{"actions": [strategy_index], "rewards": outcomes}]
            self._solver.update(traj)

        self._history.append({
            "chosen": strategy_index,
            "outcomes": outcomes,
            "strategy": self.get_strategy(outcomes).tolist(),
        })

    def get_strategy(self, payoffs: List[float] | None = None) -> np.ndarray:
        """Get current strategy weights.
        
        Note: SHOR-PSRO requires payoffs to compute the hybrid strategy.
        """
        if isinstance(self._solver, SHORPSROSolver):
            if payoffs is None:
                # If no payoffs provided, we use the last outcomes or uniform
                if self._history:
                    payoffs = self._history[-1]["outcomes"]
                else:
                    return np.ones(len(self.strategy_names)) / len(self.strategy_names)
            return self._solver.get_strategy(payoffs)
        
        return self._solver.get_strategy()

    def stats(self) -> dict[str, Any]:
        """Get strategy engine statistics."""
        strategy = self.get_strategy().tolist()
        return {
            "strategy_names": self.strategy_names,
            "weights": strategy,
            "rounds": len(self._history),
            "dominant": self.strategy_names[np.argmax(strategy)],
        }
