"""Resource allocator: distributes compute budget across strategies.

Uses the strategy solver outputs to allocate resources (tokens, time,
agents) across different approaches proportionally to their strategic weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Allocation:
    """Resource allocation for a single strategy."""
    strategy_name: str
    weight: float  # Probability/weight from the solver
    tokens: int = 0
    agents: int = 0
    time_budget: float = 0.0  # seconds


class ResourceAllocator:
    """Allocates resources across strategies based on solver weights."""

    def __init__(self, total_tokens: int = 100_000, total_agents: int = 5, total_time: float = 300.0):
        self.total_tokens = total_tokens
        self.total_agents = total_agents
        self.total_time = total_time

    def allocate(self, strategy_names: list[str], weights: list[float]) -> list[Allocation]:
        """Distribute resources proportionally to strategy weights.

        Args:
            strategy_names: Names of the strategies
            weights: Probability/weight for each strategy (should sum to ~1.0)

        Returns:
            List of Allocation objects with resources assigned
        """
        total_weight = sum(weights)
        if total_weight == 0:
            # Uniform allocation
            n = len(strategy_names)
            weights = [1.0 / n] * n
            total_weight = 1.0

        normalized = [w / total_weight for w in weights]

        allocations = []
        remaining_agents = self.total_agents

        for i, (name, weight) in enumerate(zip(strategy_names, normalized)):
            tokens = int(self.total_tokens * weight)
            time_budget = self.total_time * weight

            # Distribute agents, ensuring at least 1 for strategies with weight > 0.1
            if weight > 0.1 and remaining_agents > 0:
                agents = max(1, round(self.total_agents * weight))
                agents = min(agents, remaining_agents)
                remaining_agents -= agents
            else:
                agents = 0

            allocations.append(Allocation(
                strategy_name=name,
                weight=weight,
                tokens=tokens,
                agents=agents,
                time_budget=time_budget,
            ))

        return allocations
