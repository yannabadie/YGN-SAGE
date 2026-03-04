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


class VolatilityGatedScheduler:
    """SOTA 2026: Resource scheduler gated by process volatility.
    
    Implements VAD-CFR inspired adaptive budgeting:
    - High Volatility: Dampen resource allocation to prevent over-optimization on noisy signals.
    - Low Volatility: Aggressively increase allocation to push past local optima.
    - Hard Warm-Start: Constant allocation for initial T=500 steps.
    """

    def __init__(
        self, 
        base_allocator: ResourceAllocator,
        warm_start_steps: int = 500,
        volatility_window: int = 10
    ):
        self.allocator = base_allocator
        self.warm_start_steps = warm_start_steps
        self.volatility_history: list[float] = []
        self.window = volatility_window
        self.steps = 0
        self._ewma_volatility = 0.0

    def step(self, current_volatility: float) -> float:
        """Update internal volatility tracking and return the resource multiplier."""
        self.steps += 1
        
        # Update EWMA Volatility (SOTA 2026 approach)
        self._ewma_volatility = 0.1 * current_volatility + 0.9 * self._ewma_volatility
        
        if self.steps < self.warm_start_steps:
            return 1.0 # Constant during warm-start
            
        # VAD Multiplier: v_t normalized to [0, 1]
        v_t = min(1.0, self._ewma_volatility / 2.0)
        
        # Adaptive Multiplier: 
        # If volatility is high, we want to spend LESS (exploration-only)
        # If volatility is low, we want to spend MORE (exploitation-push)
        multiplier = max(0.5, 2.0 - 1.5 * v_t)
        
        return multiplier

    def get_adjusted_allocation(
        self, 
        strategy_names: list[str], 
        weights: list[float],
        volatility: float
    ) -> list[Allocation]:
        """Get resource allocations adjusted by the volatility gate."""
        multiplier = self.step(volatility)
        
        # Apply multiplier to allocator's base budget temporarily
        orig_tokens = self.allocator.total_tokens
        orig_time = self.allocator.total_time
        
        self.allocator.total_tokens = int(orig_tokens * multiplier)
        self.allocator.total_time = orig_time * multiplier
        
        allocations = self.allocator.allocate(strategy_names, weights)
        
        # Restore original budget
        self.allocator.total_tokens = orig_tokens
        self.allocator.total_time = orig_time
        
        return allocations
