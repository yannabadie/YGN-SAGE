"""Meta-strategy solvers inspired by PSRO/DCH game theory.

Implements Regret Matching and Projected Replicator Dynamics (PRD)
for computing mixed strategies over agent approaches.
"""
from __future__ import annotations

import math
from typing import Any


class RegretMatcher:
    """Regret Matching algorithm for computing mixed strategies.

    Each action accumulates regret for not having been chosen.
    The strategy is proportional to positive cumulative regrets.
    Used in: PSRO meta-solver, multi-agent coordination.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._cumulative_regret = [0.0] * n_actions
        self._strategy_sum = [0.0] * n_actions
        self._iterations = 0

    def get_strategy(self) -> list[float]:
        """Get current mixed strategy (probability distribution over actions)."""
        positive_regrets = [max(0.0, r) for r in self._cumulative_regret]
        total = sum(positive_regrets)
        if total > 0:
            return [r / total for r in positive_regrets]
        # Uniform if no positive regrets
        return [1.0 / self.n_actions] * self.n_actions

    def update(self, action_utilities: list[float], chosen_action: int) -> None:
        """Update regrets based on observed utilities for each action.

        Args:
            action_utilities: Utility/reward for each action this round
            chosen_action: The action that was actually taken
        """
        chosen_utility = action_utilities[chosen_action]
        for i in range(self.n_actions):
            regret = action_utilities[i] - chosen_utility
            self._cumulative_regret[i] += regret

        strategy = self.get_strategy()
        for i in range(self.n_actions):
            self._strategy_sum[i] += strategy[i]
        self._iterations += 1

    def average_strategy(self) -> list[float]:
        """Get the time-averaged strategy (converges to Nash equilibrium)."""
        if self._iterations == 0:
            return [1.0 / self.n_actions] * self.n_actions
        total = sum(self._strategy_sum)
        if total > 0:
            return [s / total for s in self._strategy_sum]
        return [1.0 / self.n_actions] * self.n_actions


class PRDSolver:
    """Projected Replicator Dynamics solver.

    Continuous-time dynamics for computing mixed strategies.
    More stable than Regret Matching for certain game structures.
    Inspired by the PRD meta-solver in PSRO literature.
    """

    def __init__(self, n_actions: int, learning_rate: float = 0.1):
        self.n_actions = n_actions
        self.lr = learning_rate
        self._weights = [1.0] * n_actions  # unnormalized

    def get_strategy(self) -> list[float]:
        """Get current strategy (softmax over weights)."""
        total = sum(self._weights)
        if total > 0:
            return [w / total for w in self._weights]
        return [1.0 / self.n_actions] * self.n_actions

    def update(self, payoffs: list[float]) -> None:
        """Update weights using replicator dynamics.

        Args:
            payoffs: Expected payoff for each action under current strategy.
        """
        strategy = self.get_strategy()
        avg_payoff = sum(s * p for s, p in zip(strategy, payoffs))

        for i in range(self.n_actions):
            # Replicator dynamics: dx_i/dt = x_i * (payoff_i - avg_payoff)
            growth = strategy[i] * (payoffs[i] - avg_payoff)
            self._weights[i] = max(1e-8, self._weights[i] + self.lr * growth)

    def entropy(self) -> float:
        """Shannon entropy of current strategy (measures exploration)."""
        strategy = self.get_strategy()
        return -sum(p * math.log(p + 1e-10) for p in strategy)
