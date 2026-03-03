"""Meta-strategy solvers inspired by PSRO/DCH game theory.

Implements Regret Matching, Projected Replicator Dynamics (PRD),
and advanced SOTA variants like VAD-CFR and SHOR-PSRO.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, List


class RegretMatcher:
    """Regret Matching algorithm for computing mixed strategies."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._cumulative_regret = np.zeros(n_actions)
        self._strategy_sum = np.zeros(n_actions)
        self._iterations = 0

    def get_strategy(self) -> np.ndarray:
        """Get current mixed strategy (probability distribution over actions)."""
        positive_regrets = np.maximum(0.0, self._cumulative_regret)
        total = np.sum(positive_regrets)
        if total > 1e-12:
            return positive_regrets / total
        # Uniform if no positive regrets
        return np.ones(self.n_actions) / self.n_actions

    def update(self, action_utilities: np.ndarray | List[float], chosen_action: int) -> None:
        """Update regrets based on observed utilities for each action."""
        action_utilities = np.array(action_utilities)
        chosen_utility = action_utilities[chosen_action]
        regret = action_utilities - chosen_utility
        self._cumulative_regret += regret

        strategy = self.get_strategy()
        self._strategy_sum += strategy
        self._iterations += 1

    def average_strategy(self) -> np.ndarray:
        """Get the time-averaged strategy (converges to Nash equilibrium)."""
        if self._iterations == 0:
            return np.ones(self.n_actions) / self.n_actions
        total = np.sum(self._strategy_sum)
        if total > 1e-12:
            return self._strategy_sum / total
        return np.ones(self.n_actions) / self.n_actions


class VolatilityAdaptiveSolver:
    """Volatility-Adaptive Discounted CFR (VAD-CFR) Solver.

    SOTA 2026: Adjusts regrets and policy accumulation based on EWMA volatility.
    Includes a hard warm-start mechanism to filter initial exploration noise.
    """

    def __init__(self, n_actions: int, warm_start_threshold: int = 500):
        self.n_actions = n_actions
        self.warm_start_threshold = warm_start_threshold
        self._cumulative_regret = np.zeros(n_actions)
        self._cumulative_policy = np.zeros(n_actions)
        self._ewma_volatility = 0.0
        self._iterations = 0

    def get_strategy(self) -> np.ndarray:
        """Get current mixed strategy from cumulative regrets."""
        positive_regrets = np.maximum(0.0, self._cumulative_regret)
        total = np.sum(positive_regrets)
        if total > 1e-12:
            return positive_regrets / total
        return np.ones(self.n_actions) / self.n_actions

    def update(self, action_utilities: np.ndarray | List[float], chosen_action: int) -> None:
        """VAD-CFR update with adaptive discounting and volatility tracking."""
        action_utilities = np.array(action_utilities)
        self._iterations += 1
        
        # 1. Update Volatility (EWMA)
        inst_regrets = action_utilities - action_utilities[chosen_action]
        inst_mag = np.max(np.abs(inst_regrets))
        self._ewma_volatility = 0.1 * inst_mag + 0.9 * self._ewma_volatility
        v_t = min(1.0, self._ewma_volatility / 2.0)

        # 2. Adaptive Discounting (alpha for positive, beta for negative)
        alpha = max(0.1, 1.5 - 0.5 * v_t)
        beta = min(alpha, -0.1 - 0.5 * v_t)
        
        t = float(self._iterations)
        d_pos = (t**alpha) / (t**alpha + 1.0)
        d_neg = (t**beta) / (t**beta + 1.0)

        # 3. Update Regrets with Boosting
        for i in range(self.n_actions):
            r_boosted = inst_regrets[i] * 1.1 if inst_regrets[i] > 0 else inst_regrets[i]
            discount = d_pos if self._cumulative_regret[i] >= 0 else d_neg
            self._cumulative_regret[i] = (self._cumulative_regret[i] * discount) + r_boosted
            # Lower cap for stability
            self._cumulative_regret[i] = max(-20.0, self._cumulative_regret[i])

        # 4. Policy Accumulation with Hard Warm-Start
        if self._iterations >= self.warm_start_threshold:
            gamma = min(4.0, 2.0 + 1.5 * v_t)
            w_time = t**gamma
            w_mag = (1.0 + (inst_mag / 2.0))**0.5
            w_stable = 1.0 / (1.0 + inst_mag**1.5)
            final_weight = w_time * w_mag * w_stable
            
            current_strategy = self.get_strategy()
            self._cumulative_policy += final_weight * current_strategy

    def average_strategy(self) -> np.ndarray:
        """Get the time-averaged policy (Nash equilibrium convergence)."""
        total = np.sum(self._cumulative_policy)
        if total > 1e-12:
            return self._cumulative_policy / total
        return self.get_strategy()


class SHORPSROSolver:
    """Smoothed Hybrid Optimistic Regret PSRO (SHOR-PSRO) Solver.

    SOTA 2026: Blends Optimistic Regret Matching with Smoothed Softmax
    using a dynamic annealing schedule for exploration/exploitation.
    """

    def __init__(self, n_actions: int, total_iters: int = 75):
        self.n_actions = n_actions
        self.total_iters = total_iters
        self._regret_matcher = RegretMatcher(n_actions)
        self._iterations = 0

    def _get_params(self) -> tuple[float, float, float]:
        """Dynamic Annealing Schedule."""
        p = min(1.0, self._iterations / self.total_iters)
        blend = 0.30 - (0.25 * p)  # Lambda
        temp = 0.50 - (0.49 * p)   # Tau (Softmax Temperature)
        div = 0.05 - (0.049 * p)   # Diversity bonus
        return blend, temp, div

    def get_strategy(self, payoffs: np.ndarray | List[float]) -> np.ndarray:
        """Hybrid Blend Strategy calculation."""
        payoffs = np.array(payoffs)
        blend, temp, div = self._get_params()

        # 1. Optimistic Regret Matching Strategy
        # Note: Simplified ORM here using diversity-boosted RM
        boosted_payoffs = payoffs + div * (1.0 - self._regret_matcher.get_strategy())
        # We simulate one internal iteration for ORM-like behavior
        sigma_orm = self._regret_matcher.get_strategy()

        # 2. Smoothed Softmax (Best Pure Strategy)
        stable_payoffs = payoffs - np.max(payoffs)
        exp_vals = np.exp(stable_payoffs / temp)
        sigma_pure = exp_vals / np.sum(exp_vals)

        # 3. Blending
        return (1.0 - blend) * sigma_orm + blend * sigma_pure

    def update(self, action_utilities: np.ndarray | List[float], chosen_action: int) -> None:
        """Update internal regret matcher and increment iteration."""
        self._regret_matcher.update(action_utilities, chosen_action)
        self._iterations += 1


class PRDSolver:
    """Projected Replicator Dynamics solver."""

    def __init__(self, n_actions: int, learning_rate: float = 0.1):
        self.n_actions = n_actions
        self.lr = learning_rate
        self._weights = np.ones(n_actions)

    def get_strategy(self) -> np.ndarray:
        """Get current strategy (normalized weights)."""
        total = np.sum(self._weights)
        if total > 0:
            return self._weights / total
        return np.ones(self.n_actions) / self.n_actions

    def update(self, payoffs: np.ndarray | List[float]) -> None:
        """Update weights using replicator dynamics."""
        payoffs = np.array(payoffs)
        strategy = self.get_strategy()
        avg_payoff = np.sum(strategy * payoffs)

        # Replicator dynamics: dx_i/dt = x_i * (payoff_i - avg_payoff)
        growth = strategy * (payoffs - avg_payoff)
        self._weights = np.maximum(1e-8, self._weights + self.lr * growth)

    def entropy(self) -> float:
        """Shannon entropy of current strategy."""
        strategy = self.get_strategy()
        return -np.sum(strategy * np.log(strategy + 1e-10))
