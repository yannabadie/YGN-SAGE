"""Meta-strategy solvers inspired by PSRO/DCH game theory.

Implements Regret Matching, Projected Replicator Dynamics (PRD),
and advanced SOTA variants like VAD-CFR and SHOR-PSRO.
"""
from __future__ import annotations

import math
from enum import Enum
import numpy as np
from typing import Any, List


class SolverMode(Enum):
    """Execution mode for decoupled SOTA solvers."""
    TRAINING = "training"
    EVALUATION = "evaluation"


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
    As derived from YGN-SAGE Core Research.
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
        # SOTA Mandate: Taux de décroissance de 0.1 pour magnitude et 0.9 pour précédent.
        inst_regrets = action_utilities - action_utilities[chosen_action]
        inst_mag = np.max(np.abs(inst_regrets))
        
        # SOTA Core Research Mandate: Decay factor (gamma) is exactly 0.9
        self._ewma_volatility = 0.1 * inst_mag + 0.9 * self._ewma_volatility
        
        # SOTA Mandate: Sensibilité à la volatilité fixée à 0.5.
        v_t = min(1.0, 0.5 * self._ewma_volatility)

        # 2. Adaptive Discounting (alpha for positive, beta for negative)
        # SOTA Mandate: Base alpha=1.5, Base beta=-0.1.
        alpha = max(0.1, 1.5 - 0.5 * v_t)
        beta = min(alpha, -0.1 - 0.5 * v_t)
        
        t = float(self._iterations)
        d_pos = (t**alpha) / (t**alpha + 1.0)
        d_neg = (t**beta) / (t**beta + 1.0)

        # 3. Update Regrets with Boosting
        # SOTA Core Research Mandate: Boost factor of exactly 1.1 for positive regrets.
        for i in range(self.n_actions):
            r_boosted = inst_regrets[i] * 1.1 if inst_regrets[i] > 0 else inst_regrets[i]
            discount = d_pos if self._cumulative_regret[i] >= 0 else d_neg
            self._cumulative_regret[i] = (self._cumulative_regret[i] * discount) + r_boosted
            
            # SOTA Mandate: Cap for negative regret at -20.0.
            self._cumulative_regret[i] = max(-20.0, self._cumulative_regret[i])

        # 4. Policy Accumulation with Hard Warm-Start
        # SOTA Core Research Mandate: Seuil de démarrage à chaud de 500 itérations.
        if self._iterations >= self.warm_start_threshold:
            # SOTA Mandate: Gamma base 2.0 max 4.0.
            gamma = min(4.0, 2.0 + 1.5 * v_t)
            w_time = t**gamma
            w_mag = (1.0 + (inst_mag / 2.0))**0.5
            w_stable = 1.0 / (1.0 + inst_mag**1.5)
            final_weight = w_time * w_mag * w_stable
            
            # Non-linear scaling mandate: proj_R ** 1.5
            current_strategy = self.get_strategy()
            scaled_strategy = np.power(np.maximum(0.0, current_strategy), 1.5)
            if np.sum(scaled_strategy) > 1e-12:
                scaled_strategy /= np.sum(scaled_strategy)
            else:
                scaled_strategy = current_strategy
                
            self._cumulative_policy += final_weight * scaled_strategy

    def average_strategy(self) -> np.ndarray:
        """Get the time-averaged policy (Nash equilibrium convergence)."""
        total = np.sum(self._cumulative_policy)
        if total > 1e-12:
            return self._cumulative_policy / total
        return self.get_strategy()


class SHORPSROSolver:
    """Smoothed Hybrid Optimistic Regret PSRO (SHOR-PSRO) Solver.

    SOTA 2026: Blends Optimistic Regret Matching with Smoothed Softmax
    using decoupled training/evaluation modes and annealing.
    """

    def __init__(self, n_actions: int, total_iters: int = 75, mode: SolverMode = SolverMode.TRAINING):
        self.n_actions = n_actions
        self.total_iters = total_iters
        self.mode = mode
        self._regret_matcher = RegretMatcher(n_actions)
        self._iterations = 0

    def _get_params(self) -> tuple[float, float, float, float]:
        """Dynamic Annealing Schedule based on Mandate."""
        if self.mode == SolverMode.EVALUATION:
            # SOTA Mandate: Fixed strict params for evaluation
            return 0.01, 0.001, 0.0, 0.2  # lambda, temp, diversity, momentum
            
        # TRAINING Mode
        p = min(1.0, self._iterations / self.total_iters)
        # SOTA Mandate: lambda 0.3 -> 0.05
        blend = 0.30 - (0.25 * p)
        # SOTA Mandate: temp 0.5 -> 0.01
        temp = 0.50 - (0.49 * p)
        # SOTA Mandate: diversity 0.05 -> 0.001
        div = 0.05 - (0.049 * p)
        # SOTA Mandate: momentum 0.5
        momentum = 0.5
        return blend, temp, div, momentum

    def get_strategy(self, payoffs: np.ndarray | List[float]) -> np.ndarray:
        """Hybrid Blend Strategy calculation."""
        payoffs = np.array(payoffs)
        blend, temp, div, momentum = self._get_params()

        # 1. Optimistic Regret Matching Strategy
        sigma_orm = self._regret_matcher.get_strategy()
        
        # Apply diversity bonus as mandated
        boosted_payoffs = payoffs + div * (1.0 - sigma_orm)

        # 2. Smoothed Softmax (Best Pure Strategy)
        stable_payoffs = boosted_payoffs - np.max(boosted_payoffs)
        exp_vals = np.exp(stable_payoffs / temp)
        sigma_pure = exp_vals / np.sum(exp_vals)

        # 3. Blending (Mandate: σ_hybrid = (1-λ)σ_ORM + λσ_Softmax)
        return (1.0 - blend) * sigma_orm + blend * sigma_pure

    def update(self, action_utilities: np.ndarray | List[float], chosen_action: int) -> None:
        """Update internal regret matcher and increment iteration."""
        self._regret_matcher.update(action_utilities, chosen_action)
        self._iterations += 1


class SAMPOSolver:
    """Stable Agentic Multi-turn Policy Optimization (SAMPO).
    
    SOTA 2026: Provides stability for long-horizon agentic trajectories 
    using sequence-level importance sampling and turn-level advantages.
    """

    def __init__(
        self, 
        n_actions: int, 
        clip_epsilon: float = 0.2, 
        filter_threshold: int = 10,
        gamma: float = 0.99
    ):
        self.n_actions = n_actions
        self.clip_epsilon = clip_epsilon
        self.filter_threshold = filter_threshold
        self.gamma = gamma
        self._policy = np.ones(n_actions) / n_actions
        self._iterations = 0

    def get_strategy(self) -> np.ndarray:
        return self._policy

    def update(self, trajectories: List[Dict[str, Any]]) -> None:
        """Update policy using SAMPO logic.
        
        Args:
            trajectories: List of dicts containing 'actions', 'rewards', 'log_probs'.
        """
        if not trajectories:
            return

        # 1. Dynamic Filtering (SOTA Mandate: Filter uninformative batches)
        valid_trajectories = []
        for traj in trajectories:
            rewards = np.array(traj['rewards'])
            # Ignore if all success or all failure within threshold
            if 0 < np.sum(rewards > 0.5) < self.filter_threshold:
                valid_trajectories.append(traj)

        if not valid_trajectories:
            return

        # 2. Sequence-Level Importance Sampling & Policy Update
        # SOTA Mandate: Sequence-level clipping to prevent forgetting
        for traj in valid_trajectories:
            # Compute turn-level advantage
            rewards = np.array(traj['rewards'])
            baseline = np.mean(rewards)
            advantages = rewards - baseline
            
            # Old policy probabilities (we assume current policy for simplification, 
            # in a full PPO we'd store log_probs, but DGM acts as an online learner here)
            old_policy = self._policy.copy()
            
            for action, adv in zip(traj['actions'], advantages):
                # Calculate ratio (r_t) - Simplified for the DGM Action Loop
                # We use a base learning rate, but clip the maximum shift
                lr = 0.05
                proposed_shift = lr * adv
                
                # SAMPO Clipping: Prevent the policy from shifting more than clip_epsilon per update
                clipped_shift = np.clip(proposed_shift, -self.clip_epsilon, self.clip_epsilon)
                
                self._policy[action] += clipped_shift
        
        # Normalize and ensure strict bounds
        self._policy = np.maximum(1e-8, self._policy)
        self._policy /= np.sum(self._policy)
        self._iterations += 1

    def stats(self) -> Dict[str, Any]:
        return {
            "iterations": self._iterations,
            "entropy": -np.sum(self._policy * np.log(self._policy + 1e-10))
        }
