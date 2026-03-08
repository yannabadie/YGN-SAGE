"""Meta-strategy solvers inspired by PSRO/DCH game theory.

Implements Regret Matching, and two experimental variants:
VAD-CFR (Volatility-Adaptive Discounted CFR) and SHOR-PSRO
(Smoothed Hybrid Optimistic Regret PSRO).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

import numpy as np


class SolverMode(Enum):
    """Execution mode for solvers (training vs evaluation)."""
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

    Adjusts regrets and policy accumulation based on EWMA volatility.
    Includes a hard warm-start mechanism to filter initial exploration noise.
    All hyperparameters below are chosen heuristically and have not been
    validated against a theoretical baseline.
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
        # Decay: 0.1 weight on new magnitude, 0.9 on history (chosen heuristically)
        inst_regrets = action_utilities - action_utilities[chosen_action]
        inst_mag = np.max(np.abs(inst_regrets))

        # EWMA decay factor 0.9 (not validated, needs tuning)
        self._ewma_volatility = 0.1 * inst_mag + 0.9 * self._ewma_volatility

        # Volatility sensitivity capped at 0.5 (arbitrary, needs tuning)
        v_t = min(1.0, 0.5 * self._ewma_volatility)

        # 2. Adaptive Discounting (alpha for positive, beta for negative)
        # Base alpha=1.5, beta=-0.1 (heuristic, not from any paper)
        alpha = max(0.1, 1.5 - 0.5 * v_t)
        beta = min(alpha, -0.1 - 0.5 * v_t)
        
        t = float(self._iterations)
        d_pos = (t**alpha) / (t**alpha + 1.0)
        d_neg = (t**beta) / (t**beta + 1.0)

        # 3. Update Regrets with Boosting
        # Boost factor 1.1 for positive regrets (heuristic, not validated)
        for i in range(self.n_actions):
            r_boosted = inst_regrets[i] * 1.1 if inst_regrets[i] > 0 else inst_regrets[i]
            discount = d_pos if self._cumulative_regret[i] >= 0 else d_neg
            self._cumulative_regret[i] = (self._cumulative_regret[i] * discount) + r_boosted
            
            # Cap negative regret at -20.0 to prevent runaway accumulation
            self._cumulative_regret[i] = max(-20.0, self._cumulative_regret[i])

        # 4. Policy Accumulation with Hard Warm-Start
        # Skip accumulation for the first N iterations (warm-start threshold, arbitrary)
        if self._iterations >= self.warm_start_threshold:
            # Gamma base 2.0, max 4.0 (heuristic time-weighting exponent)
            gamma = min(4.0, 2.0 + 1.5 * v_t)
            w_time = t**gamma
            w_mag = (1.0 + (inst_mag / 2.0))**0.5
            w_stable = 1.0 / (1.0 + inst_mag**1.5)
            final_weight = w_time * w_mag * w_stable
            
            # Non-linear scaling: proj_R ** 1.5 (heuristic sharpening)
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

    Blends Optimistic Regret Matching with Smoothed Softmax
    using decoupled training/evaluation modes and annealing.
    Annealing schedule and blend parameters are heuristic.
    """

    def __init__(self, n_actions: int, total_iters: int = 75, mode: SolverMode = SolverMode.TRAINING):
        self.n_actions = n_actions
        self.total_iters = total_iters
        self.mode = mode
        self._regret_matcher = RegretMatcher(n_actions)
        self._iterations = 0

    def _get_params(self) -> tuple[float, float, float, float]:
        """Dynamic annealing schedule (heuristic, not from any paper)."""
        if self.mode == SolverMode.EVALUATION:
            # Fixed strict params for evaluation mode
            return 0.01, 0.001, 0.0, 0.2  # lambda, temp, diversity, momentum

        # TRAINING Mode: linear annealing over total_iters
        p = min(1.0, self._iterations / self.total_iters)
        blend = 0.30 - (0.25 * p)       # lambda: 0.3 -> 0.05
        temp = 0.50 - (0.49 * p)         # temperature: 0.5 -> 0.01
        div = 0.05 - (0.049 * p)         # diversity bonus: 0.05 -> 0.001
        momentum = 0.5                    # fixed (arbitrary)
        return blend, temp, div, momentum

    def get_strategy(self, payoffs: np.ndarray | List[float]) -> np.ndarray:
        """Hybrid Blend Strategy calculation."""
        payoffs = np.array(payoffs)
        blend, temp, div, momentum = self._get_params()

        # 1. Optimistic Regret Matching Strategy
        sigma_orm = self._regret_matcher.get_strategy()
        
        # Apply diversity bonus to under-explored actions
        boosted_payoffs = payoffs + div * (1.0 - sigma_orm)

        # 2. Smoothed Softmax (Best Pure Strategy)
        stable_payoffs = boosted_payoffs - np.max(boosted_payoffs)
        exp_vals = np.exp(stable_payoffs / temp)
        sigma_pure = exp_vals / np.sum(exp_vals)

        # 3. Blending: σ_hybrid = (1-λ)σ_ORM + λσ_Softmax
        return (1.0 - blend) * sigma_orm + blend * sigma_pure

    def update(self, action_utilities: np.ndarray | List[float], chosen_action: int) -> None:
        """Update internal regret matcher and increment iteration."""
        self._regret_matcher.update(action_utilities, chosen_action)
        self._iterations += 1


class SAMPOSolver:
    """Stable Agentic Multi-turn Policy Optimization (SAMPO).

    A clipped incremental policy update solver for multi-action selection.
    Uses per-action advantage estimation with clipped shifts to prevent
    large policy changes. NOT equivalent to PPO/TRPO (no importance
    sampling ratio, no KL constraint, no GAE).

    Suitable for online learning with small batches where full PPO
    infrastructure would be overkill.
    """

    def __init__(
        self, 
        n_actions: int, 
        clip_epsilon: float = 0.2, 
        filter_threshold: int = 10,
        gamma: float = 0.99,
        base_lr: float = 0.05,
        min_lr: float = 0.005,
        max_lr: float = 0.2,
        lr_decay: float = 0.995,
        rms_beta: float = 0.95,
        epsilon: float = 1e-8,
        mixed_precision: bool = False,
        grad_scale_init: float = 128.0,
        grad_scale_growth: float = 2.0,
        grad_scale_backoff: float = 0.5,
        grad_scale_growth_interval: int = 200,
    ):
        self.n_actions = n_actions
        self.clip_epsilon = clip_epsilon
        self.filter_threshold = filter_threshold
        self.gamma = gamma
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.rms_beta = rms_beta
        self.epsilon = epsilon
        self.mixed_precision = mixed_precision
        self.grad_scale_growth = grad_scale_growth
        self.grad_scale_backoff = grad_scale_backoff
        self.grad_scale_growth_interval = max(1, grad_scale_growth_interval)
        self._policy = np.ones(n_actions) / n_actions
        self._iterations = 0
        self._adv_sq_ema = 0.0
        self._last_lr = base_lr
        self._grad_scale = max(1.0, float(grad_scale_init))
        self._stable_steps = 0

    def _adaptive_lr(self, advantage: float) -> float:
        """Compute a numerically stable adaptive learning rate.

        We combine:
        - Exponential decay over iterations (reduces late-stage oscillations).
        - RMS normalization over advantage magnitude (suppresses spikes).
        """
        adv_sq = float(advantage * advantage)
        self._adv_sq_ema = (self.rms_beta * self._adv_sq_ema) + ((1.0 - self.rms_beta) * adv_sq)

        decayed = self.base_lr * (self.lr_decay ** self._iterations)
        normalized = decayed / np.sqrt(self._adv_sq_ema + self.epsilon)
        lr = float(np.clip(normalized, self.min_lr, self.max_lr))
        self._last_lr = lr
        return lr

    def get_strategy(self) -> np.ndarray:
        return self._policy

    def update(self, trajectories: List[Dict[str, Any]]) -> None:
        """Update policy using SAMPO logic.
        
        Args:
            trajectories: List of dicts containing 'actions', 'rewards', 'log_probs'.
        """
        if not trajectories:
            return

        # 1. Dynamic Filtering: skip uninformative batches
        valid_trajectories = []
        for traj in trajectories:
            actions_raw = traj.get("actions", [])
            rewards_raw = traj.get("rewards", [])
            actions = np.asarray(actions_raw, dtype=np.int64)
            rewards = np.asarray(rewards_raw, dtype=np.float32)

            # Defensive checks: malformed trajectories are ignored instead of crashing.
            if actions.size == 0 or rewards.size == 0:
                continue
            if not np.isfinite(rewards).all():
                continue
            if np.any(actions < 0) or np.any(actions >= self.n_actions):
                continue

            # Ignore if all success or all failure within threshold
            if 0 < np.sum(rewards > 0.5) < self.filter_threshold:
                valid_trajectories.append((actions, rewards))

        if not valid_trajectories:
            return

        # 2. Vectorized per-action update with optional mixed-precision gradient scaling.
        grad_accum = np.zeros(self.n_actions, dtype=np.float64)
        max_adv = 0.0
        overflow = False
        for actions, rewards in valid_trajectories:
            if actions.size == rewards.size:
                baseline = float(np.mean(rewards))
                advantages = rewards - baseline
                actions_for_update = actions
            elif actions.size == 1:
                # Backward-compatible path used by StrategyEngine.report_outcome():
                # one chosen action and a full per-action payoff vector.
                baseline = float(np.mean(rewards))
                advantages = np.asarray([rewards[int(actions[0])] - baseline], dtype=np.float32)
                actions_for_update = actions
            else:
                continue

            adv_abs_max = float(np.max(np.abs(advantages.astype(np.float64))))
            max_adv = max(max_adv, adv_abs_max)

            if self.mixed_precision:
                f16_max = float(np.finfo(np.float16).max)
                if adv_abs_max * self._grad_scale > f16_max:
                    overflow = True
                    continue
                scaled = np.float16(advantages) * np.float16(self._grad_scale)
                if not np.isfinite(scaled).all():
                    overflow = True
                    continue
                advantages_for_update = (scaled.astype(np.float32) / self._grad_scale)
            else:
                advantages_for_update = advantages

            grad_accum += np.bincount(
                actions_for_update,
                weights=advantages_for_update.astype(np.float64),
                minlength=self.n_actions,
            )

        if overflow and self.mixed_precision:
            self._grad_scale = max(1.0, self._grad_scale * self.grad_scale_backoff)

        if not np.any(grad_accum):
            return

        lr = self._adaptive_lr(max_adv)
        delta = np.clip(lr * grad_accum, -self.clip_epsilon, self.clip_epsilon)
        self._policy += delta

        # Normalize and ensure strict bounds
        self._policy = np.maximum(1e-8, self._policy)
        self._policy /= np.sum(self._policy)
        self._iterations += 1
        if self.mixed_precision and not overflow:
            self._stable_steps += 1
            if self._stable_steps % self.grad_scale_growth_interval == 0:
                self._grad_scale *= self.grad_scale_growth

    def stats(self) -> Dict[str, Any]:
        return {
            "iterations": self._iterations,
            "entropy": -np.sum(self._policy * np.log(self._policy + 1e-10)),
            "learning_rate": self._last_lr,
            "adv_sq_ema": self._adv_sq_ema,
            "grad_scale": self._grad_scale,
        }
