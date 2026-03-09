"""Evolution engine: orchestrates the evolutionary loop.

Combines population management, LLM mutation, and evaluation cascade
into a complete evolutionary optimization cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
import logging

import numpy as np

from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator
from sage.evolution.evaluator import Evaluator

from sage.strategy.solvers import SAMPOSolver

log = logging.getLogger(__name__)

SAMPO_ACTION_DESCRIPTIONS = {
    0: "Optimize execution performance and reduce latency",
    1: "Improve correctness and fix edge cases",
    2: "Expand search space — explore novel algorithmic approaches",
    3: "Tighten constraints — make code more robust and safe",
    4: "Simplify and reduce complexity while maintaining functionality",
}

@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""
    population_size: int = 100
    feature_dims: int = 2
    bins_per_dim: int = 10
    max_generations: int = 50
    mutations_per_generation: int = 10
    elite_count: int = 3
    # Warm-start threshold: skip policy accumulation for first N mutations (arbitrary, needs tuning)
    hard_warm_start_threshold: int = 500
    # Enable SAMPO-based hyperparameter self-adjustment
    enable_sampo: bool = True
    z3_constraints: list[str] = field(default_factory=list)


class EvolutionEngine:
    """Orchestrates evolutionary optimization of code solutions.

    Uses SAMPO action selection to adjust 3 hyperparameters
    (mutations_per_generation, clip_epsilon, filter_threshold) during
    evolution. This is NOT a Godel Machine -- it does not produce
    self-proofs. It is a simple online hyperparameter adjustment loop.
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        mutator: Mutator | None = None,
        evaluator: Evaluator | None = None,
    ):
        self.config = config or EvolutionConfig()
        self._mutator = mutator or Mutator()
        self._evaluator = evaluator or Evaluator()
        self._population = Population(
            feature_dims=self.config.feature_dims,
            bins_per_dim=self.config.bins_per_dim,
        )
        self.generation: int = 0
        self.total_mutations: int = 0
        # eBPF evaluator: wire as default first stage (sub-ms execution)
        try:
            from sage.evolution.ebpf_evaluator import EbpfEvaluator
            ebpf = EbpfEvaluator()
            self._evaluator.add_stage("ebpf_sandbox", ebpf.evaluate, threshold=0.0, weight=1.0)
            log.info("eBPF evaluator wired as default evolution stage")
        except Exception:
            log.debug("eBPF evaluator not available — evolution runs without hardware sandbox")

        # SAMPO solver for action selection (hyperparameter adjustment)
        self._sampo_solver = SAMPOSolver(n_actions=5) # Actions: MutateCode, MutatePrompt, MutateHyper, etc.
        self._trajectories = []

    @property
    def population(self) -> Population:
        return self._population

    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator

    def seed(self, individuals: list[Individual]) -> int:
        """Seed the population with initial individuals. Returns count added."""
        count = 0
        for ind in individuals:
            if self._population.add(ind):
                count += 1
        return count

    async def evolve_step(
        self,
        mutate_fn: Callable[..., Awaitable[tuple[str, tuple[int, ...]]]],
    ) -> list[Individual]:
        """Run one generation of evolution.

        Samples mutator actions from SAMPOSolver to guide mutation strategy.
        """
        self.generation += 1
        accepted = []

        # Select parents from population
        parents = self._population.sample(self.config.mutations_per_generation)
        if not parents:
            return []

        current_gen_traj = {"actions": [], "rewards": []}

        for parent in parents:
            self.total_mutations += 1
            
            # Strategy action selection (see SAMPO_ACTION_DESCRIPTIONS)
            sampo_action = np.random.choice(5, p=self._sampo_solver.get_strategy())
            
            # --- Hyperparameter self-adjustment ---
            if sampo_action == 2:
                # Action 2: Expand search space (Engine Hyperparameter)
                self.config.mutations_per_generation = min(50, self.config.mutations_per_generation + 2)
            elif sampo_action == 3:
                # Action 3: Tighten SAMPO clipping (Solver Hyperparameter)
                self._sampo_solver.clip_epsilon = max(0.05, self._sampo_solver.clip_epsilon * 0.9)
            elif sampo_action == 4:
                # Action 4: Relax SAMPO filtering (Solver Hyperparameter)
                self._sampo_solver.filter_threshold = max(2, self._sampo_solver.filter_threshold - 1)
            
            # Generate mutation
            try:
                # Mutation function receives strategy context for guided mutation
                sampo_context = {
                    "action": int(sampo_action),
                    "description": SAMPO_ACTION_DESCRIPTIONS.get(int(sampo_action), ""),
                    "parent_score": parent.score,
                    "generation": self.generation,
                }
                new_code, features = await mutate_fn(parent.code, sampo_context=sampo_context)
            except Exception:
                continue

            # Z3 safety gate removed — was validating static config constraints,
            # not the actual mutated code (audit finding Z3-06).

            # Evaluate
            eval_result = await self._evaluator.evaluate(new_code)

            # Apply warm-start logic: flag early mutations as exploratory
            is_warm_up = self.total_mutations < self.config.hard_warm_start_threshold

            child = Individual(
                code=new_code,
                score=eval_result.score,
                features=features,
                generation=self.generation,
                parent_id=parent.id,
                metadata={
                    "eval_details": eval_result.details,
                    "is_warm_up": is_warm_up,
                    "sampo_action": sampo_action
                },
            )

            # SAMPO Reward Logic: Reward action if it led to a population improvement
            reward = 1.0 if eval_result.score > parent.score else 0.0
            current_gen_traj["actions"].append(sampo_action)
            current_gen_traj["rewards"].append(reward)

            if self._population.add(child):
                accepted.append(child)

        # Update SAMPO policy
        self._trajectories.append(current_gen_traj)
        if len(self._trajectories) >= 5:  # Batch update
            try:
                self._sampo_solver.update(self._trajectories)
            finally:
                self._trajectories = []

        return accepted

    def best_solution(self) -> Individual | None:
        """Get the current best solution."""
        best = self._population.best(1)
        return best[0] if best else None

    def stats(self) -> dict[str, Any]:
        """Get current evolution statistics."""
        best = self.best_solution()
        return {
            "generation": self.generation,
            "total_mutations": self.total_mutations,
            "population_size": self._population.size(),
            "coverage": self._population.coverage(),
            "best_score": best.score if best else 0.0,
            "best_id": best.id if best else None,
            "is_warm_up": self.total_mutations < self.config.hard_warm_start_threshold,
            "sampo_entropy": self._sampo_solver.stats()["entropy"]
        }
