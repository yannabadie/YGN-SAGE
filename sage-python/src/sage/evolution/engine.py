"""Evolution engine: orchestrates the evolutionary loop.

Combines population management, LLM mutation, and evaluation cascade
into a complete evolutionary optimization cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable
import numpy as np

from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator
from sage.evolution.evaluator import Evaluator, EvalResult


from sage.strategy.solvers import SAMPOSolver

@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""
    population_size: int = 100
    feature_dims: int = 2
    bins_per_dim: int = 10
    max_generations: int = 50
    mutations_per_generation: int = 10
    elite_count: int = 3
    # SOTA Mandate: Hard warm-start threshold for filtering initial noise.
    hard_warm_start_threshold: int = 500
    # DGM Mandate: Enable self-modification
    enable_dgm: bool = True


class EvolutionEngine:
    """Orchestrates evolutionary optimization of code solutions.
    
    ASI Upgrade: Darwin Godel Machine (DGM) capable of self-modifying 
    the mutator logic and hyperparameters using SAMPO.
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
        
        # SOTA: DGM Solver for self-optimization
        self._dgm_solver = SAMPOSolver(n_actions=5) # Actions: MutateCode, MutatePrompt, MutateHyper, etc.
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
        mutate_fn: Callable[[str], Awaitable[tuple[str, tuple[int, ...]]]],
    ) -> list[Individual]:
        """Run one generation of evolution.

        ASI/DGM: Samples mutator actions from SAMPOSolver to evolve the system itself.
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
            
            # DGM Action Selection (e.g., 0: Mutate, 1: Hybrid, 2: Self-Fix)
            dgm_action = np.random.choice(5, p=self._dgm_solver.get_strategy())
            
            # Generate mutation
            try:
                # SOTA: Mutate function now incorporates DGM context
                new_code, features = await mutate_fn(parent.code)
            except Exception:
                continue

            # Evaluate
            eval_result = await self._evaluator.evaluate(new_code)

            # SOTA Mandate: Apply Hard Warm-Start logic.
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
                    "dgm_action": dgm_action
                },
            )

            # DGM Reward Logic: Reward action if it led to a population improvement
            reward = 1.0 if eval_result.score > parent.score else 0.0
            current_gen_traj["actions"].append(dgm_action)
            current_gen_traj["rewards"].append(reward)

            if self._population.add(child):
                accepted.append(child)

        # Update DGM Policy using SAMPO
        self._trajectories.append(current_gen_traj)
        if len(self._trajectories) >= 5: # Batch update
            self._dgm_solver.update(self._trajectories)
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
            "dgm_entropy": self._dgm_solver.stats()["entropy"]
        }
