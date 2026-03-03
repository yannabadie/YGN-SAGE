"""Evolution engine: orchestrates the evolutionary loop.

Combines population management, LLM mutation, and evaluation cascade
into a complete evolutionary optimization cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from sage.evolution.population import Population, Individual
from sage.evolution.mutator import Mutator
from sage.evolution.evaluator import Evaluator, EvalResult


@dataclass
class EvolutionConfig:
    """Configuration for an evolution run."""
    population_size: int = 100
    feature_dims: int = 2
    bins_per_dim: int = 10
    max_generations: int = 50
    mutations_per_generation: int = 10
    elite_count: int = 3


class EvolutionEngine:
    """Orchestrates evolutionary optimization of code solutions."""

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

        Args:
            mutate_fn: Async function that takes parent code and returns
                       (mutated_code, feature_descriptor).

        Returns:
            List of new individuals that were accepted into the population.
        """
        self.generation += 1
        accepted = []

        # Select parents from population
        parents = self._population.sample(self.config.mutations_per_generation)
        if not parents:
            return []

        for parent in parents:
            # Generate mutation
            try:
                new_code, features = await mutate_fn(parent.code)
            except Exception:
                continue

            # Evaluate
            eval_result = await self._evaluator.evaluate(new_code)

            child = Individual(
                code=new_code,
                score=eval_result.score,
                features=features,
                generation=self.generation,
                parent_id=parent.id,
                metadata={"eval_details": eval_result.details},
            )

            if self._population.add(child):
                accepted.append(child)

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
            "population_size": self._population.size(),
            "coverage": self._population.coverage(),
            "best_score": best.score if best else 0.0,
            "best_id": best.id if best else None,
        }
