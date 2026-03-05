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
from sage.evolution.evaluator import Evaluator, EvalResult

try:
    from sage.sandbox.z3_validator import Z3Validator
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False

from sage.strategy.solvers import SAMPOSolver

log = logging.getLogger(__name__)

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
    # Z3 Safety Gate: validate evolved code before evaluation
    z3_safety_gate: bool = True
    z3_constraints: list[str] = field(default_factory=list)


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
        self.z3_rejections: int = 0

        # eBPF evaluator: wire as default first stage (sub-ms execution)
        try:
            from sage.evolution.ebpf_evaluator import EbpfEvaluator
            ebpf = EbpfEvaluator()
            self._evaluator.add_stage("ebpf_sandbox", ebpf.evaluate, threshold=0.0, weight=1.0)
            log.info("eBPF evaluator wired as default evolution stage")
        except Exception:
            log.debug("eBPF evaluator not available — evolution runs without hardware sandbox")

        # Z3 Safety Gate
        self._z3: Z3Validator | None = None
        if self.config.z3_safety_gate and _Z3_AVAILABLE:
            self._z3 = Z3Validator()
            log.info("Z3 safety gate enabled")
        elif self.config.z3_safety_gate and not _Z3_AVAILABLE:
            log.warning("Z3 safety gate requested but sage_core not available — skipping")

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
            
            # --- DGM Self-Modification Logic (ASI Phase 2) ---
            if dgm_action == 2:
                # Action 2: Expand search space (Engine Hyperparameter)
                self.config.mutations_per_generation = min(50, self.config.mutations_per_generation + 2)
            elif dgm_action == 3:
                # Action 3: Tighten SAMPO clipping (Solver Hyperparameter)
                self._dgm_solver.clip_epsilon = max(0.05, self._dgm_solver.clip_epsilon * 0.9)
            elif dgm_action == 4:
                # Action 4: Relax SAMPO filtering (Solver Hyperparameter)
                self._dgm_solver.filter_threshold = max(2, self._dgm_solver.filter_threshold - 1)
            
            # Generate mutation
            try:
                # SOTA: Mutate function now incorporates DGM context
                new_code, features = await mutate_fn(parent.code)
            except Exception:
                continue

            # Z3 Safety Gate: validate mutation before expensive evaluation
            if self._z3 and self.config.z3_constraints:
                result = self._z3.validate_mutation(self.config.z3_constraints)
                if not result.safe:
                    self.z3_rejections += 1
                    log.debug("Z3 rejected mutation: %s", result.violations)
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
            "z3_rejections": self.z3_rejections,
            "population_size": self._population.size(),
            "coverage": self._population.coverage(),
            "best_score": best.score if best else 0.0,
            "best_id": best.id if best else None,
            "is_warm_up": self.total_mutations < self.config.hard_warm_start_threshold,
            "dgm_entropy": self._dgm_solver.stats()["entropy"]
        }
