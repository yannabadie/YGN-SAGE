"""Discovery workflow: end-to-end research orchestration.

Combines all YGN-SAGE pillars into a unified research workflow:
- Topology Engine: coordinate research agents
- Tools Engine: search papers, run experiments
- Memory Engine: accumulate research knowledge
- Evolution Engine: evolve solutions through code mutation
- Strategy Engine: allocate resources across research strategies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from discover.researcher import ResearchAgent, Hypothesis, Discovery


@dataclass
class DiscoverConfig:
    """Configuration for a discovery workflow."""
    domain: str
    goal: str
    max_iterations: int = 10
    strategies: list[str] = field(default_factory=lambda: ["explore", "exploit", "evolve"])
    evolution_generations: int = 20
    population_size: int = 50


class DiscoverWorkflow:
    """Orchestrates the full research & discovery pipeline.

    Workflow phases:
    1. Exploration: gather background, identify open problems
    2. Hypothesis Generation: form testable claims
    3. Solution Evolution: evolve code solutions to test hypotheses
    4. Evaluation: score solutions through progressive evaluation
    5. Knowledge Integration: store discoveries in memory
    """

    def __init__(self, config: DiscoverConfig):
        self.config = config
        self._researcher = ResearchAgent(domain=config.domain, goal=config.goal)
        self._iteration = 0
        self._phase = "idle"
        self._log: list[dict[str, Any]] = []

    @property
    def researcher(self) -> ResearchAgent:
        return self._researcher

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def iteration(self) -> int:
        return self._iteration

    async def run_exploration(
        self,
        explore_fn: Callable[[str], Awaitable[list[str]]],
    ) -> list[str]:
        """Phase 1: Explore the domain and gather findings.

        Args:
            explore_fn: Async function that takes a goal and returns findings.
        """
        self._phase = "exploring"
        findings = await explore_fn(self.config.goal)
        self._log.append({"phase": "explore", "findings_count": len(findings)})
        return findings

    async def run_hypothesis_generation(
        self,
        generate_fn: Callable[[str, list[str]], Awaitable[list[str]]],
        findings: list[str],
    ) -> list[Hypothesis]:
        """Phase 2: Generate hypotheses from findings.

        Args:
            generate_fn: Async function that takes (goal, findings) and returns hypothesis strings.
        """
        self._phase = "hypothesizing"
        statements = await generate_fn(self.config.goal, findings)
        hypotheses = []
        for stmt in statements:
            h = self._researcher.add_hypothesis(stmt, evidence=findings[:3])
            hypotheses.append(h)
        self._log.append({"phase": "hypothesize", "count": len(hypotheses)})
        return hypotheses

    async def run_evolution(
        self,
        evolve_fn: Callable[[Hypothesis], Awaitable[tuple[str, float]]],
        hypothesis: Hypothesis,
    ) -> tuple[str, float]:
        """Phase 3: Evolve a solution for a hypothesis.

        Args:
            evolve_fn: Async function that takes a hypothesis and returns (code, score).
        """
        self._phase = "evolving"
        code, score = await evolve_fn(hypothesis)
        self._log.append({"phase": "evolve", "hypothesis": hypothesis.id, "score": score})
        return code, score

    async def run_evaluation(
        self,
        evaluate_fn: Callable[[str], Awaitable[float]],
        code: str,
        hypothesis: Hypothesis,
    ) -> Discovery | None:
        """Phase 4: Evaluate evolved code and confirm/reject hypothesis.

        Args:
            evaluate_fn: Async function that takes code and returns a score.
        """
        self._phase = "evaluating"
        score = await evaluate_fn(code)
        self._log.append({"phase": "evaluate", "hypothesis": hypothesis.id, "score": score})

        if score >= 0.5:
            discovery = self._researcher.confirm_hypothesis(
                hypothesis.id, code, score, {"final_eval_score": score}
            )
            return discovery
        else:
            self._researcher.reject_hypothesis(hypothesis.id, f"Score too low: {score}")
            return None

    async def run_iteration(
        self,
        explore_fn: Callable[[str], Awaitable[list[str]]],
        generate_fn: Callable[[str, list[str]], Awaitable[list[str]]],
        evolve_fn: Callable[[Hypothesis], Awaitable[tuple[str, float]]],
        evaluate_fn: Callable[[str], Awaitable[float]],
    ) -> list[Discovery]:
        """Run one full discovery iteration through all phases."""
        self._iteration += 1
        discoveries = []

        # Phase 1: Explore
        findings = await self.run_exploration(explore_fn)

        # Phase 2: Hypothesize
        hypotheses = await self.run_hypothesis_generation(generate_fn, findings)

        # Phase 3 & 4: Evolve and Evaluate each hypothesis
        for h in hypotheses:
            code, score = await self.run_evolution(evolve_fn, h)
            discovery = await self.run_evaluation(evaluate_fn, code, h)
            if discovery:
                discoveries.append(discovery)

        self._phase = "complete"
        return discoveries

    def stats(self) -> dict[str, Any]:
        """Get workflow statistics."""
        return {
            "iteration": self._iteration,
            "phase": self._phase,
            "log_entries": len(self._log),
            **self._researcher.stats(),
        }
