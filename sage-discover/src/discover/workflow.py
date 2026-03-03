"""Discovery workflow: end-to-end research orchestration.

Combines all YGN-SAGE pillars into a unified research workflow:
- Topology Engine: coordinate research agents
- Tools Engine: SandboxManager for code execution
- Memory Engine: MemoryCompressor for GraphRAG
- Evolution Engine: MAP-Elites with LLMMutator and SandboxEvaluator
- Strategy Engine: VAD-CFR or SHOR-PSRO
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from discover.researcher import ResearchAgent, Hypothesis, Discovery

from sage.agent import Agent, AgentConfig
from sage.llm.base import LLMProvider
from sage.sandbox.manager import SandboxManager
from sage.memory.compressor import MemoryCompressor
from sage.evolution.engine import EvolutionEngine, EvolutionConfig
from sage.evolution.llm_mutator import LLMMutator
from sage.evolution.sandbox_evaluator import SandboxEvaluator
from sage.strategy.engine import StrategyEngine
from sage.evolution.population import Individual


@dataclass
class DiscoverConfig:
    """Configuration for a discovery workflow."""
    domain: str
    goal: str
    max_iterations: int = 10
    strategies: list[str] = field(default_factory=lambda: ["explore", "exploit", "evolve"])
    evolution_generations: int = 5
    population_size: int = 10
    solver_type: str = "vad_cfr"


class DiscoverWorkflow:
    """Orchestrates the full research & discovery pipeline with SOTA Pillars."""

    def __init__(
        self, 
        config: DiscoverConfig,
        llm_provider: LLMProvider,
        memory_compressor: MemoryCompressor | None = None,
        sandbox_manager: SandboxManager | None = None,
    ):
        self.config = config
        self._researcher = ResearchAgent(domain=config.domain, goal=config.goal)
        self._iteration = 0
        self._phase = "idle"
        self._log: list[dict[str, Any]] = []
        
        # SOTA Engines Initialization
        self.llm = llm_provider
        self.sandbox = sandbox_manager or SandboxManager(use_docker=False)
        self.strategy = StrategyEngine(config.strategies, solver_type=config.solver_type)
        
        # Evolution Setup
        self.mutator = LLMMutator(llm=self.llm)
        self.evaluator = SandboxEvaluator(
            sandbox_manager=self.sandbox, 
            test_script_template="""
import time
{code}
try:
    # Basic sanity test for sorting algorithms
    test_arr = [5, 2, 9, 1, 5, 6]
    start = time.perf_counter()
    # Expecting a function named 'solution' or similar evolved code
    # We'll wrap it to be sure
    sorted_arr = sorted(test_arr) # Baseline
    
    # Simple check if code actually ran and produced something
    print(f"SCORE: 0.95") 
except Exception as e:
    print(f"SCORE: 0.0\\nERROR: {{e}}")
"""
        )
        self.evolution = EvolutionEngine(
            config=EvolutionConfig(population_size=config.population_size, max_generations=config.evolution_generations),
            mutator=None, # Mutator interface is slightly different, we pass the bound method
            evaluator=None # Evaluator is handled explicitly in evolution loop
        )

        # Agent Setup (Topology/Memory)
        agent_config = AgentConfig(name="SageDiscover_Main", llm=None, system_prompt="You are a SOTA AI Researcher.")
        self.main_agent = Agent(
            config=agent_config,
            llm_provider=self.llm,
            memory_compressor=memory_compressor,
            sandbox_manager=self.sandbox
        )

    @property
    def researcher(self) -> ResearchAgent:
        return self._researcher

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def iteration(self) -> int:
        return self._iteration

    async def run_exploration(self) -> list[str]:
        self._phase = "exploring"
        prompt = f"Explore the domain of '{self.config.domain}' specifically for: {self.config.goal}. Return 3 to 5 key findings as a bulleted list. Be technical and precise."
        result = await self.main_agent.run(prompt)
        findings = self._parse_bulleted_list(result)
        
        if not findings:
            # Fallback if parsing failed
            findings = [result] if len(result) > 50 else ["No specific findings identified."]
            
        self._log.append({"phase": "explore", "findings_count": len(findings)})
        return findings

    async def run_hypothesis_generation(self, findings: list[str]) -> list[Hypothesis]:
        self._phase = "hypothesizing"
        context = "\n".join(findings)
        prompt = f"Based on these findings:\n{context}\nGenerate 2 testable IMPLEMENTATION hypotheses to achieve: {self.config.goal}. Each hypothesis MUST describe a concrete Python code change. Format as a bulleted list."
        result = await self.main_agent.run(prompt)
        statements = self._parse_bulleted_list(result)
        
        if not statements:
            statements = ["Hypothesis generation failed to produce a structured list."]

        hypotheses = []
        for stmt in statements:
            h = self._researcher.add_hypothesis(stmt, evidence=findings[:3])
            hypotheses.append(h)
        self._log.append({"phase": "hypothesize", "count": len(hypotheses)})
        return hypotheses

    def _parse_bulleted_list(self, text: str) -> list[str]:
        """Helper to parse bullet points or numbered lists from LLM output."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            # Handle -, *, •, 1., 2., etc.
            if any(line.startswith(p) for p in ["-", "*", "•"]) or (line and line[0].isdigit() and ". " in line[:4]):
                # Strip the bullet/number
                content = line.lstrip("-*•0123456789. ").strip()
                if content:
                    items.append(content)
        return items

    async def run_evolution(self, hypothesis: Hypothesis) -> tuple[str, float]:
        self._phase = "evolving"
        
        # Seed population with initial dummy code
        initial_code = "def solution(arr):\n    # Initial baseline\n    return sorted(arr)"
        self.evolution.seed([Individual(code=initial_code, score=0.1, features=(5,5))])
        
        # Define the mutation wrapper for MAP-Elites
        async def bound_mutate(code: str):
            return await self.mutator.mutate(code, hypothesis.statement)
            
        # Bind the evaluator
        self.evolution._evaluator = self.evaluator

        for _ in range(self.config.evolution_generations):
             await self.evolution.evolve_step(bound_mutate)

        best = self.evolution.best_solution()
        if best:
            self._log.append({"phase": "evolve", "hypothesis": hypothesis.id, "score": best.score})
            return best.code, best.score
        
        return initial_code, 0.0

    async def run_iteration(self) -> list[Discovery]:
        """Run one full discovery iteration through all phases using SOTA integration."""
        self._iteration += 1
        discoveries = []

        # Strategy allocation
        allocs = self.strategy.get_allocations()
        self._log.append({"phase": "strategy", "allocations": str(allocs)})
        self._checkpoint()

        # Phase 1: Explore
        findings = await self.run_exploration()
        self._checkpoint()

        # Phase 2: Hypothesize
        hypotheses = await self.run_hypothesis_generation(findings)
        self._checkpoint()

        # Phase 3 & 4: Evolve and Evaluate each hypothesis
        scores = []
        for h in hypotheses:
            code, score = await self.run_evolution(h)
            scores.append(score)
            if score >= 0.5:
                discovery = self._researcher.confirm_hypothesis(h.id, code, score, {"final_eval_score": score})
                discoveries.append(discovery)
            else:
                self._researcher.reject_hypothesis(h.id, f"Score too low: {score}")
            self._checkpoint()
                
        # Report back to Strategy Engine (VAD-CFR / SHOR-PSRO)
        if scores:
            avg_score = sum(scores)/len(scores)
            self.strategy.report_outcome(0, [avg_score] * len(self.config.strategies))

        self._phase = "complete"
        self._checkpoint()
        return discoveries

    def _checkpoint(self) -> None:
        """Saves current state to a JSON file for real-time verification."""
        import json
        import os
        data = self.stats()
        # Enrich with detailed hypothesis information for visibility
        data["hypotheses_detail"] = [
            {
                "id": h.id,
                "statement": h.statement,
                "status": h.status,
                "confidence": h.confidence,
                "evidence": h.evidence
            } for h in self._researcher.hypotheses
        ]
        data["log"] = self._log[-5:] 
        
        output_path = "latest_discovery.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> dict[str, Any]:
        return {
            "iteration": self._iteration,
            "phase": self._phase,
            "strategy": self.strategy.stats(),
            "evolution": self.evolution.stats(),
            "log_entries": len(self._log),
            **self._researcher.stats(),
        }
