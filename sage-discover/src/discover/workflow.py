"""Discovery workflow: end-to-end research orchestration.

Combines all YGN-SAGE pillars into a unified research workflow:
- Topology Engine: coordinate research agents
- Tools Engine: SandboxManager for code execution
- Memory Engine: WorkingMemory & EpisodicMemory
- Evolution Engine: MAP-Elites optimization
- Strategy Engine: PSRO/VAD-CFR solvers
- Metacognitive Pillar: OpenAI Codex 5.3 X-High review
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from discover.researcher import ResearchAgent, Hypothesis, Discovery
from discover.knowledge import NotebookLMBridge

from sage.agent import Agent, AgentConfig
from sage.llm.base import LLMProvider
from sage.llm.router import ModelRouter
from sage.llm.codex import CodexExecProvider
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
        self._bridge = NotebookLMBridge()
        self.codex = CodexExecProvider(effort="xhigh")
        self._iteration = 0
        self._phase = "idle"
        self._log: list[dict[str, Any]] = []
        
        # Ensure Journal Directory exists
        os.makedirs("research_journal", exist_ok=True)
        
        # SOTA Engines Initialization
        self.llm = llm_provider
        self.sandbox = sandbox_manager or SandboxManager(use_docker=False)
        self.strategy = StrategyEngine(config.strategies, solver_type=config.solver_type)
        
        # Evolution Setup: Use PRO model for high-precision mutations
        self.mutator = LLMMutator(llm=self.llm)
        self.mutator_config = ModelRouter.get_config("critical")
        
        # Reference H96 Implementation for relative scoring
        self.h96_ref = """
import numpy as np
def reference_h96(arr):
    arr = np.array(arr)
    if len(arr) <= 1: return arr.tolist()
    pivot = arr[len(arr)//2]
    return reference_h96(arr[arr < pivot]) + arr[arr == pivot].tolist() + reference_h96(arr[arr > pivot])
"""

        self.evaluator = SandboxEvaluator(
            sandbox_manager=self.sandbox, 
            test_script_template=self.h96_ref + """
import time
import random
import numpy as np

# Evolved Code
{code}

def benchmark():
    try:
        n = 2000
        test_arr = list(range(n))
        random.shuffle(test_arr)
        
        # 1. Baseline: H96 Reference
        start_base = time.perf_counter()
        expected = reference_h96(test_arr)
        base_time = time.perf_counter() - start_base
        
        # 2. Test Evolved Solution
        if 'solution' not in globals():
            print("SCORE: 0.0\\nERROR: Function 'solution' not found")
            return
            
        start_evolve = time.perf_counter()
        result = solution(test_arr)
        evolve_time = time.perf_counter() - start_evolve
        
        # Validation
        if result != expected:
            print("SCORE: 0.0\\nERROR: Incorrect sorting result")
            return

        # Score calculation: ratio relative to H96
        score = base_time / (evolve_time + 1e-9)
        print(f"SCORE: {{score:.4f}}")
        
    except Exception as e:
        print(f"SCORE: 0.0\\nERROR: {{e}}")

if __name__ == "__main__":
    benchmark()
"""
        )
        self.evolution = EvolutionEngine(
            config=EvolutionConfig(population_size=config.population_size, max_generations=config.evolution_generations),
            mutator=None,
            evaluator=None
        )

        # Agent Setup
        agent_config = AgentConfig(name="SageDiscover_Main", llm=ModelRouter.get_config("fast"), system_prompt="You are a SOTA AI Researcher.")
        self.main_agent = Agent(
            config=agent_config,
            llm_provider=self.llm,
            memory_compressor=memory_compressor,
            sandbox_manager=self.sandbox
        )

    async def run_exploration(self) -> list[str]:
        self._phase = "exploring"
        self.main_agent.config.llm = ModelRouter.get_config("fast")
        
        if self._bridge.is_active:
            insights = await self._bridge.get_sota_insights(self.config.domain)
            if insights and not insights[0].startswith("Error"):
                return insights

        synthesis_path = "docs/plans/notebooklm_research_synthesis.md"
        if os.path.exists(synthesis_path):
            with open(synthesis_path, "r", encoding="utf-8") as f:
                content = f.read()
                prompt = f"Analyze this SOTA synthesis:\\n{content}\\nExtract 3-5 technical findings relevant to: {self.config.goal}. Be technical and precise."
                result = await self.main_agent.run(prompt)
                return self._parse_bulleted_list(result)

        prompt = f"Explore the domain of '{self.config.domain}' for: {self.config.goal}. Return 3-5 key findings."
        result = await self.main_agent.run(prompt)
        return self._parse_bulleted_list(result)

    async def run_hypothesis_generation(self, findings: list[str]) -> list[Hypothesis]:
        self._phase = "hypothesizing"
        self.main_agent.config.llm = ModelRouter.get_config("critical")
        
        context = "\\n".join(findings)
        prompt = f"Based on these findings:\\n{context}\\nGenerate 2 testable IMPLEMENTATION hypotheses to achieve: {self.config.goal}. Each hypothesis MUST describe a concrete Python/NumPy code change. Format as a bulleted list."
        result = await self.main_agent.run(prompt)
        statements = self._parse_bulleted_list(result)
        
        hypotheses = []
        for stmt in statements:
            h = self._researcher.add_hypothesis(stmt, evidence=findings[:3])
            hypotheses.append(h)
        return hypotheses

    async def run_evolution(self, hypothesis: Hypothesis) -> tuple[str, float, dict]:
        self._phase = "evolving"
        
        # Initial code is H96
        initial_code = "import numpy as np\\ndef solution(arr):\\n    arr = np.array(arr)\\n    if len(arr) <= 1: return arr.tolist()\\n    pivot = arr[len(arr)//2]\\n    return solution(arr[arr < pivot]) + arr[arr == pivot].tolist() + solution(arr[arr > pivot])"
        
        self.evolution._evaluator = self.evaluator
        base_score = 1.0 
        self.evolution.seed([Individual(code=initial_code, score=base_score, features=(8,8))])
        
        async def bound_mutate(code: str):
            return await self.mutator.mutate(code, hypothesis.statement, config=self.mutator_config)
            
        for _ in range(self.config.evolution_generations):
             await self.evolution.evolve_step(bound_mutate)

        best = self.evolution.best_solution()
        if best:
            # Metacognitive Review via Codex 5.3
            print(f"🔍 Codex 5.3 X-High reviewing {hypothesis.id}...")
            review = await self.codex.review_code(best.code, hypothesis.statement)
            struct_score = review.get("structural_score", 0.0)
            
            # Combine scores: 60% Perf, 40% Structure
            combined_score = (best.score * 0.6) + (struct_score * 0.4)
            
            # ARCHIVE EVERYTHING (Zero Loss)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"research_journal/{hypothesis.id}_{timestamp}.json"
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump({
                    "hypothesis": hypothesis.statement,
                    "code": best.code,
                    "perf_score": best.score,
                    "structural_score": struct_score,
                    "combined_score": combined_score,
                    "review": review
                }, f, indent=2)
                
            return best.code, combined_score, review
            
        return initial_code, base_score, {}

    async def run_iteration(self) -> list[Discovery]:
        self._iteration += 1
        findings = await self.run_exploration()
        hypotheses = await self.run_hypothesis_generation(findings)
        
        iteration_discoveries = []
        for h in hypotheses:
            code, score, review = await self.run_evolution(h)
            if score > 1.05: 
                d = self._researcher.confirm_hypothesis(h.id, code, score)
                iteration_discoveries.append(d)
            else:
                self._researcher.reject_hypothesis(h.id, f"Score: {score:.4f}")
        
        if iteration_discoveries:
            self.strategy.report_outcome(2, [0.1, 0.1, 1.0])
        else:
            self.strategy.report_outcome(0, [0.1, 0.1, 0.1])
            
        self._checkpoint()
        return iteration_discoveries

    def _checkpoint(self):
        data = self.stats()
        with open("latest_discovery.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> dict[str, Any]:
        return {
            "iteration": self._iteration,
            "phase": self._phase,
            "strategy": self.strategy.stats(),
            "evolution": self.evolution.stats(),
            **self._researcher.stats(),
        }

    def _parse_bulleted_list(self, text: str) -> list[str]:
        items = []
        for line in text.split("\\n"):
            line = line.strip()
            if line.startswith(("- ", "* ", "1. ", "2. ", "3. ")):
                content = line.lstrip("-* 123456789. ").strip()
                if content:
                    items.append(content)
        if not items and len(text.strip()) > 20:
            items.append(text.strip())
        return items
