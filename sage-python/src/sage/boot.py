"""Boot sequence: initialize the full YGN-SAGE agent stack."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Load .env if present (for GOOGLE_API_KEY etc.)
try:
    from dotenv import load_dotenv
    # Walk up to find .env (works from sage-python/ or repo root)
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.agent_pool import AgentPool
from sage.llm.base import LLMConfig
from sage.llm.mock import MockProvider
from sage.llm.router import ModelRouter
from sage.strategy.metacognition import MetacognitiveController
from sage.topology.evo_topology import TopologyEvolver, TopologyPopulation
from sage.memory.memory_agent import MemoryAgent
from sage.tools.registry import ToolRegistry


@dataclass
class AgentSystem:
    """The complete YGN-SAGE agent system."""
    agent_loop: AgentLoop
    agent_pool: AgentPool
    metacognition: MetacognitiveController
    topology_evolver: TopologyEvolver
    topology_population: TopologyPopulation
    memory_agent: MemoryAgent
    tool_registry: ToolRegistry

    async def run(self, task: str) -> str:
        # 1. Assess task complexity
        profile = await self.metacognition.assess_complexity_async(task)
        decision = self.metacognition.route(profile)

        # 2. Apply routing decision
        current_provider = self.agent_loop.config.llm.provider

        # enforce_system3 only for Gemini (which supports <think> tags)
        # Codex CLI refuses <think> tags — it reasons internally
        actual_provider = self.agent_loop.config.llm.provider
        if actual_provider == "codex":
            self.agent_loop.config.enforce_system3 = False
        elif decision.system == 1:
            self.agent_loop.config.enforce_system3 = False
        else:
            self.agent_loop.config.enforce_system3 = True

        # Only switch LLM if the target provider is available AND different
        new_config = ModelRouter.get_config(decision.llm_tier)

        # Don't downgrade from Codex to Gemini Flash for simple tasks —
        # Codex handles everything and always produces <think> tags
        if current_provider == "codex" and new_config.provider == "google":
            pass  # Keep Codex
        elif new_config.provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
            pass  # Google unavailable, keep current
        else:
            self.agent_loop.config.llm = new_config
            if new_config.provider == "codex":
                from sage.llm.codex import CodexProvider
                self.agent_loop._llm = CodexProvider()
            elif new_config.provider == "google":
                from sage.llm.google import GoogleProvider
                self.agent_loop._llm = GoogleProvider()

        # 3. Run the agent loop
        return await self.agent_loop.run(task)


def boot_agent_system(
    use_mock_llm: bool = False,
    llm_tier: str = "auto",
    agent_name: str = "sage-main",
) -> AgentSystem:
    """Initialize the complete agent stack.

    Args:
        llm_tier: Model tier to use. "auto" (default) picks the best
                  available provider: Codex CLI if installed, else Google
                  Gemini if GOOGLE_API_KEY is set, else raises.
    """
    import shutil

    # LLM
    if use_mock_llm:
        provider = MockProvider(responses=["<think>Processing</think>\nDone."])
        llm_config = LLMConfig(provider="mock", model="mock")
    else:
        # Auto-detect best available provider
        if llm_tier == "auto":
            if shutil.which("codex"):
                llm_tier = "codex"
            elif os.environ.get("GOOGLE_API_KEY"):
                llm_tier = "fast"
            else:
                raise RuntimeError(
                    "No LLM provider available. Install Codex CLI or set GOOGLE_API_KEY."
                )

        llm_config = ModelRouter.get_config(llm_tier)
        if llm_config.provider == "codex":
            from sage.llm.codex import CodexProvider
            provider = CodexProvider()
        else:
            from sage.llm.google import GoogleProvider
            provider = GoogleProvider()

    # Components
    tool_registry = ToolRegistry()
    agent_pool = AgentPool()
    metacognition = MetacognitiveController()
    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    memory_agent = MemoryAgent(use_llm=not use_mock_llm)

    # Agent config
    config = AgentConfig(
        name=agent_name,
        llm=llm_config,
        system_prompt=(
            "You are YGN-SAGE, a precise AI assistant. "
            "Think step-by-step. Be concise. Answer the user task directly."
        ),
        max_steps=20,
        enforce_system3=(llm_config.provider != "codex"),
    )

    # Agent loop
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        tool_registry=tool_registry,
    )
    loop.agent_pool = agent_pool
    loop.metacognition = metacognition
    loop.topology_population = topology_population

    return AgentSystem(
        agent_loop=loop,
        agent_pool=agent_pool,
        metacognition=metacognition,
        topology_evolver=topology_evolver,
        topology_population=topology_population,
        memory_agent=memory_agent,
        tool_registry=tool_registry,
    )
