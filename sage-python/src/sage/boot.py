"""Boot sequence: initialize the full YGN-SAGE agent stack."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger("sage.boot")

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
from sage.memory.compressor import MemoryCompressor
from sage.sandbox.manager import SandboxManager
from sage.memory.episodic import EpisodicMemory
from sage.memory.remote_rag import ExoCortex
from sage.tools.memory_tools import create_memory_tools
from sage.events.bus import EventBus


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
    event_bus: EventBus
    # CognitiveOrchestrator (capability-based multi-provider routing)
    orchestrator: Any = None
    # ModelRegistry (live model discovery + TOML knowledge base)
    # Note: ModelRegistry is the capability-based selection system used by
    # CognitiveOrchestrator. ModelRouter (sage.llm.router) is the legacy
    # tier->config mapping used directly by AgentLoop. Both coexist:
    # the orchestrator tries ModelRegistry first, falling back to the
    # legacy ModelRouter path if no models are discovered.
    registry: Any = None

    async def run(self, task: str) -> str:
        # Initialize registry on first use (lazy async)
        if self.registry and not self.registry._profiles:
            try:
                await self.registry.refresh()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Registry refresh failed: %s", e)

        # Use CognitiveOrchestrator if available and has models
        if self.orchestrator and self.registry and self.registry.list_available():
            return await self.orchestrator.run(task)

        # Fallback: legacy direct routing via ModelRouter
        # 1. Assess task complexity
        profile = await self.metacognition.assess_complexity_async(task)
        decision = self.metacognition.route(profile)

        # 2. Apply routing decision
        current_provider = self.agent_loop.config.llm.provider

        # Set validation level based on routing decision
        actual_provider = self.agent_loop.config.llm.provider
        if actual_provider == "codex":
            # Codex CLI reasons internally — no external validation needed
            self.agent_loop.config.validation_level = 1
        else:
            self.agent_loop.config.validation_level = decision.validation_level

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
    event_bus: EventBus | None = None,
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

    # Memory compressor (fires on pressure — MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=20,
        keep_recent=5,
    )

    # Runtime tool synthesis (Agent0/AutoTool pattern)
    from sage.tools.meta import create_python_tool, create_bash_tool
    tool_registry.register(create_python_tool)
    tool_registry.register(create_bash_tool)

    # Sandbox manager for S2 empirical validation
    # SECURITY: local host execution disabled by default (requires allow_local=True)
    sandbox_manager = SandboxManager()

    # --- Degradation warnings (loud, not silent) ---
    from sage.memory.working import _has_rust as _rust_available
    if not _rust_available:
        _log.warning(
            "sage_core Rust extension not compiled — working memory uses a "
            "pure-Python mock that returns dummy values for Arrow/S-MMU "
            "operations. Build with: cd sage-core && maturin develop"
        )

    # Episodic memory
    episodic_memory = EpisodicMemory()

    if not episodic_memory._db_path:
        _log.warning(
            "Episodic memory is volatile (in-memory only, data lost on "
            "restart). Pass db_path to EpisodicMemory for persistence."
        )

    # ExoCortex (persistent RAG via Google GenAI File Search)
    exocortex = ExoCortex()

    # Agent config
    config = AgentConfig(
        name=agent_name,
        llm=llm_config,
        system_prompt=(
            "You are YGN-SAGE, a precise AI assistant. "
            "Think step-by-step. Be concise. Answer the user task directly."
        ),
        max_steps=20,
        validation_level=1 if llm_config.provider == "codex" else 2,
    )

    # Event bus (central nervous system)
    event_bus = event_bus or EventBus()

    # CognitiveOrchestrator (capability-based multi-provider routing)
    # Only wired for real LLM providers — mock mode uses legacy direct routing
    registry = None
    orchestrator = None
    if not use_mock_llm:
        from sage.providers.registry import ModelRegistry
        from sage.orchestrator import CognitiveOrchestrator
        registry = ModelRegistry()
        orchestrator = CognitiveOrchestrator(
            registry=registry, metacognition=metacognition, event_bus=event_bus,
        )

    # Agent loop
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
        on_event=event_bus.emit,
    )
    loop.agent_pool = agent_pool
    loop.metacognition = metacognition
    loop.topology_population = topology_population
    loop.episodic_memory = episodic_memory
    loop.sandbox_manager = sandbox_manager
    loop.exocortex = exocortex

    # Semantic memory + MemoryAgent wiring
    from sage.memory.semantic import SemanticMemory
    loop.memory_agent = memory_agent  # Already created above but never injected!
    loop.semantic_memory = SemanticMemory()

    # AgeMem: 7 memory tools (3 STM + 4 LTM)
    for tool in create_memory_tools(loop.working_memory, episodic_memory, memory_compressor):
        tool_registry.register(tool)

    # ExoCortex tools (search)
    from sage.tools.exocortex_tools import create_exocortex_tools
    for tool in create_exocortex_tools(exocortex):
        tool_registry.register(tool)

    # Guardrails
    from sage.guardrails.base import GuardrailPipeline
    from sage.guardrails.builtin import CostGuardrail
    loop.guardrail_pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),  # Default budget limit
    ])

    return AgentSystem(
        agent_loop=loop,
        agent_pool=agent_pool,
        metacognition=metacognition,
        topology_evolver=topology_evolver,
        topology_population=topology_population,
        memory_agent=memory_agent,
        tool_registry=tool_registry,
        event_bus=event_bus,
        orchestrator=orchestrator,
        registry=registry,
    )
