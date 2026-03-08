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
from sage.strategy.metacognition import ComplexityRouter
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
    metacognition: ComplexityRouter
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
        """Run a task through the agent system.

        Primary path: CognitiveOrchestrator (multi-provider, score-based).
        Fallback: legacy AgentLoop with ModelRouter (Codex + Google only).
        Mock mode: direct AgentLoop (no orchestrator).
        """
        # 1. Assess task complexity
        profile = await self.metacognition.assess_complexity_async(task)
        decision = self.metacognition.route(profile)

        # 1b. Speculative execution: detect indecisive zone
        if 0.35 <= profile.complexity <= 0.55 and decision.system <= 2:
            _log.info(
                "Speculative zone: complexity=%.2f (indecisive). "
                "Would fire S1+S2 in parallel when architecture supports it. "
                "Using S%d for now.",
                profile.complexity, decision.system,
            )

        # 2. Set validation level from routing decision
        if decision.system >= 3:
            self.agent_loop.config.validation_level = 3
        elif decision.system == 2 and self.agent_loop.sandbox_manager:
            self.agent_loop.config.validation_level = 2
        else:
            self.agent_loop.config.validation_level = 1

        current_provider = self.agent_loop.config.llm.provider

        # Mock mode: skip orchestrator, use AgentLoop directly
        if current_provider == "mock":
            return await self.agent_loop.run(task)

        # 3. Try CognitiveOrchestrator as primary path (multi-provider)
        if self.orchestrator and self.registry and self.registry.list_available():
            try:
                result = await self.orchestrator.run(task)
                await self._persist_memory()
                return result
            except Exception as e:
                _log.warning(
                    "Orchestrator failed (%s), falling back to legacy routing", e
                )

        # 4. Fallback: legacy ModelRouter path (Codex + Google only)
        new_config = ModelRouter.get_config(decision.llm_tier)
        if current_provider == "codex" and new_config.provider == "google":
            pass  # Don't downgrade from Codex to Gemini
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

        result = await self.agent_loop.run(task)
        await self._persist_memory()
        return result

    async def _persist_memory(self) -> None:
        """Persist semantic and causal memory after a run."""
        if hasattr(self.agent_loop, "semantic_memory") and self.agent_loop.semantic_memory:
            try:
                self.agent_loop.semantic_memory.save()
            except Exception:
                _log.warning("Failed to persist semantic memory", exc_info=True)
        if hasattr(self.agent_loop, "causal_memory") and self.agent_loop.causal_memory:
            try:
                self.agent_loop.causal_memory.save()
            except Exception:
                _log.warning("Failed to persist causal memory", exc_info=True)


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
    metacognition = ComplexityRouter(llm_provider=provider if not use_mock_llm else None)
    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    memory_agent = MemoryAgent(use_llm=not use_mock_llm, llm_provider=provider)

    # Memory compressor (fires on pressure — MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=20,
        keep_recent=5,
    )

    # Embedder for S-MMU semantic edges
    from sage.memory.embedder import Embedder
    memory_compressor.embedder = Embedder()

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

    # Episodic memory — defaults to persistent SQLite
    _ep_db = Path.home() / ".sage" / "episodic.db"
    _ep_db.parent.mkdir(parents=True, exist_ok=True)
    episodic_memory = EpisodicMemory(db_path=str(_ep_db))

    # Safety net: warn if someone overrides with db_path=None upstream
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
        validation_level=1,  # Default S1 — routing promotes to S2 only for code tasks
    )

    # Event bus (central nervous system)
    event_bus = event_bus or EventBus()

    # ModelRegistry: always created (even in mock mode) so callers can inspect it
    from sage.providers.registry import ModelRegistry
    registry = ModelRegistry()
    orchestrator = None

    if not use_mock_llm:
        from sage.orchestrator import CognitiveOrchestrator

        # Auto-discover available models at boot
        import asyncio
        try:
            try:
                _running_loop = asyncio.get_running_loop()
            except RuntimeError:
                _running_loop = None
            if _running_loop and _running_loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(registry.refresh())).result(timeout=15)
            else:
                asyncio.run(registry.refresh())
            _log.info(
                "Boot: discovered %d models (%d available)",
                len(registry.profiles),
                len(registry.list_available()),
            )
        except Exception as e:
            _log.warning("Boot: model discovery failed (%s), continuing with legacy routing", e)

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

    # Evolution disabled by default — Sprint 3 evidence shows marginal value
    # (0.50 best score vs 0.33 random mutation). Still available for explicit use.
    loop._auto_evolve = False

    # Semantic memory + MemoryAgent wiring (persistent SQLite in real mode)
    from sage.memory.semantic import SemanticMemory
    if not use_mock_llm:
        _sem_db = Path.home() / ".sage" / "semantic.db"
        _sem_db.parent.mkdir(parents=True, exist_ok=True)
        semantic_memory = SemanticMemory(db_path=str(_sem_db))
        semantic_memory.load()
    else:
        semantic_memory = SemanticMemory()
    loop.memory_agent = memory_agent  # Already created above but never injected!
    loop.semantic_memory = semantic_memory

    # Causal memory (persistent SQLite in real mode)
    from sage.memory.causal import CausalMemory
    if not use_mock_llm:
        _causal_db = Path.home() / ".sage" / "causal.db"
        _causal_db.parent.mkdir(parents=True, exist_ok=True)
        causal_memory = CausalMemory(db_path=str(_causal_db))
        causal_memory.load()
    else:
        causal_memory = CausalMemory()
    loop.causal_memory = causal_memory

    # AgeMem: 7 memory tools (3 STM + 4 LTM)
    for tool in create_memory_tools(loop.working_memory, episodic_memory, memory_compressor):
        tool_registry.register(tool)

    # ExoCortex tools (search)
    from sage.tools.exocortex_tools import create_exocortex_tools
    for tool in create_exocortex_tools(exocortex):
        tool_registry.register(tool)

    # Guardrails
    from sage.guardrails.base import GuardrailPipeline
    from sage.guardrails.builtin import CostGuardrail, OutputGuardrail
    loop.guardrail_pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),  # Default budget limit
        OutputGuardrail(min_length=1),  # Free-text output validation
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
