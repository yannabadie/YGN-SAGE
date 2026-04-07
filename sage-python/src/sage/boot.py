"""Boot sequence: initialize the full YGN-SAGE agent stack."""
from __future__ import annotations

import logging
import os
import re
import time
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

from sage.agent import AgentConfig  # noqa: E402
from sage.agent_loop import AgentLoop  # noqa: E402
from sage.agent_pool import AgentPool  # noqa: E402
from sage.llm.base import LLMConfig  # noqa: E402
from sage.llm.router import ModelRouter  # noqa: E402
from sage.strategy.adaptive_router import AdaptiveRouter  # noqa: E402
# Legacy fallback (34% GT) — kNN (92%) is primary. Kept for backward compatibility.
from sage.strategy.metacognition import ComplexityRouter  # noqa: E402
from sage.topology.evo_topology import TopologyEvolver, TopologyPopulation  # noqa: E402
from sage.tools.registry import ToolRegistry  # noqa: E402
from sage.events.bus import EventBus  # noqa: E402
from sage.routing.shadow import ShadowRouter  # noqa: E402
from sage.constants import (  # noqa: E402
    DEFAULT_BUDGET_USD,
    EXPLORATION_BUDGET_LOW,
    EXPLORATION_BUDGET_HIGH,
    MAX_TOPOLOGY_AGENTS,
    LLM_SYNTHESIS_MIN_SYSTEM,
    COST_GUARDRAIL_MAX_USD,
    OUTPUT_GUARDRAIL_MIN_LENGTH,
    MAX_AGENT_STEPS,
    SPECULATIVE_ZONE_MIN,
    SPECULATIVE_ZONE_MAX,
)

# Sub-modules (extracted for readability)
from sage.boot_providers import init_llm_provider  # noqa: E402
from sage.boot_tools import (  # noqa: E402
    init_tools, check_sandbox_availability, init_metacognition,
)
from sage.boot_memory import init_memory  # noqa: E402
from sage.boot_topology import init_topology  # noqa: E402
from sage.boot_pipeline import init_pipeline  # noqa: E402

# Re-export for backward compat (test_boot_sandbox_warning imports this)
_check_sandbox_availability = check_sandbox_availability


@dataclass
class AgentSystem:
    """The complete YGN-SAGE agent system."""
    agent_loop: AgentLoop
    agent_pool: AgentPool
    metacognition: AdaptiveRouter | ComplexityRouter
    topology_evolver: TopologyEvolver
    topology_population: TopologyPopulation
    memory_agent: Any
    tool_registry: ToolRegistry
    event_bus: EventBus
    # Legacy field — kept for backward compat, always None.
    orchestrator: Any = None
    # ModelRegistry (live model discovery + TOML knowledge base)
    registry: Any = None
    # CapabilityMatrix: semantic capability lookup for discovered providers
    capability_matrix: Any = None
    # Rust SystemRouter (None if sage_core not compiled with cognitive engine)
    rust_router: Any = None
    # ShadowRouter: dual Rust/Python routing comparison (None if shadow mode inactive)
    shadow_router: ShadowRouter | None = None
    # Phase 6: Rust TopologyEngine (6-path generate, MAP-Elites, bandit)
    topology_engine: Any = None
    # Phase 6: Standalone ContextualBandit for model selection
    bandit: Any = None
    # Rust ModelRegistry (None if sage_core not compiled or cards.toml not found)
    _rust_registry: Any = None
    # CognitiveOrchestrationPipeline (5-stage: classify->decompose->topology->assign->execute)
    pipeline: Any = None
    # Execution path tracking (Issue A audit fix): set after each run()
    _last_execution_path: str = ""

    @property
    def model_info(self) -> dict[str, str]:
        """Return resolved model metadata for benchmark artifacts."""
        info: dict[str, str] = {"model": "unknown", "provider": "", "tier": ""}
        loop = self.agent_loop
        if hasattr(loop, "_llm") and loop._llm:
            info["model"] = getattr(loop._llm, "model_id", "unknown")
            info["provider"] = type(loop._llm).__name__
        if hasattr(self, "metacognition") and self.metacognition:
            info["tier"] = getattr(self.metacognition, "_current_tier", "")
        return info

    async def run(self, task: str) -> str:
        """Run a task through the agent system.

        Primary path: CognitiveOrchestrationPipeline (5-stage).
        Secondary path: CognitiveOrchestrator (multi-provider, score-based).
        Fallback: legacy AgentLoop with ModelRouter (Codex + Google only).
        Mock mode: direct AgentLoop.
        """
        # Try CognitiveOrchestrationPipeline first (if wired and not mock)
        # Mock mode bypasses the pipeline so agent_loop phases (PERCEIVE/THINK/ACT/LEARN)
        # are exercised — important for testing phase events and guardrail wiring.
        if self.pipeline and self.agent_loop.config.llm.provider != "mock":
            try:
                _budget = self._guardrail_budget if hasattr(self, '_guardrail_budget') else DEFAULT_BUDGET_USD
                result = await self.pipeline.run(task, budget_usd=_budget)
                self._last_execution_path = "pipeline"
                await self._persist_memory()
                _log.info("Execution path: pipeline (5-stage)")
                return result
            except (RuntimeError, TimeoutError) as exc:
                self._last_execution_path = "pipeline_fallback_legacy"
                _log.warning("Pipeline FAILED — falling back to legacy path: %s", exc)

        _run_start = time.perf_counter()
        if not self._last_execution_path:
            self._last_execution_path = "legacy"
            _log.info("Execution path: legacy (pipeline unavailable or mock mode)")
        self._last_decision = None  # Routing decision for telemetry feedback
        # Reset topology so each run generates a fresh one.
        # Externally-forced topologies (TopologyBench) are re-set by the caller
        # before each run() call, so this is safe.
        self.agent_loop._current_topology = None

        # 1. Route task to cognitive system
        budget = DEFAULT_BUDGET_USD
        if hasattr(self, '_guardrail_budget'):
            budget = self._guardrail_budget

        if self.shadow_router:
            # Shadow mode: runs both routers, returns primary decision
            decision = await self.shadow_router.route(task, budget)
            # Determine which router produced the primary decision
            if self.rust_router:
                system_num = int(decision.system)
                model_id = decision.model_id
                _log.info(
                    "Shadow routing (Rust primary): %s -> S%d, model=%s (conf=%.2f, cost=%.4f)",
                    task[:60], system_num, model_id,
                    decision.confidence, decision.estimated_cost,
                )
            else:
                system_num = decision.system
                model_id = None
                # Speculative zone detection (Python-only path via ShadowRouter)
                profile = await self.metacognition.assess_complexity_async(task)
                if SPECULATIVE_ZONE_MIN <= profile.complexity <= SPECULATIVE_ZONE_MAX and decision.system <= 2:
                    _log.info(
                        "Speculative zone: complexity=%.2f (indecisive). Using S%d for now.",
                        profile.complexity, decision.system,
                    )
        elif self.rust_router:
            # Primary path: Rust SystemRouter (no shadow)
            decision = self.rust_router.route(task, budget)
            system_num = int(decision.system)  # CognitiveSystem enum -> int
            model_id = decision.model_id
            _log.info(
                "Rust routing: %s -> system=S%d, model=%s (conf=%.2f, cost=%.4f)",
                task[:60], system_num, model_id,
                decision.confidence, decision.estimated_cost,
            )
        else:
            # Fallback: Python AdaptiveRouter
            profile = await self.metacognition.assess_complexity_async(task)
            decision = self.metacognition.route(profile)
            system_num = decision.system
            model_id = None  # Python path uses llm_tier, not model_id
            # Speculative execution detection
            if SPECULATIVE_ZONE_MIN <= profile.complexity <= SPECULATIVE_ZONE_MAX and decision.system <= 2:
                _log.info(
                    "Speculative zone: complexity=%.2f (indecisive). Using S%d for now.",
                    profile.complexity, decision.system,
                )

        self._last_decision = decision  # Store for telemetry in _record_topology_outcome

        # 2. Topology generation (Rust engine, 6-path strategy)
        #    If _current_topology is already set (externally forced by benchmark
        #    scripts or TopologyBench), skip generation to preserve the forced topology.
        topology_result = None
        _externally_forced = self.agent_loop._current_topology
        if _externally_forced:
            _log.info(
                "Topology externally forced (%d nodes, template=%s), skipping generation",
                _externally_forced.node_count(),
                getattr(_externally_forced, 'template_type', 'unknown'),
            )
        elif self.topology_engine:
            try:
                exploration_budget = EXPLORATION_BUDGET_LOW if system_num <= 2 else EXPLORATION_BUDGET_HIGH
                # Compute real embedding for S-MMU semantic retrieval
                task_embedding = None
                try:
                    from sage.memory.embedder import Embedder
                    _emb = Embedder()
                    if _emb.is_semantic:
                        task_embedding = _emb.embed(task[:500])
                except (ImportError, RuntimeError):
                    pass
                topology_result = self.topology_engine.generate(
                    task,
                    task_embedding,
                    system_num,
                    exploration_budget,
                )
                # Cache topology for later outcome recording
                self.topology_engine.cache_topology(topology_result.topology)
                self.agent_loop._current_topology = topology_result.topology
                _log.info(
                    "Topology generated: source=%s, confidence=%.2f, template=%s",
                    topology_result.source,
                    topology_result.confidence,
                    topology_result.topology.template_type,
                )
                # Path 3 hook: if engine returned template_fallback AND
                # system >= 2 AND not mock, try LLM synthesis
                if (topology_result.source == "template_fallback"
                        and system_num >= LLM_SYNTHESIS_MIN_SYSTEM
                        and self.agent_loop.config.llm.provider != "mock"):
                    try:
                        from sage.topology.llm_caller import synthesize_topology
                        llm_graph = await synthesize_topology(
                            self.agent_loop._llm,
                            task,
                            max_agents=MAX_TOPOLOGY_AGENTS,
                            available_models=["gemini-2.5-flash", "gemini-3-flash-preview"],
                        )
                        if llm_graph and llm_graph.node_count() > 0:
                            self.topology_engine.cache_topology(llm_graph)
                            # Update topology_result with LLM synthesis result
                            topology_result = self.topology_engine.generate(
                                task, None, system_num, 0.0,
                            )
                            _log.info("Path 3: LLM synthesis produced %d-node topology",
                                      llm_graph.node_count())
                    except (RuntimeError, TimeoutError) as e:
                        _log.debug("Path 3 LLM synthesis skipped: %s", e)
                # Emit topology event for dashboard
                from sage.agent_loop import AgentEvent
                self.event_bus.emit(AgentEvent(
                    type="TOPOLOGY",
                    step=0,
                    timestamp=time.time(),
                    meta={
                        "topology_source": topology_result.source,
                        "topology_confidence": topology_result.confidence,
                        "topology_template": topology_result.topology.template_type,
                        "topology_id": topology_result.topology.id,
                        "topology_nodes": topology_result.topology.node_count(),
                    },
                ))
            except (ImportError, RuntimeError) as e:
                _log.warning("Topology generation failed (%s), continuing without", e)
                self.agent_loop._current_topology = None
        else:
            self.agent_loop._current_topology = None

        # Track whether integrated routing handled bandit
        bandit_decision = None

        # Integrated routing: use route_integrated when bandit is wired into router
        if (self.rust_router and hasattr(self.rust_router, 'route_integrated')
                and self.bandit):
            try:
                from sage_core import RoutingConstraints  # noqa: E402
                constraints = RoutingConstraints(
                    max_cost_usd=budget,
                    exploration_budget=EXPLORATION_BUDGET_LOW if system_num <= 2 else EXPLORATION_BUDGET_HIGH,
                )
                topology_id_str = (
                    topology_result.topology.id if topology_result else ""
                )
                integrated_decision = self.rust_router.route_integrated(
                    task, constraints, topology_id_str,
                )
                # Override decision with integrated result
                decision = integrated_decision
                system_num = int(decision.system)
                model_id = decision.model_id
                _log.info(
                    "Integrated routing: S%d, model=%s, topology=%s",
                    system_num, model_id, decision.topology_id,
                )
                bandit_decision = None  # Bandit handled inside route_integrated
                self._last_decision = decision  # Update stored decision
            except (ImportError, RuntimeError) as e:
                _log.debug("Integrated routing failed (%s), using separate paths", e)

        # Phase 6: Bandit model suggestion (Thompson sampling)
        if self.bandit and bandit_decision is None:
            try:
                template_type = (
                    topology_result.topology.template_type
                    if topology_result else "sequential"
                )
                # Seed arms from all registered models in Rust ModelRegistry.
                # NOTE: self.registry is the Python providers.registry.ModelRegistry.
                # The Rust registry is stored as _rust_registry during boot.
                # The Rust method is list_ids() (NOT all_model_ids()).
                if self._rust_registry:
                    for model_id in self._rust_registry.list_ids():
                        self.bandit.register_arm(model_id, template_type)
                else:
                    # Fallback: seed from Python registry's available models
                    for profile in self.registry.list_available():
                        self.bandit.register_arm(profile.id, template_type)
                    if not self.registry.list_available():
                        _log.debug("Bandit: no registry models available, skipping arm seeding")

                bandit_decision = self.bandit.select(EXPLORATION_BUDGET_LOW)
                _log.info(
                    "Bandit suggestion: model=%s, template=%s, quality=%.3f, explore=%s",
                    bandit_decision.model_id, bandit_decision.template,
                    bandit_decision.expected_quality, bandit_decision.exploration,
                )
            except (ImportError, RuntimeError) as e:
                _log.warning("Bandit model selection failed (%s), using default", e)

        # 3. Set validation level from routing decision
        if system_num >= 3:
            self.agent_loop.config.validation_level = 3
        elif system_num == 2 and self.agent_loop.sandbox_manager:
            self.agent_loop.config.validation_level = 2
        else:
            self.agent_loop.config.validation_level = 1

        current_provider = self.agent_loop.config.llm.provider

        # Mock mode: use AgentLoop directly
        if current_provider == "mock":
            result = await self.agent_loop.run(task)
            self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
            return result

        # 4a. Multi-node topology: use AgentLoop → TopologyRunner (direct LLM)
        if self.agent_loop._current_topology:
            _node_count = 0
            try:
                _node_count = self.agent_loop._current_topology.node_count()
            except AttributeError:
                pass
            if _node_count > 1:
                _log.info(
                    "Multi-node topology (%d nodes): using TopologyRunner",
                    _node_count,
                )
                result = await self.agent_loop.run(task)
                await self._persist_memory()
                self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
                return result

        # 5. Fallback: legacy ModelRouter path (only used with Python router)
        if not self.rust_router:
            new_config = ModelRouter.get_config(decision.llm_tier)
            if new_config.provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
                pass  # Google unavailable, keep current
            else:
                self.agent_loop.config.llm = new_config
                if new_config.provider == "google":
                    from sage.llm.google import GoogleProvider
                    self.agent_loop._llm = GoogleProvider()

        result = await self.agent_loop.run(task)
        await self._persist_memory()
        self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
        return result

    def _record_topology_outcome(self, task: str, result: str, topology_result: Any, bandit_decision: Any = None, run_start: float = 0.0) -> None:
        """Record outcome into topology engine's learning loop (S-MMU + MAP-Elites)."""
        if not self.topology_engine or topology_result is None:
            return
        try:
            # Estimate quality from result — returns float or None (abstain)
            from sage.quality_estimator import QualityEstimator  # noqa: E402
            _qe = QualityEstimator()
            quality = _qe.estimate(
                task, result, latency_ms=(time.perf_counter() - run_start) * 1000,
            )
            cost = self.agent_loop.total_cost_usd
            latency_ms = (time.perf_counter() - run_start) * 1000

            # When quality is unknown (None), use 0.5 as neutral recording.
            # The bandit needs observations to learn, even imprecise ones.
            # Only truly empty results (quality=0.0) are definitively bad.
            if quality is None:
                quality = 0.5
                _log.info("Topology outcome: quality=None -> 0.5 (neutral recording)")

            # Extract keywords from task
            keywords = list(set(
                w.lower() for w in re.findall(r'\b\w{4,}\b', task)
            ))[:10]

            topology_id = topology_result.topology.id
            # Compute real embedding for S-MMU record
            outcome_embedding = None
            try:
                from sage.memory.embedder import Embedder
                _emb = Embedder()
                if _emb.is_semantic:
                    outcome_embedding = _emb.embed(task[:500])
            except (ImportError, RuntimeError):
                pass
            self.topology_engine.record_outcome(
                topology_id,
                task[:200],
                keywords,
                outcome_embedding,
                quality,
                cost,
                latency_ms,
            )
            _log.info(
                "Topology outcome recorded: id=%s, quality=%.2f, cost=%.4f, latency=%.0fms",
                topology_id, quality, cost, latency_ms,
            )

            # Update bandit posteriors
            if self.bandit and bandit_decision is not None:
                try:
                    self.bandit.record(
                        bandit_decision.decision_id, quality, cost, latency_ms,
                    )
                except (ImportError, RuntimeError) as e2:
                    _log.warning("Bandit outcome recording failed (%s)", e2)

            # Feed telemetry back to SystemRouter
            if self.rust_router and hasattr(self.rust_router, 'record_outcome'):
                try:
                    _decision = getattr(self, '_last_decision', None)
                    self.rust_router.record_outcome(
                        getattr(_decision, 'decision_id', ''),
                        quality, cost, latency_ms,
                    )
                except (ImportError, RuntimeError) as e3:
                    _log.debug("Router telemetry recording failed (%s)", e3)
        except (ImportError, RuntimeError, ValueError) as e:
            _log.warning("Topology outcome recording failed (%s)", e)

    async def _persist_memory(self) -> None:
        """Persist semantic and causal memory after a run."""
        if hasattr(self.agent_loop, "semantic_memory") and self.agent_loop.semantic_memory:
            try:
                self.agent_loop.semantic_memory.save()
            except (IOError, OSError):
                _log.warning("Failed to persist semantic memory", exc_info=True)
        if hasattr(self.agent_loop, "causal_memory") and self.agent_loop.causal_memory:
            try:
                self.agent_loop.causal_memory.save()
            except (IOError, OSError):
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
    # 1. LLM provider
    provider, llm_config = init_llm_provider(use_mock_llm, llm_tier)

    # 2. Tools, sandbox, kNN router
    tool_registry, sandbox_manager, knn_router = init_tools(event_bus, provider, use_mock_llm)

    # 3. Metacognition (AdaptiveRouter + kNN)
    metacognition = init_metacognition(provider, use_mock_llm, knn_router)

    # 4. Topology engine, bandit, shadow router
    topo = init_topology(rust_registry=None, metacognition=metacognition)
    rust_router = topo["rust_router"]
    rust_registry = topo["rust_registry"]
    py_model_registry = topo["py_model_registry"]
    shadow_router = topo["shadow_router"]
    rust_topology_engine = topo["topology_engine"]
    rust_bandit = topo["bandit"]

    # 5. Evolution + memory agent
    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    agent_pool = AgentPool()

    # Agent config
    config = AgentConfig(
        name=agent_name,
        llm=llm_config,
        system_prompt=(
            "You are YGN-SAGE, a precise AI assistant. "
            "Think step-by-step. Be concise. Answer the user task directly."
        ),
        max_steps=MAX_AGENT_STEPS,
        validation_level=1,  # Default S1 — routing promotes to S2 only for code tasks
    )

    # Event bus (central nervous system)
    event_bus = event_bus or EventBus()

    # 6. Memory tiers
    # We need the agent loop first to wire memory into it, but we need memory_compressor
    # for the loop constructor. So we create a temporary compressor for the loop.
    from sage.memory.compressor import MemoryCompressor
    from sage.memory.embedder import Embedder
    from sage.constants import MEMORY_COMPRESSION_THRESHOLD, MEMORY_KEEP_RECENT
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=MEMORY_COMPRESSION_THRESHOLD,
        keep_recent=MEMORY_KEEP_RECENT,
    )
    memory_compressor.embedder = Embedder()

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
    loop.sandbox_manager = sandbox_manager

    # Wire memory tiers into loop
    mem = init_memory(event_bus, provider, use_mock_llm, loop)
    memory_agent = mem["memory_agent"]
    episodic_memory = mem["episodic_memory"]
    consolidator = mem["consolidator"]
    causal_memory = mem["causal_memory"]

    # ToolExecutor for S2 AVR code validation (Rust tree-sitter + subprocess)
    try:
        from sage_core import ToolExecutor as RustToolExecutor
        tool_executor = RustToolExecutor()
        _log.info("ToolExecutor (Rust): tree-sitter validator + subprocess executor")
    except ImportError:
        tool_executor = None
        _log.info("ToolExecutor (Rust) not available — S2 AVR uses Python sandbox")
    loop.tool_executor = tool_executor
    loop.topology_engine = rust_topology_engine

    # Enable online evolution when Rust TopologyEngine is available
    if rust_topology_engine is not None:
        loop._auto_evolve = True
        _log.info("Online evolution enabled (Rust TopologyEngine available)")

    # CORAL Phase 1: persistent evolution memory (arXiv 2604.01658)
    # Lazy init: EvolutionMemory.initialize() is called on first async use,
    # not here (boot is synchronous, can't safely run async init).
    try:
        from sage.evolution.memory import EvolutionMemory
        _evo_mem = EvolutionMemory()
        # Wire into evolution engine if available
        if hasattr(loop, '_evolution_engine') and loop._evolution_engine:
            loop._evolution_engine._evolution_memory = _evo_mem
        # Wire into LLM mutator for skill injection
        if hasattr(loop, '_mutator') and hasattr(loop._mutator, 'evolution_memory'):
            loop._mutator.evolution_memory = _evo_mem
        loop.evolution_memory = _evo_mem
        _log.info("EvolutionMemory wired (CORAL persistent skills at %s)", _evo_mem._db_path)
    except Exception as exc:
        _log.debug("EvolutionMemory not available: %s", exc)

    # AgeMem: 8 memory tools (3 STM + 4 LTM + 1 Causal)
    from sage.tools.memory_tools import create_memory_tools
    for tool in create_memory_tools(loop.working_memory, episodic_memory, memory_compressor, causal_memory=causal_memory):
        tool_registry.register(tool)

    # ExoCortex tools (search)
    from sage.tools.exocortex_tools import create_exocortex_tools
    for tool in create_exocortex_tools(mem["exocortex"]):
        tool_registry.register(tool)

    # Guardrails
    from sage.guardrails.base import GuardrailPipeline
    from sage.guardrails.builtin import CostGuardrail, OutputGuardrail
    loop.guardrail_pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=COST_GUARDRAIL_MAX_USD),
        OutputGuardrail(min_length=OUTPUT_GUARDRAIL_MIN_LENGTH),
    ])

    # Sandbox availability check — warn loudly if neither Wasm nor Docker present
    check_sandbox_availability()

    # 7. Pipeline, controller, quality, ToolForge
    from sage.providers.registry import ModelRegistry
    registry = ModelRegistry()

    pipe = init_pipeline(
        router=metacognition,
        engine=rust_topology_engine,
        provider=provider,
        llm_config=llm_config,
        bandit=rust_bandit,
        rust_registry=rust_registry,
        py_model_registry=py_model_registry,
        registry=registry,
        event_bus=event_bus,
        use_mock_llm=use_mock_llm,
        consolidator=consolidator,
        working_memory=loop.working_memory,
        episodic_memory=episodic_memory,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
    )

    # Log capability surface at boot (Issue E audit fix)
    from sage.memory.working import get_memory_backend
    _log.info("Memory backend: %s", get_memory_backend())
    _log.info(
        "Capabilities: pipeline=%s, quality=%s, bandit=%s, engine=%s",
        pipe["pipeline"] is not None,
        pipe["quality_estimator"] is not None,
        rust_bandit is not None,
        rust_topology_engine is not None,
    )

    return AgentSystem(
        agent_loop=loop,
        agent_pool=agent_pool,
        metacognition=metacognition,
        topology_evolver=topology_evolver,
        topology_population=topology_population,
        memory_agent=memory_agent,
        tool_registry=tool_registry,
        event_bus=event_bus,
        orchestrator=None,  # Legacy field, kept for backward compat
        registry=pipe["registry"],
        capability_matrix=pipe["capability_matrix"],
        rust_router=rust_router,
        shadow_router=shadow_router,
        topology_engine=rust_topology_engine,
        bandit=rust_bandit,
        _rust_registry=rust_registry or py_model_registry,
        pipeline=pipe["pipeline"],
    )
