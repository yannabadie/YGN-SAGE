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

# Rust Cognitive Engine (primary routing when sage_core is compiled)
try:
    from sage_core import SystemRouter as RustSystemRouter
    from sage_core import ModelRegistry as RustModelRegistry
    from sage_core import TopologyEngine as RustTopologyEngine  # Phase 6
    from sage_core import ContextualBandit as RustBandit  # Phase 6
    _HAS_RUST_ROUTER = True
except ImportError:
    _log.info(
        "sage_core not available — SystemRouter, ModelRegistry, TopologyEngine, "
        "ContextualBandit using Python fallbacks"
    )
    _HAS_RUST_ROUTER = False

from sage.agent import AgentConfig  # noqa: E402
from sage.agent_loop import AgentLoop  # noqa: E402
from sage.agent_pool import AgentPool  # noqa: E402
from sage.llm.base import LLMConfig  # noqa: E402
from sage.llm.mock import MockProvider  # noqa: E402
from sage.llm.router import ModelRouter  # noqa: E402
from sage.strategy.adaptive_router import AdaptiveRouter  # noqa: E402
from sage.strategy.metacognition import ComplexityRouter  # backward compat  # noqa: E402
from sage.topology.evo_topology import TopologyEvolver, TopologyPopulation  # noqa: E402
from sage.memory.memory_agent import MemoryAgent  # noqa: E402
from sage.tools.registry import ToolRegistry  # noqa: E402
from sage.memory.compressor import MemoryCompressor  # noqa: E402
from sage.sandbox.manager import SandboxManager  # noqa: E402
from sage.memory.episodic import EpisodicMemory  # noqa: E402
from sage.memory.remote_rag import ExoCortex  # noqa: E402
from sage.tools.memory_tools import create_memory_tools  # noqa: E402
from sage.events.bus import EventBus  # noqa: E402
from sage.routing.shadow import ShadowRouter  # noqa: E402

# ModelCard + ModelRegistry — Python implementations (migrated from Rust in Phase 1)
# Rust versions still exist as internal deps of system_router.rs but are no longer
# exported to Python callers.
from sage.llm.model_card import ModelCard, CognitiveSystem  # noqa: E402
from sage.llm.model_registry import ModelRegistry as PyModelRegistry  # noqa: E402


def _check_sandbox_availability() -> bool:
    """Check if any code execution sandbox is available. Warns if not."""
    has_wasm = False
    has_docker = False

    try:
        from sage_core import ToolExecutor
        te = ToolExecutor()
        has_wasm = te.has_wasm() or te.has_wasi()
    except Exception:
        pass

    if not has_wasm:
        try:
            import shutil
            has_docker = shutil.which("docker") is not None
        except Exception:
            pass

    available = has_wasm or has_docker
    if not available:
        _log.warning(
            "Code execution sandbox unavailable (no Wasm, no Docker). "
            "Tool execution will fail unless allow_local=True."
        )
    return available


@dataclass
class AgentSystem:
    """The complete YGN-SAGE agent system."""
    agent_loop: AgentLoop
    agent_pool: AgentPool
    metacognition: AdaptiveRouter | ComplexityRouter
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
    # CapabilityMatrix: semantic capability lookup for discovered providers
    capability_matrix: Any = None
    # Rust SystemRouter (None if sage_core not compiled with cognitive engine)
    rust_router: Any = None
    # ShadowRouter: dual Rust/Python routing comparison (None if shadow mode inactive)
    shadow_router: ShadowRouter | None = None
    # Phase 6: Rust TopologyEngine (5-path generate, MAP-Elites, bandit)
    topology_engine: Any = None
    # Phase 6: Standalone ContextualBandit for model selection
    bandit: Any = None
    # Rust ModelRegistry (None if sage_core not compiled or cards.toml not found)
    _rust_registry: Any = None

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

        Primary path: CognitiveOrchestrator (multi-provider, score-based).
        Fallback: legacy AgentLoop with ModelRouter (Codex + Google only).
        Mock mode: direct AgentLoop (no orchestrator).
        """
        _run_start = time.perf_counter()
        self._last_decision = None  # Routing decision for telemetry feedback

        # 1. Route task to cognitive system
        budget = 10.0  # Default budget (USD)
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
                if 0.35 <= profile.complexity <= 0.55 and decision.system <= 2:
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
            if 0.35 <= profile.complexity <= 0.55 and decision.system <= 2:
                _log.info(
                    "Speculative zone: complexity=%.2f (indecisive). Using S%d for now.",
                    profile.complexity, decision.system,
                )

        self._last_decision = decision  # Store for telemetry in _record_topology_outcome

        # 2. Topology generation (Rust engine, 5-path strategy)
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
                exploration_budget = 0.3 if system_num <= 2 else 0.5
                topology_result = self.topology_engine.generate(
                    task,
                    None,  # embedding (populated after first run)
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
                        and system_num >= 2
                        and self.agent_loop.config.llm.provider != "mock"):
                    try:
                        from sage.topology.llm_caller import synthesize_topology
                        llm_graph = await synthesize_topology(
                            self.agent_loop._llm,
                            task,
                            max_agents=4,
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
                    except Exception as e:
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
            except Exception as e:
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
                    exploration_budget=0.3 if system_num <= 2 else 0.5,
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
            except Exception as e:
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

                bandit_decision = self.bandit.select(0.3)
                _log.info(
                    "Bandit suggestion: model=%s, template=%s, quality=%.3f, explore=%s",
                    bandit_decision.model_id, bandit_decision.template,
                    bandit_decision.expected_quality, bandit_decision.exploration,
                )
            except Exception as e:
                _log.warning("Bandit model selection failed (%s), using default", e)

        # 3. Set validation level from routing decision
        if system_num >= 3:
            self.agent_loop.config.validation_level = 3
        elif system_num == 2 and self.agent_loop.sandbox_manager:
            self.agent_loop.config.validation_level = 2
        else:
            self.agent_loop.config.validation_level = 1

        current_provider = self.agent_loop.config.llm.provider

        # Mock mode: skip orchestrator, use AgentLoop directly
        if current_provider == "mock":
            result = await self.agent_loop.run(task)
            self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
            return result

        # 4a. Multi-node topology: use AgentLoop → TopologyRunner (direct LLM)
        #     The CognitiveOrchestrator ignores topologies and creates its own
        #     ModelAgents with quality cascades. When a multi-node topology is
        #     active, we MUST bypass the orchestrator so TopologyRunner executes
        #     the actual topology graph with per-node LLM calls.
        if self.agent_loop._current_topology:
            _node_count = 0
            try:
                _node_count = self.agent_loop._current_topology.node_count()
            except Exception:
                pass
            if _node_count > 1:
                _log.info(
                    "Multi-node topology (%d nodes): bypassing orchestrator -> TopologyRunner",
                    _node_count,
                )
                result = await self.agent_loop.run(task)
                await self._persist_memory()
                self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
                return result

        # 4b. Try CognitiveOrchestrator as primary path (multi-provider)
        if self.orchestrator and self.registry and self.registry.list_available():
            try:
                # Construct authoritative ExecutionDecision from routing result
                from sage.execution_decision import ExecutionDecision
                exec_decision = ExecutionDecision(
                    system=getattr(self._last_decision, "system", 2),
                    model_id=getattr(self._last_decision, "model_id", "unknown") or "unknown",
                    topology_id=getattr(self._last_decision, "topology_id", None),
                )
                result = await self.orchestrator.run(task, decision=exec_decision)
                await self._persist_memory()
                self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
                return result
            except Exception as e:
                _log.warning(
                    "Orchestrator failed (%s), falling back to legacy routing", e
                )

        # 5. Fallback: legacy ModelRouter path (only used with Python router)
        if not self.rust_router:
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
        self._record_topology_outcome(task, result, topology_result, bandit_decision, _run_start)
        return result

    def _record_topology_outcome(self, task: str, result: str, topology_result: Any, bandit_decision: Any = None, run_start: float = 0.0) -> None:
        """Record outcome into topology engine's learning loop (S-MMU + MAP-Elites)."""
        if not self.topology_engine or topology_result is None:
            return
        try:
            # Estimate quality from result (multi-signal, 0.0-1.0)
            from sage.quality_estimator import QualityEstimator  # noqa: E402
            quality = QualityEstimator.estimate(
                task, result, latency_ms=(time.perf_counter() - run_start) * 1000,
                had_errors=bool(getattr(self.agent_loop, '_last_error', None)),
                avr_iterations=getattr(self.agent_loop, '_last_avr_iterations', 0),
            )
            cost = self.agent_loop.total_cost_usd
            latency_ms = (time.perf_counter() - run_start) * 1000

            # Extract keywords from task
            keywords = list(set(
                w.lower() for w in re.findall(r'\b\w{4,}\b', task)
            ))[:10]

            topology_id = topology_result.topology.id
            self.topology_engine.record_outcome(
                topology_id,
                task[:200],
                keywords,
                None,  # embedding (future: compute at learn time)
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
                except Exception as e2:
                    _log.warning("Bandit outcome recording failed (%s)", e2)

            # Feed telemetry back to SystemRouter
            if self.rust_router and hasattr(self.rust_router, 'record_outcome'):
                try:
                    _decision = getattr(self, '_last_decision', None)
                    self.rust_router.record_outcome(
                        getattr(_decision, 'decision_id', ''),
                        quality, cost, latency_ms,
                    )
                except Exception as e3:
                    _log.debug("Router telemetry recording failed (%s)", e3)
        except Exception as e:
            _log.warning("Topology outcome recording failed (%s)", e)

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

    # Stage 0.5: kNN router (arXiv 2505.12601 — kNN on embeddings beats complex routers)
    _knn_router = None
    try:
        from sage.strategy.knn_router import KnnRouter
        _knn_router = KnnRouter()
        if not _knn_router.is_ready:
            # Try building from ground truth on-the-fly
            if _knn_router.build_from_ground_truth():
                _log.info(
                    "Boot: kNN router built from ground truth (%d exemplars, %s)",
                    _knn_router.exemplar_count, _knn_router.embedder_backend,
                )
            else:
                _knn_router = None
        else:
            _log.info(
                "Boot: kNN router loaded (%d exemplars, %s)",
                _knn_router.exemplar_count, _knn_router.embedder_backend,
            )
    except Exception as e:
        _log.info("Boot: kNN router unavailable (%s)", e)

    metacognition = AdaptiveRouter(
        llm_provider=provider if not use_mock_llm else None,
        knn_router=_knn_router,
    )

    # Load kNN exemplars into Rust AdaptiveRouter for native SIMD kNN search
    if _knn_router is not None and _knn_router.is_ready and metacognition.has_rust:
        try:
            import numpy as np
            emb = _knn_router._exemplar_embeddings
            labels = _knn_router._exemplar_labels
            if emb is not None and labels is not None:
                flat_emb = emb.flatten().tolist()
                flat_labels = labels.astype(np.uint8).tolist()
                n = metacognition._rust.load_exemplars(flat_emb, flat_labels)
                if n > 0:
                    _log.info("Boot: Rust kNN loaded %d exemplars (native SIMD search)", n)
        except Exception as e:
            _log.info("Boot: Rust kNN exemplar load failed (%s), using Python kNN", e)

    # Rust SystemRouter (primary path when sage_core cognitive engine is compiled)
    rust_router = None
    rust_registry = None  # Hoisted for Phase 6 bandit warm-start
    py_model_registry = None  # Python fallback for rationalization

    # Search for cards.toml in standard locations
    _cards_toml = None
    for _cards_dir in [
        Path.cwd() / "sage-core" / "config" / "cards.toml",
        Path(__file__).parent.parent.parent.parent.parent / "sage-core" / "config" / "cards.toml",
        Path.home() / ".sage" / "cards.toml",
    ]:
        if _cards_dir.exists():
            _cards_toml = str(_cards_dir)
            break

    if _HAS_RUST_ROUTER:
        try:
            if _cards_toml:
                rust_registry = RustModelRegistry.from_toml_file(_cards_toml)
                rust_router = RustSystemRouter(rust_registry)
                _log.info(
                    "Boot: Rust SystemRouter active (%d models from %s)",
                    len(rust_registry), _cards_toml,
                )
            else:
                _log.info("Boot: cards.toml not found, using Python AdaptiveRouter")
        except Exception as e:
            _log.warning(
                "Boot: Rust SystemRouter init failed (%s), using Python AdaptiveRouter", e,
            )

    # Python ModelRegistry fallback — used when Rust is unavailable
    if rust_registry is None and _cards_toml:
        try:
            py_model_registry = PyModelRegistry.from_toml_file(_cards_toml)
            _log.info(
                "Boot: Python ModelRegistry active (%d models from %s)",
                len(py_model_registry), _cards_toml,
            )
        except Exception as e:
            _log.warning("Boot: Python ModelRegistry init failed (%s)", e)

    # Shadow router: dual Rust/Python comparison when both are available.
    # Always created — handles all combinations internally:
    # both routers (shadow comparison), rust-only, python-only.
    # Zero-overhead when only one router is present.
    shadow_router = ShadowRouter(
        rust_router=rust_router,
        python_metacognition=metacognition,
    )
    if rust_router is not None:
        _log.info(
            "Boot: ShadowRouter active (dual Rust/Python comparison, "
            "traces -> %s)", shadow_router._trace_path,
        )

    # Phase 5 gate: load existing traces for cross-session continuity
    if shadow_router._shadow_active:
        shadow_router.load_existing_traces()
        if shadow_router.is_phase5_hard_ready():
            _log.info(
                "Shadow Phase 5 HARD gate passed (%d traces, %.1f%% divergence) — "
                "Python router can be safely removed",
                shadow_router.stats.get("total_comparisons", 0),
                shadow_router.divergence_rate() * 100,
            )
        elif shadow_router.is_phase5_soft_ready():
            _log.info(
                "Shadow Phase 5 SOFT gate passed (%d traces, %.1f%% divergence) — "
                "Rust router preferred",
                shadow_router.stats.get("total_comparisons", 0),
                shadow_router.divergence_rate() * 100,
            )
        else:
            _log.info(
                "Shadow Phase 5: %d/1000 traces collected (divergence=%.1f%%)",
                shadow_router.stats.get("total_comparisons", 0),
                shadow_router.divergence_rate() * 100,
            )

    # Phase 2: Topology templates + HybridVerifier are internal to
    # DynamicTopologyEngine (Rust). No separate Python instantiation needed.
    # (Removed: template_store + verifier were instantiated but never used — audit P10)

    # Phase 6: Rust TopologyEngine (5-path generation + learning loop)
    rust_topology_engine = None
    rust_bandit = None
    if _HAS_RUST_ROUTER:
        try:
            rust_topology_engine = RustTopologyEngine()
            rust_bandit = RustBandit(0.995, 0.1)
            if rust_router and rust_bandit:
                try:
                    rust_router.set_bandit(rust_bandit)
                    _log.info("Boot: Bandit wired into SystemRouter for integrated routing")
                except Exception as e:
                    _log.debug("Boot: Failed to wire bandit into router (%s)", e)
            # Warm-start bandit arms from ModelCard affinities
            if rust_registry and rust_bandit:
                try:
                    cards = rust_registry.all_models()
                    templates = ["sequential", "avr", "parallel", "debate"]
                    model_ids = [c.id for c in cards]
                    # Build affinities in row-major: [model0_tmpl0, model0_tmpl1, ..., modelN_tmplT]
                    affinities: list[float] = []
                    for c in cards:
                        for t in templates:
                            if t in ("sequential", "avr"):
                                affinities.append(c.s2_affinity)
                            elif t in ("parallel", "debate"):
                                affinities.append(c.s3_affinity)
                            else:
                                affinities.append(max(c.s1_affinity, c.s2_affinity, c.s3_affinity))
                    rust_bandit.warm_start_from_affinities(model_ids, templates, affinities)
                    _log.info(
                        "Boot: Bandit warm-started with %d models x %d templates (%d arms)",
                        len(model_ids), len(templates), len(model_ids) * len(templates),
                    )
                except Exception as e:
                    _log.debug("Boot: Bandit warm-start failed (%s)", e)
            _log.info(
                "Boot: Phase 6 active — TopologyEngine + ContextualBandit ready"
            )
        except Exception as e:
            _log.warning("Boot: Phase 6 TopologyEngine init failed (%s)", e)

    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    memory_agent = MemoryAgent(use_llm=not use_mock_llm, llm_provider=provider if not use_mock_llm else None)

    # Memory compressor (fires on pressure — MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=20,
        keep_recent=5,
    )

    # Embedder for S-MMU semantic edges
    from sage.memory.embedder import Embedder
    memory_compressor.embedder = Embedder()

    # Runtime tool synthesis — sandboxed (SEC-01/SEC-02 fixed).
    # Tools execute in subprocess isolation, not in-process exec().
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
    _cap_matrix = None

    if not use_mock_llm:
        from sage.orchestrator import CognitiveOrchestrator

        # Auto-discover available models at boot.
        # NOTE: The ThreadPoolExecutor pattern below is intentional and safe.
        # registry.refresh() only performs HTTP health-check calls (no shared
        # state mutation).  When a running event loop already exists (e.g. in
        # Jupyter or async test harnesses), we cannot call asyncio.run() on
        # the same thread, so we delegate to a separate thread with its own
        # event loop.  This avoids "cannot run nested event loop" errors.
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
            # Log per-provider summary
            from collections import Counter
            available = registry.list_available()
            provider_counts = Counter(p.provider for p in available)
            total = len(registry.profiles)
            avail = len(available)
            summary_parts = [f"{name}: {count}" for name, count in sorted(provider_counts.items())]
            _log.info(
                "Boot: discovered %d models (%d available) — %s",
                total, avail, ", ".join(summary_parts) if summary_parts else "none",
            )
        except Exception as e:
            _log.warning("Boot: model discovery failed (%s), continuing with legacy routing", e)

        orchestrator = CognitiveOrchestrator(
            registry=registry, metacognition=metacognition, event_bus=event_bus,
        )

        # Auto-populate capability matrix from discovered providers.
        # Build runtime adapter instances so CapabilityMatrix trusts their
        # capabilities() report over static _KNOWN_CAPABILITIES claims.
        from sage.providers.capabilities import CapabilityMatrix as _CapMatrix
        from sage.providers.connector import PROVIDER_CONFIGS
        from sage.providers.openai_compat import OpenAICompatProvider
        _cap_matrix = _CapMatrix()
        _discovered_providers = {p.provider for p in registry.list_available()}
        _runtime_adapters: dict[str, Any] = {}
        for _cfg in PROVIDER_CONFIGS:
            _pname = _cfg["provider"]
            if _pname in _discovered_providers and _cfg.get("sdk") == "openai":
                _api_key = os.environ.get(_cfg["api_key_env"], "")
                if _api_key:
                    _runtime_adapters[_pname] = OpenAICompatProvider(
                        api_key=_api_key,
                        base_url=_cfg.get("base_url"),
                        provider_name=_pname,
                    )
        _cap_matrix.populate_from_providers(
            list(_discovered_providers), adapters=_runtime_adapters,
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

    # Semantic memory + MemoryAgent wiring (persistent SQLite in real mode)
    from sage.memory.semantic import SemanticMemory
    if not use_mock_llm:
        _sem_db = Path.home() / ".sage" / "semantic.db"
        _sem_db.parent.mkdir(parents=True, exist_ok=True)
        semantic_memory = SemanticMemory(db_path=str(_sem_db))
        semantic_memory.load()
    else:
        semantic_memory = SemanticMemory()
    loop.memory_agent = memory_agent
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

    # Sandbox availability check — warn loudly if neither Wasm nor Docker present
    _check_sandbox_availability()

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
        capability_matrix=_cap_matrix,
        rust_router=rust_router,
        shadow_router=shadow_router,
        topology_engine=rust_topology_engine,
        bandit=rust_bandit,
        _rust_registry=rust_registry or py_model_registry,
    )
