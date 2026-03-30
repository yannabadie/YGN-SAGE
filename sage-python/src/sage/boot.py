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
# Legacy fallback (34% GT) — kNN (92%) is primary. Kept for backward compatibility.
from sage.strategy.metacognition import ComplexityRouter  # noqa: E402
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
from sage.constants import (  # noqa: E402
    DEFAULT_BUDGET_USD,
    EXPLORATION_BUDGET_LOW,
    EXPLORATION_BUDGET_HIGH,
    MAX_TOPOLOGY_AGENTS,
    LLM_SYNTHESIS_MIN_SYSTEM,
    MEMORY_COMPRESSION_THRESHOLD,
    MEMORY_KEEP_RECENT,
    COST_GUARDRAIL_MAX_USD,
    OUTPUT_GUARDRAIL_MIN_LENGTH,
    MAX_AGENT_STEPS,
    SPECULATIVE_ZONE_MIN,
    SPECULATIVE_ZONE_MAX,
)

# ModelCard + ModelRegistry — Python implementations (migrated from Rust in Phase 1)
# Rust versions still exist as internal deps of system_router.rs but are no longer
# exported to Python callers.
from sage.llm.model_card import ModelCard, CognitiveSystem  # noqa: E402
from sage.llm.model_registry import ModelCardCatalog as PyModelCardCatalog  # noqa: E402


def _check_sandbox_availability() -> bool:
    """Check if any code execution sandbox is available. Warns if not."""
    has_wasm = False
    has_subprocess = False
    has_docker = False

    try:
        from sage_core import ToolExecutor
        te = ToolExecutor()
        has_wasm = te.has_wasm() or te.has_wasi()
        # tree-sitter + subprocess are always available when ToolExecutor loads
        has_subprocess = True
    except Exception:
        pass

    if not has_subprocess:
        try:
            import shutil
            has_docker = shutil.which("docker") is not None
        except Exception:
            pass

    available = has_wasm or has_subprocess or has_docker
    if not available:
        _log.warning(
            "Code execution unavailable (no sage_core, no Docker). "
            "Tool execution will fail unless allow_local=True."
        )
    elif not has_wasm:
        _log.info(
            "Sandbox: tree-sitter + subprocess (no Wasm component loaded). "
            "Load a .wasm module via ToolExecutor.load_component() for full isolation."
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
                _budget = self._guardrail_budget if hasattr(self, '_guardrail_budget') else DEFAULT_BUDGET_USD / 2
                result = await self.pipeline.run(task, budget_usd=_budget)
                self._last_execution_path = "pipeline"
                _log.info("Execution path: pipeline (5-stage)")
                return result
            except Exception as exc:
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
                except Exception:
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

                bandit_decision = self.bandit.select(EXPLORATION_BUDGET_LOW)
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
            except Exception:
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
            except Exception:
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
            # Route to correct provider based on model_id
            # Rust-first: model_id determines provider, not hardcoded Google
            from sage.providers.openai_compat import OpenAICompatProvider
            model_id = llm_config.model or ""
            if "deepseek" in model_id:
                provider = OpenAICompatProvider(
                    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com/v1",
                    model_id=model_id,
                    provider_name="deepseek",
                )
            elif "gpt-" in model_id or "o1" in model_id:
                provider = OpenAICompatProvider(
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    base_url="https://api.openai.com/v1",
                    model_id=model_id,
                    provider_name="openai",
                )
            elif "grok" in model_id:
                provider = OpenAICompatProvider(
                    api_key=os.environ.get("GROK_API_KEY", ""),
                    base_url="https://api.x.ai/v1",
                    model_id=model_id,
                    provider_name="xai",
                )
            elif "gemini" in model_id:
                from sage.llm.google import GoogleProvider
                provider = GoogleProvider()
            else:
                # Default: Google if available, else DeepSeek
                if os.environ.get("GOOGLE_API_KEY"):
                    from sage.llm.google import GoogleProvider
                    provider = GoogleProvider()
                elif os.environ.get("DEEPSEEK_API_KEY"):
                    provider = OpenAICompatProvider(
                        api_key=os.environ["DEEPSEEK_API_KEY"],
                        base_url="https://api.deepseek.com/v1",
                        model_id="deepseek-chat",
                        provider_name="deepseek",
                    )
                else:
                    raise RuntimeError("No LLM provider available.")

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
        Path.cwd().parent / "sage-core" / "config" / "cards.toml",
        Path(__file__).resolve().parent.parent.parent.parent / "sage-core" / "config" / "cards.toml",
        Path.home() / ".sage" / "cards.toml",
        Path.cwd() / "config" / "cards.toml",
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
            py_model_registry = PyModelCardCatalog.from_toml_file(_cards_toml)
            _log.info(
                "Boot: Python ModelRegistry active (%d models from %s)",
                len(py_model_registry), _cards_toml,
            )
        except Exception as e:
            _log.warning("Boot: Python ModelRegistry init failed (%s)", e)

    # Shadow router: dual Rust/Python comparison when both are available.
    # DEPRECATED: 49.6% divergence — shadow comparison disabled by default.
    # Set SAGE_ENABLE_SHADOW=1 to re-enable for trace collection.
    # When disabled, ShadowRouter acts as a zero-overhead passthrough to
    # whichever router is available (Rust preferred).
    shadow_router = ShadowRouter(
        rust_router=rust_router,
        python_metacognition=metacognition,
    )
    if shadow_router._shadow_active:
        _log.info(
            "Boot: ShadowRouter active (dual Rust/Python comparison, "
            "traces -> %s)", shadow_router._trace_path,
        )
    elif rust_router is not None:
        _log.info(
            "Boot: ShadowRouter shadow comparison disabled (49.6%% divergence). "
            "Rust SystemRouter is primary. Set SAGE_ENABLE_SHADOW=1 to re-enable."
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

    # Phase 6: Rust TopologyEngine (6-path generation + learning loop)
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

    # P1: Restore persisted bandit + MAP-Elites state from previous session
    _sage_state_dir = str(Path.home() / ".sage")
    if rust_topology_engine is not None:
        try:
            if hasattr(rust_topology_engine, 'load_state'):
                bandit_arms, archive_cells = rust_topology_engine.load_state(_sage_state_dir)
                if bandit_arms > 0 or archive_cells > 0:
                    _log.info(
                        "Boot: Restored persisted state — %d bandit arms, %d archive cells from %s",
                        bandit_arms, archive_cells, _sage_state_dir,
                    )
        except Exception as e:
            _log.debug("Boot: No persisted state loaded (%s)", e)

    # P1: Register atexit handler to save bandit + MAP-Elites state at shutdown
    if rust_topology_engine is not None and hasattr(rust_topology_engine, 'save_state'):
        import atexit

        def _save_engine_state(engine=rust_topology_engine, state_dir=_sage_state_dir):
            try:
                engine.save_state(state_dir)
                _log.info("Shutdown: Saved engine state to %s", state_dir)
            except Exception as exc:
                _log.warning("Shutdown: Failed to save engine state (%s)", exc)

        atexit.register(_save_engine_state)
        _log.info("Boot: atexit handler registered for engine state persistence")

    # Bootstrap S-MMU with template topologies on cold start (P5)
    # On first run, S-MMU has 0 chunks → Path 1 always fails. Seed it with
    # templates so that S-MMU retrieval has initial data to work with.
    if rust_topology_engine is not None and rust_topology_engine.smmu_chunk_count() == 0:
        _bootstrap_systems = [1, 2, 3]  # S1=sequential, S2=avr, S3=debate
        _bootstrapped = 0
        for _sys in _bootstrap_systems:
            try:
                _result = rust_topology_engine.generate(
                    f"bootstrap_s{_sys}", None, _sys, 0.0,
                )
                rust_topology_engine.cache_topology(_result.topology)
                rust_topology_engine.record_outcome(
                    _result.topology.id,
                    f"bootstrap_s{_sys}",
                    [f"bootstrap", f"s{_sys}"],
                    None,
                    0.5,  # neutral quality
                    0.0,
                    0.0,
                )
                _bootstrapped += 1
            except Exception:
                pass
        if _bootstrapped > 0:
            _log.info(
                "S-MMU bootstrapped with %d template topologies (%d chunks)",
                _bootstrapped, rust_topology_engine.smmu_chunk_count(),
            )

    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    memory_agent = MemoryAgent(use_llm=not use_mock_llm, llm_provider=provider if not use_mock_llm else None)

    # Memory compressor (fires on pressure — MEM1 pattern)
    memory_compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=MEMORY_COMPRESSION_THRESHOLD,
        keep_recent=MEMORY_KEEP_RECENT,
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
        max_steps=MAX_AGENT_STEPS,
        validation_level=1,  # Default S1 — routing promotes to S2 only for code tasks
    )

    # Event bus (central nervous system)
    event_bus = event_bus or EventBus()

    # ModelRegistry: always created (even in mock mode) so callers can inspect it
    from sage.providers.registry import ModelRegistry
    registry = ModelRegistry()
    _cap_matrix = None
    _runtime_adapters: dict[str, Any] = {}

    if not use_mock_llm:
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
                    pool.submit(lambda: asyncio.run(registry.refresh())).result(timeout=30)
            else:
                # Wrap in timeout to prevent hanging on slow provider discovery
                async def _refresh_with_timeout():
                    try:
                        await asyncio.wait_for(registry.refresh(), timeout=30)
                    except asyncio.TimeoutError:
                        _log.warning("Provider discovery timed out (30s) — using cached/TOML models only")
                asyncio.run(_refresh_with_timeout())
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
            if _pname not in _discovered_providers:
                continue
            _api_key = os.environ.get(_cfg["api_key_env"], "")
            # Fallback for legacy env var spelling
            if not _api_key and _pname == "deepseek":
                _api_key = os.environ.get("DEEP_SEEK_API_KEY", "")
            if not _api_key:
                continue
            if _cfg.get("sdk") == "google-genai":
                from sage.llm.google import GoogleProvider
                _runtime_adapters[_pname] = GoogleProvider(api_key=_api_key)
            else:
                _runtime_adapters[_pname] = OpenAICompatProvider(
                    api_key=_api_key,
                    base_url=_cfg.get("base_url"),
                    provider_name=_pname,
                )
        # Codex CLI provider (uses subprocess, not API key)
        if "codex" in _discovered_providers:
            try:
                from sage.llm.codex import CodexProvider
                _runtime_adapters["codex"] = CodexProvider()
                _log.info("Boot: Codex CLI provider added to runtime adapters")
            except Exception as e:
                _log.warning("Boot: Codex provider init failed (%s)", e)

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

    # Enable online evolution when Rust TopologyEngine is available
    # The engine records outcomes into MAP-Elites + CMA-ME archive,
    # improving future topology selection (SA-3: Online Evolution)
    if rust_topology_engine is not None:
        loop._auto_evolve = True
        _log.info("Online evolution enabled (Rust TopologyEngine available)")

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
        CostGuardrail(max_usd=COST_GUARDRAIL_MAX_USD),
        OutputGuardrail(min_length=OUTPUT_GUARDRAIL_MIN_LENGTH),
    ])

    # Sandbox availability check — warn loudly if neither Wasm nor Docker present
    _check_sandbox_availability()

    # --- Cognitive Orchestration Pipeline (Task 11/12) ---
    # Model assigner: Rust-first with Python fallback
    model_assigner = None
    try:
        from sage_core import ModelAssigner as RustModelAssigner  # type: ignore[import-not-found]
        if rust_registry:
            model_assigner = RustModelAssigner(rust_registry)
    except ImportError:
        pass
    if model_assigner is None:
        try:
            from sage.llm.model_assigner import ModelAssigner as PyModelAssigner
            if py_model_registry:
                model_assigner = PyModelAssigner(py_model_registry)
        except Exception:
            pass

    # Provider pool: wraps default provider + registry for per-node resolution
    _provider_pool = None
    if provider and registry:
        try:
            from sage.llm.provider_pool import ProviderPool
            _provider_pool = ProviderPool(
                default_provider=provider,
                registry=registry,
                default_config=llm_config,
                providers=_runtime_adapters,
            )
            _log.info("ProviderPool: %d live providers — %s", len(_runtime_adapters), list(_runtime_adapters.keys()))
        except Exception as exc:
            _log.warning("ProviderPool init failed: %s", exc)

    # Pipeline: 5-stage orchestration (optional — None if deps missing)
    _pipeline = None
    if model_assigner and _provider_pool:
        try:
            from sage.pipeline import CognitiveOrchestrationPipeline
            _pipeline = CognitiveOrchestrationPipeline(
                router=metacognition,
                engine=rust_topology_engine,
                assigner=model_assigner,
                provider_pool=_provider_pool,
                bandit=rust_bandit,
                quality_estimator=None,  # Populated dynamically if available
                event_bus=event_bus,
                llm_provider=provider,
                llm_config=llm_config,
            )
            _log.info("CognitiveOrchestrationPipeline initialized")
        except Exception as exc:
            _log.warning("Pipeline init failed: %s — using legacy path", exc)

    # TopologyController (Phase C — runtime adaptation)
    _controller = None
    if model_assigner:
        try:
            from sage.topology_controller import TopologyController
            _pv = None
            try:
                from sage.contracts.policy import PolicyVerifier
                _pv = PolicyVerifier
            except ImportError:
                pass
            # QualityEstimator: instantiate for controller quality scoring
            _qe = None
            try:
                from sage.quality_estimator import QualityEstimator
                _qe = QualityEstimator()
            except Exception:
                pass
            # PRM: from agent_loop if available
            _prm = getattr(loop, 'prm', None)
            _controller = TopologyController(
                assigner=model_assigner,
                quality_estimator=_qe,
                prm=_prm,
                policy_verifier=_pv,
                embedder=memory_compressor.embedder,
                event_bus=event_bus,
            )
            _log.info("TopologyController initialized (Phase C)")
        except Exception as exc:
            _log.warning("TopologyController init failed: %s", exc)

    # Pass controller to pipeline
    if _pipeline and _controller:
        _pipeline.controller = _controller

    # Wire QualityEstimator into pipeline Stage 5 LEARN for bandit feedback
    # (ETH-SRI ICLR '25, PILOT 2508.21141: bandit must learn from actual quality)
    _pipeline_qe = locals().get("_qe")  # defined inside TopologyController block
    if not _pipeline_qe:
        try:
            from sage.quality_estimator import QualityEstimator
            _pipeline_qe = QualityEstimator()
        except Exception:
            pass
    if _pipeline and _pipeline_qe:
        _pipeline.quality_estimator = _pipeline_qe

    # Log capability surface at boot (Issue E audit fix)
    from sage.memory.working import get_memory_backend
    _log.info("Memory backend: %s", get_memory_backend())
    _log.info(
        "Capabilities: pipeline=%s, quality=%s, bandit=%s, engine=%s",
        _pipeline is not None,
        _pipeline_qe is not None,
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
        registry=registry,
        capability_matrix=_cap_matrix,
        rust_router=rust_router,
        shadow_router=shadow_router,
        topology_engine=rust_topology_engine,
        bandit=rust_bandit,
        _rust_registry=rust_registry or py_model_registry,
        pipeline=_pipeline,
    )
