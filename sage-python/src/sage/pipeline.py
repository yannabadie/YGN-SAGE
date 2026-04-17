"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from sage.pipeline_stages import (
    _infer_domain,
    compute_dag_features,
    select_macro_topology,
    DAGFeatures,
)

# OxiZ formal verification — imported lazily to allow graceful fallback
try:
    from sage.contracts.z3_verify import verify_provider_assignment, ProviderSpec
    _Z3_VERIFY_AVAILABLE = True
except ImportError:
    _Z3_VERIFY_AVAILABLE = False
    verify_provider_assignment = None  # type: ignore[assignment]
    ProviderSpec = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """State that flows through the 5 pipeline stages."""

    task: str
    budget: float = 5.0
    domain: str = ""
    system: int = 0
    task_dag: Any = None
    dag_features: DAGFeatures | None = None
    topology: Any = None
    topology_id: str = ""
    assignments: dict[int, str] = field(default_factory=dict)
    provider_hints: dict[int, str] = field(default_factory=dict)  # node_idx -> provider_name
    result: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0
    bandit_decision_id: str | None = None
    verification_passed: bool = True
    axis_hint: str = ""  # MASBENCH axis hint for topology selection
    tool_call_count: int = 0
    tool_turn_count: int = 0
    executed_commands: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)


class CognitiveOrchestrationPipeline:
    """5-stage pipeline: Classify -> Decompose -> Select Topology -> Assign Models -> Execute.

    Parameters
    ----------
    router : AdaptiveRouter
        For Stage 0 (classify).
    engine : TopologyEngine (Rust) or None
        For Stage 2 (select topology). If None, uses sequential template.
    assigner : ModelAssigner (Rust or Python)
        For Stage 3 (assign models per node).
    provider_pool : ProviderPool
        For Stage 4 (resolve model_id -> provider at execution).
    bandit : ContextualBandit or None
        For Stage 5 (learn from outcome).
    quality_estimator : QualityEstimator or None
        For Stage 5 (quality scoring).
    event_bus : EventBus or None
        For observability (emit events at each stage transition).
    llm_provider : LLMProvider
        Default provider for AgentLoop / TopologyRunner.
    llm_config : LLMConfig or None
        Default config.
    """

    def __init__(
        self,
        router: Any,
        engine: Any,
        assigner: Any,
        provider_pool: Any,
        bandit: Any = None,
        quality_estimator: Any = None,
        event_bus: Any = None,
        llm_provider: Any = None,
        llm_config: Any = None,
        prm: Any = None,
        controller: Any = None,
        smmu: Any = None,
        consolidator: Any = None,
        working_memory: Any = None,
        episodic_memory: Any = None,
        tool_forge: Any = None,
        tool_registry: Any = None,
        harness_config: Any = None,
        agent_loop: Any = None,
    ) -> None:
        self.router = router
        self.engine = engine
        self.assigner = assigner
        self.provider_pool = provider_pool
        self.bandit = bandit
        self.quality_estimator = quality_estimator
        self.event_bus = event_bus
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self.prm = prm
        self.controller = controller
        self.tool_registry = tool_registry
        self._rust_registry = None  # Set by boot if Rust ModelRegistry available
        self._rust_router = None    # Set by boot if Rust SystemRouter available
        self._smmu = smmu
        self.consolidator = consolidator
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.tool_forge = tool_forge
        self.harness_config = harness_config  # Meta-Harness: loaded from config/harness.json at boot
        self._harness_patcher = None
        if harness_config:
            try:
                from sage.meta_harness.patcher import HarnessPatcher
                self._harness_patcher = HarnessPatcher(harness_config)
                log.info("Meta-Harness config '%s' loaded: %s",
                         harness_config.id, harness_config.description)
            except ImportError:
                log.debug("meta_harness module not available, skipping harness config")
        self._agent_loop = agent_loop
        self._task_count = 0

    def _record_to_memory(self, ctx: PipelineContext) -> None:
        """Write execution trace to Tier 0 (working memory) and Tier 1 (episodic).

        This closes the gap where the pipeline bypassed memory entirely.
        The consolidator (Stage 5) then migrates episodic→semantic→causal.
        """
        # Tier 0: Working memory S-MMU events
        if self.working_memory:
            try:
                self.working_memory.add_event("TASK", ctx.task[:500])
                self.working_memory.add_event(
                    "TOPOLOGY",
                    f"system=S{ctx.system}, nodes={ctx.topology.node_count() if ctx.topology and hasattr(ctx.topology, 'node_count') else 0}, "
                    f"assignments={ctx.assignments}",
                )
                self.working_memory.add_event(
                    "RESULT",
                    ctx.result[:500] if ctx.result else "empty",
                )
                self.working_memory.add_event(
                    "METRICS",
                    f"latency={ctx.latency_ms:.0f}ms, cost={ctx.cost:.4f}, path={getattr(self, '_last_path', 'pipeline')}",
                )
                # Compact to Arrow chunk for S-MMU graph storage
                if self.working_memory.event_count() >= 4:
                    self.working_memory.compact_to_arrow()
            except (RuntimeError, IOError) as exc:
                log.debug("Memory write (Tier 0) failed: %s", exc)

        # Tier 1: Episodic memory (persistent SQLite)
        if self.episodic_memory:
            try:
                import asyncio
                entry = {
                    "task": ctx.task[:200],
                    "system": ctx.system,
                    "topology_nodes": ctx.topology.node_count() if ctx.topology and hasattr(ctx.topology, 'node_count') else 0,
                    "assignments": str(ctx.assignments),
                    "result_len": len(ctx.result) if ctx.result else 0,
                    "latency_ms": ctx.latency_ms,
                    "cost": ctx.cost,
                }
                import json
                content = json.dumps(entry, default=str)
                # Use sync add if available, else async
                if hasattr(self.episodic_memory, 'add'):
                    self.episodic_memory.add(key=f"pipeline-{self._task_count}", content=content)
                elif hasattr(self.episodic_memory, 'add_episode'):
                    self.episodic_memory.add_episode(key=f"pipeline-{self._task_count}", content=content)
            except (RuntimeError, IOError) as exc:
                log.debug("Memory write (Tier 1) failed: %s", exc)

    def _emit(self, stage: str, data: dict) -> None:  # type: ignore[type-arg]
        """Emit a PIPELINE event on EventBus if available."""
        if self.event_bus and hasattr(self.event_bus, "emit"):
            try:
                from sage.agent_loop import AgentEvent

                self.event_bus.emit(
                    AgentEvent(
                        type="PIPELINE",
                        step=0,
                        timestamp=time.time(),
                        meta={"stage": stage, **data},
                    )
                )
            except (ImportError, RuntimeError):
                pass

    async def run(
        self,
        task: str,
        budget_usd: float = 10.0,
        system_hint: int | None = None,
    ) -> str:
        """Execute the full 5-stage pipeline.

        Args:
            task: The user's task.
            budget_usd: Soft budget cap for the run.
            system_hint: Optional override for Stage 0 routing (1, 2, or 3).
                Benchmark adapters use this when they already know the task
                complexity (e.g. SWE-bench tasks are always S3). When set,
                the Rust SystemRouter still runs (so we keep the model
                assignment + bandit posteriors), but `ctx.system` is forced
                to the hint afterwards.
        """
        t0 = time.monotonic()
        ctx = PipelineContext(task=task, budget=budget_usd)

        # Stage 0: CLASSIFY
        ctx = self._stage_classify(ctx)
        if system_hint in (1, 2, 3) and ctx.system != system_hint:
            log.info(
                "Stage 0: system_hint=S%d overrides router S%d",
                system_hint, ctx.system,
            )
            ctx.system = system_hint
        self._emit("CLASSIFY", {"system": ctx.system, "domain": ctx.domain})

        # Stage 1: DECOMPOSE (S2/S3 only)
        ctx = await self._stage_decompose(ctx)
        dag_node_count = 0
        if ctx.task_dag is not None:
            if hasattr(ctx.task_dag, "node_count"):
                dag_node_count = ctx.task_dag.node_count
            elif hasattr(ctx.task_dag, "node_ids"):
                dag_node_count = len(list(ctx.task_dag.node_ids))
        self._emit(
            "DECOMPOSE",
            {
                "dag_nodes": dag_node_count,
                "features": (
                    {
                        "omega": ctx.dag_features.omega,
                        "delta": ctx.dag_features.delta,
                        "gamma": ctx.dag_features.gamma,
                    }
                    if ctx.dag_features
                    else {}
                ),
            },
        )

        # Stage 2: SELECT TOPOLOGY
        ctx = self._stage_select_topology(ctx)
        topo_nodes = (
            ctx.topology.node_count()
            if ctx.topology and hasattr(ctx.topology, "node_count")
            else 0
        )
        self._emit("SELECT_TOPOLOGY", {"node_count": topo_nodes})

        # Stage 3: ASSIGN MODELS
        ctx = self._stage_assign_models(ctx)
        self._emit(
            "ASSIGN_MODELS", {"assignments": ctx.assignments, "domain": ctx.domain}
        )

        # Stage 4: EXECUTE
        ctx = await self._stage_execute(ctx)
        ctx.latency_ms = (time.monotonic() - t0) * 1000

        # Write execution trace to memory (Tier 0 + Tier 1)
        self._record_to_memory(ctx)

        # Stage 5: LEARN
        await self._stage_learn(ctx)
        self._emit("LEARN", {"latency_ms": ctx.latency_ms})

        # Expose full context for observability (bench, tracing, debugging)
        self.last_context = ctx

        return ctx.result

    # ── Stage 0: Classify ───────────────────────────────────────────────────

    def _stage_classify(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 0: Classify task complexity, domain, and select model.

        Priority: Rust SystemRouter (kNN + bandit + domain + constraints)
                > Python kNN fallback > heuristic (34%).

        The Rust SystemRouter combines:
        - Structural feature extraction (complexity, uncertainty, tool_required)
        - kNN 93.3% GT accuracy
        - Thompson sampling bandit (per-model posteriors)
        - Domain scoring from cards.toml
        - Budget-constrained model selection
        - Formal keyword detection (prove, theorem → S3)

        Returns: system (S1/S2/S3), domain, and stores RoutingDecision for
        model selection in Stage 3 (ModelAssigner) and Stage 5 (Learn).
        """
        # Priority 1: Rust SystemRouter (full integrated routing)
        if self._rust_router:
            try:
                decision = self._rust_router.route(ctx.task, ctx.budget)
                ctx.system = int(decision.system)
                ctx.domain = _infer_domain(ctx.task)
                # Store decision for model selection + telemetry
                self._last_routing_decision = decision
                log.info(
                    "Stage 0: Rust routing → S%d model=%s (conf=%.2f, cost=%.4f)",
                    ctx.system, decision.model_id, decision.confidence, decision.estimated_cost,
                )
                return ctx
            except Exception as exc:
                log.warning("Stage 0: Rust SystemRouter failed (%s), falling back to Python", exc)

        # Priority 2: Python kNN (93.3% accuracy, Rust-accelerated embedding)
        if self.router and hasattr(self.router, '_knn') and self.router._knn is not None:
            try:
                knn_result = self.router._knn.route(ctx.task)
                if knn_result is not None:
                    ctx.system = knn_result.system
                    log.info("Stage 0: kNN routing → S%d (conf=%.2f, %s)",
                             knn_result.system, knn_result.confidence, knn_result.method)
                    ctx.domain = _infer_domain(ctx.task)
                    return ctx
            except (ImportError, RuntimeError) as exc:
                log.debug("Stage 0: kNN failed (%s), falling back", exc)

        # Priority 3: AdaptiveRouter heuristic
        if self.router:
            try:
                profile = self.router.assess_complexity(ctx.task)
                decision = self.router.route(profile)
                ctx.system = getattr(decision, "system", 2)
            except (ImportError, RuntimeError) as exc:
                log.warning("Stage 0 classify failed: %s, defaulting to S2", exc)
                ctx.system = 2
        else:
            ctx.system = 2

        ctx.domain = _infer_domain(ctx.task)
        return ctx

    # ── Stage 1: Decompose ──────────────────────────────────────────────────

    async def _stage_decompose(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Decompose task into sub-tasks (S2/S3 only)."""
        if ctx.system == 1:
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)
            return ctx

        # Try LLM decomposition via TaskPlanner if available
        try:
            from sage.contracts.planner import TaskPlanner

            planner = TaskPlanner()
            if self.llm_provider and hasattr(planner, "plan_auto"):
                result = await planner.plan_auto(ctx.task, self.llm_provider)
                ctx.task_dag = result.dag
                ctx.dag_features = compute_dag_features(result.dag)
            else:
                ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)
        except (RuntimeError, TimeoutError) as exc:
            log.warning("Stage 1 decompose failed: %s, using single-node DAG", exc)
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)

        return ctx

    # ── Structure-driven topology selection ──────────────────────────────

    def _build_topology_from_hint(self, hint: str) -> Any | None:
        """Create a topology from a template hint using Rust TemplateStore.

        No hardcoded prompts — nodes use their role-based defaults.
        The runner builds system prompts from each node's role field.
        """
        try:
            from sage_core import PyTemplateStore  # type: ignore[import-not-found]
            store = PyTemplateStore()
            return store.create(hint, "")
        except (ImportError, ValueError):
            return None

    # ── Stage 2: Select Topology ────────────────────────────────────────────

    def _stage_select_topology(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Select optimal topology.

        S1 (simple) tasks skip topology entirely — direct single-agent call
        is faster AND equally effective (confirmed by MASBENCH: topology helps
        only when base accuracy < 60%, per AdaptOrch arXiv 2602.16873).
        """
        # Sprint 5 ablation: force bypass to measure framework delta.
        if os.environ.get("SAGE_ABLATION_NO_TOPOLOGY") == "1":
            ctx.topology = None
            log.info("Stage 2: topology disabled by SAGE_ABLATION_NO_TOPOLOGY=1 (ablation)")
            return ctx

        # S1 fast path: skip topology for non-math tasks.
        # Math tasks use formal_solver (SatLM NeurIPS 2023): LLM formalizes,
        # Rust solves exactly. Falls back to single-agent if solver fails.
        if ctx.system == 1:
            if ctx.domain == "math":
                topo = self._build_topology_from_hint("formal_solver")
                if topo:
                    ctx.topology = topo
                    log.info("S1 math: formal_solver (formalizer → Rust solver, fallback to CoT)")
                    self._check_topology_budget(ctx)
                    return ctx
            ctx.topology = None
            log.debug("S1 task: skipping topology (direct single-agent)")
            return ctx

        # Structure-driven template selection from DAG decomposition
        # Uses omega (parallelism), delta (depth), gamma (coupling) —
        # no regex heuristics, purely structural signals from Stage 1.
        hint = "sequential"
        if ctx.dag_features:
            hint = select_macro_topology(ctx.dag_features, ctx.system, ctx.domain)

        # S2+sequential: use the sequential topology template instead of bypass.
        # Research (AdaptOrch 2602.16873, MASS 2502.02533) shows sequential
        # planner→coder→synthesizer pipeline beats single-agent by 12-23%.
        # The old bypass (SAGE_BYPASS_S2_SEQUENTIAL=1) is available for A/B testing.
        if ctx.system == 2 and hint == "sequential":
            # `os` is imported at module level; a redundant local `import os`
            # here would shadow it and break the earlier SAGE_ABLATION_NO_TOPOLOGY
            # check (UnboundLocalError when Python treats `os` as local).
            if os.environ.get("SAGE_BYPASS_S2_SEQUENTIAL") == "1":
                ctx.topology = None
                log.info("Stage 2: BYPASS topology (SAGE_BYPASS_S2_SEQUENTIAL=1)")
                return ctx

        # Build topology from template hint. All DAG-selected templates go
        # through TemplateStore which creates multi-node topologies.
        if hint in ("sequential", "avr", "parallel", "robust", "horizon_pipeline", "parallel_fanout"):
            topo = self._build_topology_from_hint(hint)
            if topo:
                ctx.topology = topo
                log.info(
                    "Stage 2: DAG-driven template=%s (%d nodes, omega=%s delta=%s gamma=%s)",
                    hint, topo.node_count(),
                    ctx.dag_features.omega if ctx.dag_features else "?",
                    ctx.dag_features.delta if ctx.dag_features else "?",
                    f"{ctx.dag_features.gamma:.2f}" if ctx.dag_features else "?",
                )
                self._check_topology_budget(ctx)
                return ctx

        # Try DynamicTopologyEngine
        if self.engine:
            try:
                # Compute real embedding for S-MMU semantic retrieval
                task_embedding = None
                try:
                    from sage.memory.embedder import Embedder
                    _emb = Embedder()
                    if _emb.is_semantic:
                        task_embedding = _emb.embed(ctx.task[:500])
                except (ImportError, RuntimeError, OSError):
                    # OSError: model weights not on disk / HF_HUB_OFFLINE=1 without cache
                    pass
                result = self.engine.generate(ctx.task, task_embedding, ctx.system, ctx.budget)
                if result and hasattr(result, "topology"):
                    ctx.topology = result.topology
                    if hasattr(result, "topology_id"):
                        ctx.topology_id = result.topology_id()
                    elif hasattr(ctx.topology, "id"):
                        ctx.topology_id = ctx.topology.id
                elif result:
                    ctx.topology = result
                self._check_topology_budget(ctx)
                return ctx
            except (ImportError, RuntimeError) as exc:
                log.warning(
                    "Stage 2 topology engine failed: %s, using template", exc
                )

        # Fallback: create topology from template
        try:
            from sage_core import TopologyGraph, TopologyNode  # type: ignore[import-not-found]

            topo = TopologyGraph(hint)
            node = TopologyNode(role="agent", model_id="", system=ctx.system)
            topo.add_node(node)
            ctx.topology = topo
            log.debug("Stage 2 fallback to template: %s", hint)
        except ImportError:
            log.debug("sage_core unavailable, topology=None (single-agent mode)")
            ctx.topology = None

        self._check_topology_budget(ctx)
        return ctx

    def _check_topology_budget(self, ctx: PipelineContext) -> None:
        """Pre-validate budget feasibility — degrade to single-node if over budget."""
        if ctx.topology and hasattr(ctx.topology, 'node_count'):
            total_node_cost = 0.0
            nc = ctx.topology.node_count()
            for i in range(nc):
                node = ctx.topology.get_node(i) if hasattr(ctx.topology, 'get_node') else None
                if node:
                    total_node_cost += getattr(node, 'max_cost_usd', 0.0)
            if total_node_cost > ctx.budget:
                log.warning(
                    "Topology budget %.2f > pipeline budget %.2f — degrading to single-node",
                    total_node_cost, ctx.budget,
                )
                self._emit("TOPOLOGY_BUDGET_WARNING", {"total_cost": total_node_cost, "budget": ctx.budget})
                # Degrade: replace with single-node template topology
                ctx.topology = self._make_single_node_topology(ctx)

    def _make_single_node_topology(self, ctx: PipelineContext) -> Any:
        """Create a minimal single-node topology as budget-safe fallback."""
        try:
            from sage_core import TopologyGraph, TopologyNode  # type: ignore[import-not-found]

            topo = TopologyGraph("sequential")
            node = TopologyNode(role="agent", model_id="", system=ctx.system)
            topo.add_node(node)
            return topo
        except ImportError:
            log.debug("sage_core unavailable, topology=None (single-agent mode)")
            return None

    # ── Cost Estimation ──────────────────────────────────────────────────────

    def _estimate_topology_cost(self, ctx: PipelineContext) -> float:
        """Estimate execution cost from per-model pricing in cards.toml.

        Loads the model catalog once (cached) and looks up cost_input_per_m /
        cost_output_per_m for each node's assigned model.  Falls back to
        $0.001 per node when the catalog or model is unavailable.

        Assumes ~500 input tokens + ~300 output tokens per node call (conservative
        estimate for typical multi-agent pipeline nodes).
        """
        if not ctx.topology or not hasattr(ctx.topology, 'node_count'):
            return 0.0

        n_nodes = ctx.topology.node_count()
        if n_nodes == 0:
            return 0.0

        # Lazy-load model catalog for pricing
        catalog = self._load_model_catalog()

        total_cost = 0.0
        for i in range(n_nodes):
            model_id = ctx.assignments.get(i, '')
            if not model_id and hasattr(ctx.topology, 'get_node'):
                node = ctx.topology.get_node(i)
                model_id = getattr(node, 'model_id', '') if node else ''

            card = catalog.get(model_id) if catalog and model_id else None
            if card:
                # ~500 tokens input + ~300 tokens output per node
                total_cost += (
                    500 * card.cost_input_per_m + 300 * card.cost_output_per_m
                ) / 1_000_000
            else:
                total_cost += 0.001  # fallback estimate per node

        return total_cost

    def _load_model_catalog(self) -> Any:
        """Load ModelCardCatalog from cards.toml (cached after first call)."""
        if hasattr(self, '_model_catalog'):
            return self._model_catalog

        self._model_catalog = None
        # Try Python ModelCardCatalog (always available, no Rust dependency)
        try:
            from sage.llm.model_registry import ModelCardCatalog
            from pathlib import Path

            # Search common locations for cards.toml
            for candidate in [
                Path("config/cards.toml"),
                Path("sage-python/config/cards.toml"),
                Path(__file__).parent.parent.parent / "config" / "cards.toml",
            ]:
                if candidate.exists():
                    self._model_catalog = ModelCardCatalog.from_toml_file(str(candidate))
                    log.debug("Cost estimator: loaded %d models from %s",
                              len(self._model_catalog), candidate)
                    break
        except (IOError, OSError, ValueError) as exc:
            log.debug("Cost estimator: catalog unavailable (%s)", exc)

        return self._model_catalog

    # ── Stage 3: Assign Models ──────────────────────────────────────────────

    def _stage_assign_models(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Assign model_id to each topology node."""
        if ctx.topology is None or self.assigner is None:
            return ctx

        try:
            # Pass provider hints from Path 6 policy output (multi-provider dimension).
            hints_list = (
                list(ctx.provider_hints.items()) if ctx.provider_hints else None
            )
            # F7: forward the OVERALL task tier so the Rust ModelAssigner can
            # promote producer-role nodes (planner, coder, worker, verifier)
            # that the template hardcoded at a low system tier. Without this,
            # an S3 SWE-bench task's planner (template system=1) was matched
            # against S1 affinity and picked a flash-lite model — see
            # docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
            task_system = ctx.system if ctx.system in (1, 2, 3) else None
            n_assigned = self.assigner.assign_models(
                ctx.topology,
                ctx.domain,
                ctx.budget,
                hints_list,
                task_system,
            )
            log.info(
                "Assigned models to %d nodes (domain=%s, budget=%.2f, task_system=%s, provider_hints=%d)",
                n_assigned,
                ctx.domain,
                ctx.budget,
                task_system,
                len(ctx.provider_hints),
            )

            # Record assignments for observability
            node_count = (
                ctx.topology.node_count()
                if hasattr(ctx.topology, "node_count")
                else 0
            )
            for i in range(node_count):
                node = (
                    ctx.topology.get_node(i)
                    if hasattr(ctx.topology, "get_node")
                    else None
                )
                if node:
                    ctx.assignments[i] = getattr(node, "model_id", "")
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 assign failed: %s", exc)

        # Bandit feedback: Thompson sampling already handles exploration/exploitation
        # via the Beta posterior. No pre-execution budget reduction — it creates a
        # self-degrading loop (Audit2 + Audit3 confirmed). The bandit learns post-
        # execution in Stage 5 (LEARN) and naturally deprioritizes bad arms.

        # Filter out models whose provider is dead (health check or circuit breaker)
        if self.provider_pool and hasattr(self.provider_pool, 'is_model_available'):
            for node_idx, model_id in list(ctx.assignments.items()):
                if not model_id:
                    continue
                if not self.provider_pool.is_model_available(model_id):
                    provider_name = self.provider_pool.infer_provider(model_id)
                    default_model = getattr(self.llm_config, 'model', '') if self.llm_config else ''
                    if default_model:
                        log.info(
                            "Stage 3: %s provider unavailable, "
                            "node %d reassigned %s -> %s",
                            provider_name, node_idx, model_id, default_model,
                        )
                        ctx.assignments[node_idx] = default_model
                        if hasattr(ctx.topology, 'set_node_model_id'):
                            ctx.topology.set_node_model_id(node_idx, default_model)

        # Formal verification (non-blocking): prove every node has a valid provider
        try:
            self._verify_assignment_formal(ctx)
        except (ImportError, RuntimeError) as exc:
            log.warning("Stage 3 formal verification error (non-blocking): %s", exc)

        return ctx

    def _verify_assignment_formal(self, ctx: PipelineContext) -> None:
        """Formally verify provider assignment via OxiZ / Z3 (NON-BLOCKING).

        Builds a lightweight adapter that bridges TopologyGraph nodes into the
        interface expected by ``verify_provider_assignment`` without requiring
        a full TaskDAG conversion.

        Skips silently when:
        - No SMT backend is available (ImportError from z3_verify)
        - topology is None
        - No nodes with capability requirements are present
        """
        if not _Z3_VERIFY_AVAILABLE or ctx.topology is None:
            return

        node_count = (
            ctx.topology.node_count()
            if hasattr(ctx.topology, "node_count")
            else 0
        )
        if node_count == 0:
            return

        # ── Build minimal adapter objects ──────────────────────────────────

        # Collect (node_index, capabilities) from topology
        topo_nodes: list[tuple[str, list[str]]] = []
        for i in range(node_count):
            node = (
                ctx.topology.get_node(i)
                if hasattr(ctx.topology, "get_node")
                else None
            )
            if node is None:
                continue
            # Capabilities: TopologyNode may expose .capabilities or .capabilities_required
            caps: list[str] = []
            for attr in ("capabilities", "capabilities_required"):
                raw = getattr(node, attr, None)
                if raw:
                    caps = list(raw)
                    break
            topo_nodes.append((str(i), caps))

        # Only verify if at least one node has capability requirements
        if not any(caps for _, caps in topo_nodes):
            return

        # ── DAG adapter ────────────────────────────────────────────────────

        class _NodeAdapter:
            """Minimal shim that looks like TaskNode to z3_verify."""

            def __init__(self, nid: str, capabilities: list[str]) -> None:
                self._nid = nid
                self.capabilities_required = capabilities

        class _DagAdapter:
            """Minimal shim that looks like TaskDAG to z3_verify."""

            def __init__(self, nodes: list[tuple[str, list[str]]]) -> None:
                self._nodes = {nid: _NodeAdapter(nid, caps) for nid, caps in nodes}

            @property
            def node_ids(self) -> list[str]:
                return list(self._nodes.keys())

            def get_node(self, nid: str) -> _NodeAdapter | None:
                return self._nodes.get(nid)

        dag_adapter = _DagAdapter(topo_nodes)

        # ── ProviderSpec list: one entry per distinct assigned model_id ────

        # Build providers from assigned model_ids.
        # Priority: ctx.assignments (set by assigner) > topology node model_id attribute.
        # Each model is treated as a provider that offers the capabilities
        # of the node it was assigned to (optimistic: if a model was chosen
        # for a node, it can serve that node's capabilities).
        model_caps: dict[str, set[str]] = {}

        # Try ctx.assignments first (populated by _stage_assign_models assigner)
        for i, model_id in ctx.assignments.items():
            if not model_id:
                continue
            nid = str(i)
            node = dag_adapter.get_node(nid)
            caps = set(node.capabilities_required) if node else set()
            if model_id not in model_caps:
                model_caps[model_id] = set()
            model_caps[model_id].update(caps)

        # Fallback: read model_id directly from topology nodes
        if not model_caps:
            for nid, caps in topo_nodes:
                node_obj = (
                    ctx.topology.get_node(int(nid))
                    if hasattr(ctx.topology, "get_node")
                    else None
                )
                model_id = getattr(node_obj, "model_id", "") if node_obj else ""
                if not model_id:
                    continue
                if model_id not in model_caps:
                    model_caps[model_id] = set()
                model_caps[model_id].update(caps)

        if not model_caps:
            log.debug(
                "Stage 3 formal verify: no model_ids found in topology, skipping SAT check"
            )
            return

        providers = [
            ProviderSpec(name=model_id, capabilities=caps)
            for model_id, caps in model_caps.items()
        ]

        # ── Run SAT check ──────────────────────────────────────────────────

        try:
            verdict = verify_provider_assignment(dag_adapter, providers)  # type: ignore[arg-type]
        except ImportError as exc:
            log.debug("Stage 3 formal verify skipped (no SMT backend): %s", exc)
            return
        except RuntimeError as exc:
            log.warning("Stage 3 formal verify raised unexpected error: %s", exc)
            return

        if not verdict.satisfied:
            ctx.verification_passed = False
            log.warning(
                "Stage 3 formal provider assignment verification FAILED "
                "(non-blocking): %s",
                verdict.counterexample,
            )
            self._emit(
                "ASSIGN_MODELS_VERIFY_FAIL",
                {"counterexample": verdict.counterexample or "UNSAT"},
            )
        else:
            log.debug(
                "Stage 3 formal provider assignment verification PASSED"
            )

    # ── Stage 4: Execute ────────────────────────────────────────────────────

    async def _stage_execute(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 4: Execute topology with per-node model resolution."""
        # Bandit: choose arm BEFORE execution to get decision_id
        # Pass task context features when available for contextual arm selection
        if self.bandit and hasattr(self.bandit, "select_with_context"):
            try:
                task_context = [
                    float(ctx.system),  # cognitive system tier (1, 2, or 3)
                    float(len(ctx.task)),  # task length as complexity proxy
                    float(
                        ctx.topology.node_count()
                        if ctx.topology and hasattr(ctx.topology, "node_count")
                        else 0
                    ),  # topology complexity
                ]
                decision = self.bandit.select_with_context(0.1, task_context)
                ctx.bandit_decision_id = decision.decision_id
            except (ImportError, RuntimeError):
                pass
        elif self.bandit and hasattr(self.bandit, "select"):
            try:
                decision = self.bandit.select(0.1)  # 10% exploration
                ctx.bandit_decision_id = decision.decision_id
            except (ImportError, RuntimeError):
                pass

        if not ctx.verification_passed:
            log.warning("Stage 4: executing with unverified provider assignment (SAT check failed)")
            self._emit("EXECUTE_UNVERIFIED", {"reason": "SAT check failed in Stage 3"})

        # Single-agent mode (no topology or single node)
        if ctx.topology is None or (
            hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
        ):
            if self._agent_loop:
                # Phase 1: agent_loop.run() provides tools + S2/S3 validation +
                # guardrails + memory. Replaces the raw provider.generate() loop.

                # H1: Skip routing in agent_loop (pipeline already routed in Stage 0)
                self._agent_loop._skip_routing = True
                # H4: Clear topology (pipeline owns topology, not agent_loop)
                self._agent_loop._current_topology = None

                # Set validation level from system classification
                if ctx.system >= 3:
                    self._agent_loop.config.validation_level = 3
                elif ctx.system >= 2 and self._agent_loop.sandbox_manager:
                    self._agent_loop.config.validation_level = 2
                else:
                    self._agent_loop.config.validation_level = 1

                # Resolve model from Rust routing decision (preserve model selection)
                routing_decision = getattr(self, '_last_routing_decision', None)
                _original_llm = self._agent_loop._llm
                _original_config = self._agent_loop.config.llm
                if routing_decision and routing_decision.model_id and self.provider_pool:
                    try:
                        if self.provider_pool.is_model_available(routing_decision.model_id):
                            resolved_provider, resolved_config = self.provider_pool.resolve(
                                routing_decision.model_id
                            )
                            self._agent_loop._llm = resolved_provider
                            self._agent_loop.config.llm = resolved_config
                            log.info(
                                "Stage 4 bypass: agent_loop using Rust-selected %s (S%d)",
                                routing_decision.model_id, ctx.system,
                            )
                    except Exception:
                        pass  # Keep default provider

                try:
                    ctx.result = await self._agent_loop.run(ctx.task)
                    ctx.cost = self._agent_loop.total_cost_usd
                finally:
                    # Restore agent_loop state (safe for next run)
                    self._agent_loop._skip_routing = False
                    self._agent_loop._llm = _original_llm
                    self._agent_loop.config.llm = _original_config

            elif self.llm_provider:
                # Simple fallback: single provider.generate() call (no tool loop).
                # Used only when pipeline is created without agent_loop (e.g., tests).
                from sage.llm.base import Message, Role

                messages = [Message(role=Role.USER, content=ctx.task)]
                try:
                    response = await self.llm_provider.generate(
                        messages=messages, config=self.llm_config,
                    )
                    ctx.result = response.content or ""
                except (RuntimeError, TimeoutError) as exc:
                    log.error("Stage 4 fallback failed: %s", exc)
                    ctx.result = f"Error: {exc}"
            return ctx

        # Multi-agent mode: use TopologyRunner with ProviderPool
        try:
            from sage.topology.runner import TopologyRunner  # type: ignore[import-not-found]

            # Get executor
            try:
                from sage_core import TopologyExecutor  # type: ignore[import-not-found]

                executor = TopologyExecutor(ctx.topology)
            except ImportError:
                log.warning("sage_core TopologyExecutor unavailable, falling back")
                ctx.result = "Error: TopologyExecutor unavailable"
                return ctx

            # Phase 2: create agent_loop factory for per-node execution
            _agent_loop_factory = None
            if self._agent_loop and self.tool_registry:
                from sage.agent_loop_factory import create_node_agent_loop
                from functools import partial

                _agent_loop_factory = partial(
                    create_node_agent_loop,
                    tool_registry=self.tool_registry,
                    system_level=ctx.system,
                    on_event=(
                        self.event_bus.emit
                        if self.event_bus and hasattr(self.event_bus, "emit")
                        else None
                    ),
                )

            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
                provider_pool=self.provider_pool,
                controller=self.controller,  # Phase C
                axis_hint=ctx.axis_hint,
                agent_loop_factory=_agent_loop_factory,
            )
            result = await runner.run(ctx.task)
            if result == "__REROUTE__" and self.engine:
                log.info("Topology reroute triggered — REBUILDING full topology (not in-place mutation)")
                self._emit("REROUTE_REBUILD", {"reason": "controller_triggered"})
                ctx = self._stage_select_topology(ctx)  # new topology
                ctx = self._stage_assign_models(ctx)    # re-assign models
                # Refresh bandit decision for the new topology
                if self.bandit and hasattr(self.bandit, "select_with_context"):
                    try:
                        task_context = [
                            float(ctx.system),
                            float(len(ctx.task)),
                            float(
                                ctx.topology.node_count()
                                if ctx.topology and hasattr(ctx.topology, "node_count")
                                else 0
                            ),
                        ]
                        new_decision = self.bandit.select_with_context(0.1, task_context)
                        ctx.bandit_decision_id = new_decision.decision_id
                    except (ImportError, RuntimeError):
                        pass
                elif self.bandit and hasattr(self.bandit, "select"):
                    try:
                        new_decision = self.bandit.select(0.1)
                        ctx.bandit_decision_id = new_decision.decision_id
                    except (ImportError, RuntimeError):
                        pass
                # Fresh executor for the regenerated topology (old one is stale)
                from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                executor_rerouted = _TE(ctx.topology)
                # Re-execute with new topology (no controller to avoid infinite loop)
                runner2 = TopologyRunner(
                    graph=ctx.topology, executor=executor_rerouted,
                    llm_provider=self.llm_provider, llm_config=self.llm_config,
                    provider_pool=self.provider_pool,
                    controller=None,  # no controller on retry to prevent loop
                    agent_loop_factory=_agent_loop_factory,
                )
                result = await runner2.run(ctx.task)

            # FrugalGPT quality-gated cascade: if result quality is low, retry with upgraded models
            if result and result != "__REROUTE__" and self.quality_estimator:
                quality = self.quality_estimator.estimate(ctx.task, result)
                if quality is not None and quality < 0.3 and self.assigner:
                    log.info("Stage 4: quality=%.2f < 0.3, triggering FrugalGPT cascade retry", quality)
                    # Reassign with upgraded models (exclude current + budget escalation)
                    try:
                        if hasattr(ctx.topology, 'node_count'):
                            for i in range(ctx.topology.node_count()):
                                if self.assigner and hasattr(self.assigner, 'assign_single_node'):
                                    current_model = ctx.assignments.get(i, "")
                                    # F7 wiring (2026-04-17): forward task_system so the
                                    # Rust ModelAssigner promotes producer nodes correctly
                                    # during the cascade upgrade (otherwise the upgrade picks
                                    # the next best per-node-tier model, ignoring the overall
                                    # task complexity).
                                    #
                                    # Interaction note (advisor 2026-04-17): the cascade
                                    # stays at the F7-effective tier (S2 floor for non-rigour
                                    # S3 tasks). It does NOT escalate beyond what F7 already
                                    # set — exhausting S2 candidates before touching S3.
                                    # That's intentional: cascade is "swap to a different
                                    # model in the same tier", not "tier-escalate". If a
                                    # task genuinely needs an S3 model on a node F7 floored
                                    # at S2, that's a separate routing decision (not yet
                                    # implemented; would need a TierEscalator).
                                    cascade_task_system = (
                                        ctx.system if isinstance(getattr(ctx, "system", None), int)
                                        and ctx.system in (1, 2, 3) else None
                                    )
                                    try:
                                        self.assigner.assign_single_node(
                                            ctx.topology, i, ctx.domain,
                                            ctx.budget * 1.5,
                                            exclude_model_ids=[current_model] if current_model else None,
                                            task_system=cascade_task_system,
                                        )
                                    except TypeError:
                                        # Older binding without task_system kwarg.
                                        try:
                                            self.assigner.assign_single_node(
                                                ctx.topology, i, ctx.domain,
                                                ctx.budget * 1.5,
                                                exclude_model_ids=[current_model] if current_model else None,
                                            )
                                        except (ValueError, RuntimeError):
                                            pass
                                    except (ValueError, RuntimeError):
                                        pass
                                # Verify upgraded model has an available provider
                                node = ctx.topology.get_node(i) if hasattr(ctx.topology, 'get_node') else None
                                new_model = getattr(node, 'model_id', '') if node else ''
                                if new_model and self.provider_pool and hasattr(self.provider_pool, 'is_model_available'):
                                    if not self.provider_pool.is_model_available(new_model):
                                        default_model = getattr(self.llm_config, 'model', '') if self.llm_config else ''
                                        if default_model and hasattr(ctx.topology, 'set_node_model_id'):
                                            ctx.topology.set_node_model_id(i, default_model)
                                            log.debug("FrugalGPT: reverted node %d %s -> %s (provider dead)", i, new_model, default_model)
                        # Re-execute with upgraded models
                        from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                        executor2 = _TE(ctx.topology)
                        runner3 = TopologyRunner(
                            graph=ctx.topology, executor=executor2,
                            llm_provider=self.llm_provider, llm_config=self.llm_config,
                            provider_pool=self.provider_pool,
                            agent_loop_factory=_agent_loop_factory,
                        )
                        retry_result = await runner3.run(ctx.task)
                        if retry_result:
                            result = retry_result
                            log.info("Stage 4: FrugalGPT cascade succeeded on retry")
                    except (RuntimeError, TimeoutError) as exc:
                        log.debug("Stage 4: FrugalGPT cascade retry failed: %s", exc)

            ctx.result = result
            # Estimate cost from topology execution
            # Uses per-model pricing from cards.toml when available
            ctx.cost = self._estimate_topology_cost(ctx)
        except (ImportError, RuntimeError, TimeoutError) as exc:
            log.error("Stage 4 multi-agent execution failed: %s — falling back to single-agent", exc)
            # Fallback: run task directly with default provider
            if self.llm_provider:
                try:
                    from sage.llm.base import Message, Role
                    response = await self.llm_provider.generate(
                        messages=[Message(role=Role.USER, content=ctx.task)],
                        config=self.llm_config,
                    )
                    ctx.result = response.content or ""
                    log.info("Stage 4 fallback single-agent succeeded (%d chars)", len(ctx.result))
                except (RuntimeError, TimeoutError) as fallback_exc:
                    log.error("Stage 4 fallback also failed: %s", fallback_exc)
                    ctx.result = ""
            else:
                ctx.result = ""

        return ctx

    # ── Stage 5: Learn ──────────────────────────────────────────────────────

    async def _stage_learn(self, ctx: PipelineContext) -> None:
        """Stage 5: Record outcome for learning.

        Quality signal for bandit feedback (ETH-SRI ICLR '25, PILOT 2508.21141):
        - Empty result: quality = 0.0 (definitively bad, bandit learns from it)
        - QualityEstimator returns float: use it
        - QualityEstimator returns None: abstain — bandit does NOT record
        - No estimator: abstain — bandit does NOT record
        """
        import re

        quality: float | None = None

        # Empty result => total failure, bandit must learn from it
        if not ctx.result or not ctx.result.strip():
            quality = 0.0
        elif self.quality_estimator:
            try:
                quality = self.quality_estimator.estimate(
                    ctx.task, ctx.result, ctx.latency_ms
                )
            except (ImportError, RuntimeError):
                quality = None  # cannot assess — abstain

        # PRM lightweight scoring (Phase C) — 6th formal signal
        # Guard: only call PRM on structured content (<think>, assert, code)
        # Only blend when quality is known (not None)
        _STRUCTURED = re.compile(r'<think>|```|assert\s|def\s+test_', re.IGNORECASE)
        if self.prm and quality is not None and ctx.result and _STRUCTURED.search(ctx.result):
            try:
                r_path, _ = self.prm.calculate_r_path(ctx.result)
                if r_path >= 0.0:  # valid score (negative = penalty for no reasoning)
                    quality = 0.8 * quality + 0.2 * r_path
                    log.debug("PRM blended quality: %.2f (estimator + PRM)", quality)
            except (RuntimeError, ValueError) as exc:
                log.warning("PRM scoring failed in LEARN: %s", exc)

        # Only record to bandit when quality is known — never guess
        if quality is not None and self.bandit and hasattr(self.bandit, "record_outcome"):
            if ctx.bandit_decision_id:
                try:
                    self.bandit.record_outcome(ctx.bandit_decision_id, quality, ctx.cost, ctx.latency_ms)
                    log.debug("Bandit outcome recorded (in-memory, not persisted across restarts)")
                except (ImportError, RuntimeError):
                    pass

        # Evolution feedback: record outcome in TopologyEngine archive
        # Feeds MAP-Elites + CMA-ME + S-MMU bridge for future topology selection
        if self.engine and quality is not None and ctx.topology is not None:
            try:
                topology_id = ctx.topology_id or getattr(ctx.topology, 'id', '')
                if topology_id and hasattr(self.engine, 'record_outcome'):
                    keywords = list(set(
                        w.lower() for w in re.findall(r'\b\w{4,}\b', ctx.task)
                    ))[:10]
                    # Compute real task embedding for S-MMU retrieval
                    task_embedding = None
                    try:
                        from sage.memory.embedder import Embedder
                        _embedder = Embedder()
                        if _embedder.is_semantic:
                            task_embedding = _embedder.embed(ctx.task[:500])
                    except (ImportError, RuntimeError):
                        pass  # Embedding unavailable, degrade gracefully

                    self.engine.record_outcome(
                        topology_id,
                        ctx.task[:200],
                        keywords,
                        task_embedding,  # real embedding instead of None
                        quality,
                        ctx.cost,
                        ctx.latency_ms,
                    )
                    log.debug(
                        "Evolution: recorded outcome for topology %s (quality=%.2f)",
                        topology_id[:8], quality,
                    )
            except (ImportError, RuntimeError) as exc:
                log.debug("Evolution feedback failed: %s", exc)

        # ── Periodic maintenance ───────────────────────────────────────────
        self._task_count += 1

        # Inter-tier consolidation: episodic → semantic → causal (MAGMA 2601.03236)
        from sage.constants import CONSOLIDATION_INTERVAL_STEPS
        if (self._task_count % CONSOLIDATION_INTERVAL_STEPS == 0
                and self.consolidator is not None):
            try:
                consolidation_result = await self.consolidator.consolidate()
                if hasattr(consolidation_result, 'processed') and consolidation_result.processed > 0:
                    log.debug(
                        "Pipeline consolidation: %d episodes → %d entities",
                        consolidation_result.processed,
                        getattr(consolidation_result, 'entities_added', 0),
                    )
            except (RuntimeError, IOError):
                pass  # Best-effort, never blocks pipeline

        # Bandit + MAP-Elites state persistence (crash-safe, WAL write ~5ms)
        from sage.constants import BANDIT_FLUSH_INTERVAL
        if (self._task_count % BANDIT_FLUSH_INTERVAL == 0
                and self.engine and hasattr(self.engine, 'save_state')):
            try:
                from pathlib import Path
                state_dir = str(Path.home() / ".sage")
                self.engine.save_state(state_dir)
                log.debug("Periodic state flush (%d tasks)", self._task_count)
            except (RuntimeError, IOError):
                pass  # Best-effort, never blocks pipeline
