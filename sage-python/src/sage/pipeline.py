"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sage.pipeline_stages import (
    _infer_domain,
    compute_dag_features,
    select_macro_topology,
    DAGFeatures,
)

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
    assignments: dict[int, str] = field(default_factory=dict)
    result: str = ""
    latency_ms: float = 0.0


class CognitiveOrchestrationPipeline:
    """5-stage pipeline: Classify -> Decompose -> Select Topology -> Assign Models -> Execute.

    Parameters
    ----------
    router : AdaptiveRouter or ComplexityRouter
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
            except Exception:
                pass

    async def run(self, task: str, budget_usd: float = 5.0) -> str:
        """Execute the full 5-stage pipeline."""
        t0 = time.monotonic()
        ctx = PipelineContext(task=task, budget=budget_usd)

        # Stage 0: CLASSIFY
        ctx = self._stage_classify(ctx)
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

        # Stage 5: LEARN
        self._stage_learn(ctx)
        self._emit("LEARN", {"latency_ms": ctx.latency_ms})

        return ctx.result

    # ── Stage 0: Classify ───────────────────────────────────────────────────

    def _stage_classify(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 0: Classify task complexity and domain."""
        if self.router:
            try:
                profile = self.router.assess_complexity(ctx.task)
                decision = self.router.route(profile)
                ctx.system = getattr(decision, "system", 2)
            except Exception as exc:
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
        except Exception as exc:
            log.warning("Stage 1 decompose failed: %s, using single-node DAG", exc)
            ctx.dag_features = DAGFeatures(omega=1, delta=1, gamma=0.0)

        return ctx

    # ── Stage 2: Select Topology ────────────────────────────────────────────

    def _stage_select_topology(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Select optimal topology."""
        # Path 0: AdaptOrch heuristic for macro topology hint
        hint = "sequential"
        if ctx.dag_features:
            hint = select_macro_topology(ctx.dag_features)

        # Try DynamicTopologyEngine
        if self.engine:
            try:
                result = self.engine.generate(ctx.task, ctx.system, ctx.budget)
                if result and hasattr(result, "topology"):
                    ctx.topology = result.topology
                elif result:
                    ctx.topology = result
                return ctx
            except Exception as exc:
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
        except ImportError:
            log.debug("sage_core unavailable, topology=None (single-agent mode)")
            ctx.topology = None

        return ctx

    # ── Stage 3: Assign Models ──────────────────────────────────────────────

    def _stage_assign_models(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Assign model_id to each topology node."""
        if ctx.topology is None or self.assigner is None:
            return ctx

        try:
            n_assigned = self.assigner.assign_models(
                ctx.topology, ctx.domain, ctx.budget
            )
            log.info(
                "Assigned models to %d nodes (domain=%s, budget=%.2f)",
                n_assigned,
                ctx.domain,
                ctx.budget,
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
        except Exception as exc:
            log.warning("Stage 3 assign failed: %s", exc)

        return ctx

    # ── Stage 4: Execute ────────────────────────────────────────────────────

    async def _stage_execute(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 4: Execute topology with per-node model resolution."""
        # Single-agent mode (no topology or single node)
        if ctx.topology is None or (
            hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
        ):
            # Use LLM provider directly
            if self.llm_provider:
                from sage.llm.base import Message, Role

                try:
                    response = await self.llm_provider.generate(
                        messages=[Message(role=Role.USER, content=ctx.task)],
                        config=self.llm_config,
                    )
                    ctx.result = response.content or ""
                except Exception as exc:
                    log.error("Stage 4 single-agent execution failed: %s", exc)
                    ctx.result = f"Error: {exc}"
            return ctx

        # Multi-agent mode: use TopologyRunner with ProviderPool
        try:
            from sage.topology.runner import TopologyRunner  # type: ignore[import-not-found]

            # Get executor
            try:
                from sage_core import TopologyExecutor  # type: ignore[import-not-found]

                executor = TopologyExecutor()
            except ImportError:
                log.warning("sage_core TopologyExecutor unavailable, falling back")
                ctx.result = "Error: TopologyExecutor unavailable"
                return ctx

            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
                provider_pool=self.provider_pool,
            )
            ctx.result = await runner.run(ctx.task)
        except Exception as exc:
            log.error("Stage 4 multi-agent execution failed: %s", exc)
            ctx.result = f"Error: {exc}"

        return ctx

    # ── Stage 5: Learn ──────────────────────────────────────────────────────

    def _stage_learn(self, ctx: PipelineContext) -> None:
        """Stage 5: Record outcome for learning."""
        quality = 0.5
        if self.quality_estimator and ctx.result:
            try:
                quality = self.quality_estimator.estimate(
                    ctx.task, ctx.result, ctx.latency_ms / 1000.0
                )
            except Exception:
                pass

        if self.bandit and hasattr(self.bandit, "record"):
            try:
                self.bandit.record("pipeline", quality, 0.0, ctx.latency_ms)
            except Exception:
                pass
