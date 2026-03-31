"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import logging
import os
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
        prm: Any = None,
        controller: Any = None,
        smmu: Any = None,
        consolidator: Any = None,
        tool_forge: Any = None,
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
        self._smmu = smmu
        self.consolidator = consolidator
        self.tool_forge = tool_forge
        self._task_count = 0

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
        await self._stage_learn(ctx)
        self._emit("LEARN", {"latency_ms": ctx.latency_ms})

        return ctx.result

    # ── Stage 0: Classify ───────────────────────────────────────────────────

    def _stage_classify(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 0: Classify task complexity and domain.

        Priority: Rust kNN (92%) > AdaptiveRouter kNN > heuristic (34%).
        """
        # Priority 1: kNN router (92% accuracy, Rust-accelerated)
        if self.router and hasattr(self.router, '_knn') and self.router._knn is not None:
            try:
                knn_result = self.router._knn.route(ctx.task)
                if knn_result is not None:
                    ctx.system = knn_result.system
                    log.info("Stage 0: kNN routing → S%d (conf=%.2f, %s)",
                             knn_result.system, knn_result.confidence, knn_result.method)
                    ctx.domain = _infer_domain(ctx.task)
                    return ctx
            except Exception as exc:
                log.debug("Stage 0: kNN failed (%s), falling back", exc)

        # Priority 2: AdaptiveRouter / ComplexityRouter
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
        """Stage 2: Select optimal topology.

        S1 (simple) tasks skip topology entirely — direct single-agent call
        is faster AND equally effective (confirmed by MASBENCH: topology helps
        only when base accuracy < 60%, per AdaptOrch arXiv 2602.16873).
        """
        # S1 fast path: skip topology, use direct single-agent call
        # This reduces latency from ~200s (multi-node) to ~15s (direct)
        if ctx.system == 1:
            ctx.topology = None
            log.debug("S1 task: skipping topology (direct single-agent)")
            return ctx

        # Path 0: AdaptOrch heuristic for macro topology hint
        hint = "sequential"
        if ctx.dag_features:
            hint = select_macro_topology(ctx.dag_features)

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
                except Exception:
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
            except Exception as exc:
                log.warning(
                    "Stage 2 topology engine failed: %s, using template", exc
                )

        # Path 6: Learned topology policy (Nemotron-8B V2 or Phi-4-mini V1)
        # Only enable when explicitly opted in (heavy: loads 8B model on GPU)
        if os.environ.get("SAGE_ENABLE_PATH6"):
            try:
                from sage.topology.llm_caller import generate_topology_from_policy

                policy_result = generate_topology_from_policy(ctx.task)
                if policy_result and "nodes" in policy_result:
                    from sage_core import TopologyGraph, TopologyNode, TopologyEdge  # type: ignore[import-not-found]

                    topo = TopologyGraph("learned_policy")
                    for node_data in policy_result["nodes"]:
                        model_tier = node_data.get("model_tier", "")
                        node = TopologyNode(
                            role=node_data.get("role", "agent"),
                            model_id=model_tier,
                            system=ctx.system,
                            prompt=node_data.get("prompt", ""),
                        )
                        topo.add_node(node)

                    # Parse edges from policy output (V2 produces edges)
                    for edge_data in policy_result.get("edges", []):
                        if isinstance(edge_data, dict):
                            fi = edge_data.get("from_idx", 0)
                            ti = edge_data.get("to_idx", 0)
                            if 0 <= fi < topo.node_count() and 0 <= ti < topo.node_count():
                                flow = edge_data.get("flow_type", "message")
                                topo.add_edge(fi, ti, TopologyEdge(flow))

                    # Add sequential edges if policy didn't provide any
                    if topo.edge_count() == 0 and topo.node_count() > 1:
                        for i in range(topo.node_count() - 1):
                            topo.add_edge(i, i + 1, TopologyEdge("message"))

                    ctx.topology = topo
                    # Store provider hints for Stage 3 (multi-provider dimension)
                    ctx.provider_hints = {
                        i: node_data.get("provider_hint", "")
                        for i, node_data in enumerate(policy_result["nodes"])
                        if node_data.get("provider_hint")
                    }
                    log.info(
                        "Stage 2 Path 6 (learned policy): %d nodes, %d edges, %d provider hints",
                        topo.node_count(), topo.edge_count(), len(ctx.provider_hints),
                    )
                    self._check_topology_budget(ctx)
                    return ctx
            except Exception as exc:
                log.debug("Path 6 failed: %s", str(exc)[:100])

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
        except Exception as exc:
            log.debug("Cost estimator: catalog unavailable (%s)", exc)

        return self._model_catalog

    # ── Stage 3: Assign Models ──────────────────────────────────────────────

    def _stage_assign_models(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Assign model_id to each topology node."""
        if ctx.topology is None or self.assigner is None:
            return ctx

        try:
            # Pass provider hints from Path 6 policy output (multi-provider dimension)
            hints_list = (
                list(ctx.provider_hints.items()) if ctx.provider_hints else None
            )
            n_assigned = self.assigner.assign_models(
                ctx.topology, ctx.domain, ctx.budget, hints_list
            )
            log.info(
                "Assigned models to %d nodes (domain=%s, budget=%.2f, provider_hints=%d)",
                n_assigned,
                ctx.domain,
                ctx.budget,
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
        except Exception as exc:
            log.warning("Stage 3 assign failed: %s", exc)

        # Filter out models assigned to unavailable providers (circuit breaker open)
        if self.provider_pool and hasattr(self.provider_pool, 'is_available'):
            for node_idx, model_id in list(ctx.assignments.items()):
                profile = (
                    self.provider_pool._registry.get(model_id)
                    if self.provider_pool._registry else None
                )
                provider_name = getattr(profile, 'provider', '') if profile else ''
                if provider_name and not self.provider_pool.is_available(provider_name):
                    default_model = getattr(self.llm_config, 'model', '') if self.llm_config else ''
                    if default_model:
                        log.info(
                            "Stage 3: %s provider unavailable (circuit open), "
                            "node %d reassigned %s -> %s",
                            provider_name, node_idx, model_id, default_model,
                        )
                        ctx.assignments[node_idx] = default_model
                        if hasattr(ctx.topology, 'set_node_model_id'):
                            ctx.topology.set_node_model_id(node_idx, default_model)

        # Formal verification (non-blocking): prove every node has a valid provider
        try:
            self._verify_assignment_formal(ctx)
        except Exception as exc:
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
        except Exception as exc:
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
            except Exception:
                pass
        elif self.bandit and hasattr(self.bandit, "select"):
            try:
                decision = self.bandit.select(0.1)  # 10% exploration
                ctx.bandit_decision_id = decision.decision_id
            except Exception:
                pass

        if not ctx.verification_passed:
            log.warning("Stage 4: executing with unverified provider assignment (SAT check failed)")
            self._emit("EXECUTE_UNVERIFIED", {"reason": "SAT check failed in Stage 3"})

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

                executor = TopologyExecutor(ctx.topology)
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
                controller=self.controller,  # Phase C
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
                    except Exception:
                        pass
                elif self.bandit and hasattr(self.bandit, "select"):
                    try:
                        new_decision = self.bandit.select(0.1)
                        ctx.bandit_decision_id = new_decision.decision_id
                    except Exception:
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
                                    try:
                                        self.assigner.assign_single_node(
                                            ctx.topology, i, ctx.domain,
                                            ctx.budget * 1.5,  # Budget escalation
                                            exclude_model_ids=[current_model] if current_model else None,
                                        )
                                    except (ValueError, Exception):
                                        pass  # Keep current model if no upgrade available
                        # Re-execute with upgraded models
                        from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                        executor2 = _TE(ctx.topology)
                        runner3 = TopologyRunner(
                            graph=ctx.topology, executor=executor2,
                            llm_provider=self.llm_provider, llm_config=self.llm_config,
                            provider_pool=self.provider_pool,
                        )
                        retry_result = await runner3.run(ctx.task)
                        if retry_result:
                            result = retry_result
                            log.info("Stage 4: FrugalGPT cascade succeeded on retry")
                    except Exception as exc:
                        log.debug("Stage 4: FrugalGPT cascade retry failed: %s", exc)

            ctx.result = result
            # Estimate cost from topology execution
            # Uses per-model pricing from cards.toml when available
            ctx.cost = self._estimate_topology_cost(ctx)
        except Exception as exc:
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
                except Exception as fallback_exc:
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
            except Exception:
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
            except Exception as exc:
                log.warning("PRM scoring failed in LEARN: %s", exc)

        # Only record to bandit when quality is known — never guess
        if quality is not None and self.bandit and hasattr(self.bandit, "record_outcome"):
            if ctx.bandit_decision_id:
                try:
                    self.bandit.record_outcome(ctx.bandit_decision_id, quality, ctx.cost, ctx.latency_ms)
                    log.debug("Bandit outcome recorded (in-memory, not persisted across restarts)")
                except Exception:
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
                    except Exception:
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
            except Exception as exc:
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
            except Exception:
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
            except Exception:
                pass  # Best-effort, never blocks pipeline
