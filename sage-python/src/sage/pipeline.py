"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Mapping

from sage.contracts.cost_tracker import CostTracker
from sage.events import (
    EXECUTE_BUDGET_EXCEEDED,  # noqa: F401 - re-exported for tests/test_pipeline_budget.py + pipeline_v2.memory_gate uses sage.events directly
    EXECUTE_HALTED_UNVERIFIED,  # noqa: F401 - imported by pipeline_v2.execute
    EXECUTE_UNVERIFIED,  # noqa: F401 - imported by pipeline_v2.execute
)

from sage.pipeline_stages import DAGFeatures
from sage.runtime.oracle import EvidenceRef, OracleConfig, OracleVerdict, oracle_enabled
from sage.runtime.run_frame import RunFrame, RunStatus

# OxiZ formal verification — imported lazily to allow graceful fallback.
# Annotated as `Any` so mypy does not infer the real Callable / type and
# then complain about the `None` sentinels in the ImportError branch.
verify_provider_assignment: Any = None
ProviderSpec: Any = None
_Z3_VERIFY_AVAILABLE = False
try:
    from sage.contracts import z3_verify as _z3_verify_mod
    verify_provider_assignment = _z3_verify_mod.verify_provider_assignment
    ProviderSpec = _z3_verify_mod.ProviderSpec
    _Z3_VERIFY_AVAILABLE = True
except ImportError:
    pass

log = logging.getLogger(__name__)

BUDGET_EXCEEDED_RESULT = "[sage: budget exceeded]"
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
BanditAttributionState = Literal["pending", "verified", "mismatch", "skipped"]
BanditAttributionReasonCode = Literal[
    "router_fallback_degraded",
    "model_mismatch",
    "template_mismatch",
    "multi_node_ambiguous",
    "decision_unknown",
    "recorder_instance_mismatch",
]
_BANDIT_ATTRIBUTION_REASON_CODES: tuple[BanditAttributionReasonCode, ...] = (
    "router_fallback_degraded",
    "model_mismatch",
    "template_mismatch",
    "multi_node_ambiguous",
    "decision_unknown",
    "recorder_instance_mismatch",
)

def _new_runtime_run_id() -> str:
    """Return a canonical 26-char uppercase ULID, with a local fallback."""
    try:
        import ulid

        return str(ulid.new()).upper()
    except Exception:  # noqa: BLE001 - tracing must not depend on ulid availability
        timestamp_ms = (time.time_ns() // 1_000_000) & ((1 << 48) - 1)
        value = (timestamp_ms << 80) | secrets.randbits(80)
        chars: list[str] = []
        for _ in range(26):
            chars.append(_ULID_ALPHABET[value & 0x1F])
            value >>= 5
        return "".join(reversed(chars))


def _is_strict_governance() -> bool:
    """Read the SAGE_STRICT_GOVERNANCE env var (A0b, 2026-04-23).

    When truthy, governance failures (write-gate init failure,
    verification-failed provider assignment) abort the pipeline
    instead of logging-and-continuing. Default off — the existing
    dev-friendly fail-open behaviour is preserved unless an operator
    explicitly opts in. Accepts ``1`` / ``true`` / ``yes`` / ``on``
    (case-insensitive) as truthy; everything else is off.
    """
    v = os.environ.get("SAGE_STRICT_GOVERNANCE", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _resolve_task_budget_usd(budget_usd: float | None) -> float:
    """Resolve task-level spend cap; 0 means unlimited."""
    raw_budget: float | str | None = budget_usd
    if raw_budget is None:
        env_budget = os.environ.get("SAGE_TASK_BUDGET_USD")
        if env_budget is None or not env_budget.strip():
            return 0.0
        raw_budget = env_budget
    try:
        return float(raw_budget)
    except (TypeError, ValueError):
        log.warning("Invalid SAGE_TASK_BUDGET_USD=%r; task budget disabled", raw_budget)
        return 0.0


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
    bandit_decision_id: str = ""
    bandit_model_id: str = ""
    bandit_template: str = ""
    bandit_context: list[float] = field(default_factory=list)
    executed_model_id: str = ""
    executed_template: str = ""
    executed_model_ids: list[str] = field(default_factory=list)
    bandit_attribution_state: BanditAttributionState = "skipped"
    verification_passed: bool = True
    axis_hint: str = ""  # MASBENCH axis hint for topology selection
    tool_call_count: int = 0
    tool_turn_count: int = 0
    executed_commands: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    cost_tracker: Any = None
    oracle_verdict: OracleVerdict | None = None
    bench_result: Mapping[str, Any] | None = None


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

    _model_catalog: Any = None

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
        semantic_memory: Any = None,
        memory_agent: Any = None,
        causal_memory: Any = None,
        tool_forge: Any = None,
        tool_registry: Any = None,
        harness_config: Any = None,
        agent_loop: Any = None,
        budget_usd: float | None = None,
        oracle_config: OracleConfig | None = None,
        llm_tier: str = "",
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
        # T2 phase 0/1 (cgpro 2026-04-29): forward the other 3 memory
        # backends to per-node agent loops so write-gate skips can target
        # real backends instead of "memory_backend_unwired".
        self.semantic_memory = semantic_memory
        self.memory_agent = memory_agent
        self.causal_memory = causal_memory
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
        self.budget_usd = _resolve_task_budget_usd(budget_usd)
        self._llm_tier = llm_tier
        self._oracle_config = oracle_config or OracleConfig()

        # G-series audit fix (2026-04-19 docs/audits/2026-04-18-astropy-14995-*):
        # RustCompositeWriteGate was built, exported, but never called at
        # runtime (investigation confirmed 0 runtime call sites). Memory
        # writes in phases/act.py and _record_to_memory here all skipped
        # the 5-signal salience check.
        #
        # Weights: w_confidence=0.0 because AgentLoop has no per-turn
        # confidence signal — redistributing that 0.25 to novelty (+0.10)
        # and relevance (+0.15) keeps the composite summing to 1.0 and
        # leans on signals that ARE available (task text + content text).
        # Not a heuristic tweak: an honest statement that this engine cannot
        # produce the "confidence" input the research paper assumed.
        #
        # Gate is REBUILT per-task in `run()` (not reset in-place) so the
        # Rust class — which has no `reset_task()` method yet — doesn't need
        # an ABI bump. `_gate_config` holds the construction args; `write_gate`
        # is swapped out per task.
        self._gate_config = dict(
            threshold=0.35,
            w_confidence=0.0,
            w_novelty=0.40,
            w_reliability=0.20,
            w_recency=0.10,
            w_relevance=0.30,
        )
        self.write_gate = self._build_write_gate()

    # ── Memory gate + budget emit (Phase 2.1 Step B1, 2026-05-06) ──────────
    # Bodies live in `sage.pipeline_v2.memory_gate`. Methods preserved as
    # delegators for mockability (~10 test files). LOCAL imports.

    def _emit_budget_exceeded(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.memory_gate import emit_budget_exceeded
        emit_budget_exceeded(self, ctx)

    def _build_write_gate(self) -> Any:
        from sage.pipeline_v2.memory_gate import build_write_gate
        return build_write_gate(self)

    def _record_to_memory(
        self,
        ctx: PipelineContext,
        *,
        is_training_evidence: bool | None = None,
    ) -> None:
        from sage.pipeline_v2.memory_gate import record_to_memory
        record_to_memory(self, ctx, is_training_evidence=is_training_evidence)

    def _emit(self, stage: str, data: dict) -> None:  # type: ignore[type-arg]
        """Emit a PIPELINE event on EventBus if available.

        Cycle-13 K Phase 2.1 Step A3 (2026-05-06): body moved to
        `sage.pipeline_v2.runtime_events.emit`. This is now a 1-line
        delegator. The method form is preserved so:
          - existing call sites in pipeline_v2/execute.py:{153,162,346}
            (`self._emit(stage, data)` where self is the pipeline)
            continue working unchanged
          - internal pipeline.py call sites continue byte-identical
        LOCAL import per cgpro DESIGN trap on circular-import risk.
        """
        from sage.pipeline_v2.runtime_events import emit as _v2_emit
        _v2_emit(self, stage, data)

    # ── Runtime-event helpers (Phase 2.1 Step B4, 2026-05-06) ──────────────
    # Bodies in `sage.pipeline_v2.runtime_events`. Methods preserved as
    # delegators for mockability. LOCAL imports.

    def _emit_bandit_attribution_mismatch(
        self,
        ctx: PipelineContext,
        reason_code: BanditAttributionReasonCode,
    ) -> None:
        from sage.pipeline_v2.runtime_events import emit_bandit_attribution_mismatch
        emit_bandit_attribution_mismatch(self, ctx, reason_code)

    @staticmethod
    def _bandit_reason_from_exception(exc: Exception) -> BanditAttributionReasonCode:
        from sage.pipeline_v2.runtime_events import bandit_reason_from_exception
        return bandit_reason_from_exception(exc)

    @staticmethod
    def _runtime_node_count(topology: Any) -> int:
        from sage.pipeline_v2.runtime_events import runtime_node_count
        return runtime_node_count(topology)

    @staticmethod
    def _runtime_edge_type(value: Any) -> str:
        from sage.pipeline_v2.runtime_events import runtime_edge_type
        return runtime_edge_type(value)

    def _runtime_edge_summary(self, topology: Any) -> tuple[int, list[dict[str, Any]]]:
        from sage.pipeline_v2.runtime_events import runtime_edge_summary
        return runtime_edge_summary(self, topology)

    def _runtime_node_summary(self, ctx: PipelineContext) -> list[dict[str, Any]]:
        from sage.pipeline_v2.runtime_events import runtime_node_summary
        return runtime_node_summary(self, ctx)

    def _runtime_provider_id_for_model(self, model_id: str, ctx: PipelineContext) -> str:
        from sage.pipeline_v2.runtime_events import runtime_provider_id_for_model
        return runtime_provider_id_for_model(self, model_id, ctx)

    @staticmethod
    def _runtime_node_capabilities(node: Any) -> tuple[str, ...]:
        from sage.pipeline_v2.runtime_events import runtime_node_capabilities
        return runtime_node_capabilities(node)

    def _runtime_graph_digest(
        self,
        *,
        nodes_summary: list[dict[str, Any]],
        edges_summary: list[dict[str, Any]],
    ) -> str:
        from sage.pipeline_v2.runtime_events import runtime_graph_digest
        return runtime_graph_digest(
            nodes_summary=nodes_summary, edges_summary=edges_summary
        )

    def _runtime_emit_topology_selected(
        self,
        ctx: PipelineContext,
        event_log: Any,
        run_frame_builder: Any | None = None,
        *,
        reason: str = "initial",
    ) -> None:
        from sage.pipeline_v2.runtime_events import runtime_emit_topology_selected
        runtime_emit_topology_selected(
            self, ctx, event_log, run_frame_builder, reason=reason
        )

    def _runtime_emit_model_assigned(
        self,
        ctx: PipelineContext,
        event_log: Any,
        run_frame_builder: Any | None = None,
    ) -> None:
        from sage.pipeline_v2.runtime_events import runtime_emit_model_assigned
        runtime_emit_model_assigned(self, ctx, event_log, run_frame_builder)

    def _runtime_final_status(self, ctx: PipelineContext | None) -> RunStatus:
        from sage.pipeline_v2.runtime_events import runtime_final_status
        return runtime_final_status(self, ctx)

    def _runtime_final_node_count(self, ctx: PipelineContext | None) -> int:
        from sage.pipeline_v2.runtime_events import runtime_final_node_count
        return runtime_final_node_count(self, ctx)

    async def run(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> str:
        """Execute the full 5-stage pipeline and return only the output string."""
        result, _frame = await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
        )
        return result

    async def run_with_frame(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> tuple[str, RunFrame]:
        """Like run() but returns (output, frozen RunFrame).

        Signature mirrors run() so bench/traced adapters can use either
        entry point without parameter loss (cgpro 2026-04-29 cycle 4
        reassess R7.0.2).
        """
        return await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
        )

    async def run_with_bench_evaluator(
        self,
        task: str,
        evaluator: "Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]",
        *,
        budget_usd: float | None = None,
        system_hint: int | None = None,
    ) -> tuple[str, RunFrame]:
        """Run the pipeline with a synchronous-eval bench evaluator wired in.

        cgpro 2026-04-29 R6.1a verify Path E: synchronous-eval benches
        (BigCodeBench, EvalPlus, HumanEval) need their pass/fail to be
        available to the OracleStack BEFORE final_result + oracle_verdict
        + Stage 5 learning fire. Without this seam, those adapters call
        ``system.run()``, get the output, and only then evaluate — but by
        then the live oracle has already abstained because ``ctx.bench_result``
        was never populated.

        Locked event order::

            Stage 0-4 execute  →  evaluator(final_output)  →
            final_result  →  oracle_verdict  →  Stage 5 learn  →
            run_frame_summary

        The evaluator MUST return a Mapping with at least ``{"passed": bool}``;
        ``score``, ``reason``, ``output_sha256``, ``tool_call_id``,
        ``verifier_id`` are accepted by ``_exact_oracle``. If the evaluator
        raises or returns an invalid shape, ``ctx.bench_result`` stays None
        and the oracle abstains as if no evaluator were attached
        (fail-closed by design via ``_exact_oracle`` itself, which returns
        None on missing/malformed input).

        Sync and async evaluators are both supported: if ``evaluator(...)``
        returns an awaitable, it is awaited.
        """
        return await self._run_internal(
            task,
            budget_usd=budget_usd,
            system_hint=system_hint,
            emit_run_frame_summary=os.environ.get("SAGE_RUN_FRAME") == "1",
            bench_evaluator=evaluator,
        )

    async def _run_internal(
        self,
        task: str,
        budget_usd: float | None = None,
        system_hint: int | None = None,
        *,
        emit_run_frame_summary: bool = False,
        bench_evaluator: (
            "Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]] | None"
        ) = None,
    ) -> tuple[str, RunFrame]:
        """Execute the full 5-stage pipeline.

        Args:
            task: The user's task.
            budget_usd: Task-level spend cap for the run. ``None`` uses the
                constructor/env value; ``0`` means unlimited.
            system_hint: Optional override for Stage 0 routing (1, 2, or 3).
                Benchmark adapters use this when they already know the task
                complexity (e.g. SWE-bench tasks are always S3). When set,
                the Rust SystemRouter still runs (so we keep the model
                assignment + bandit posteriors), but `ctx.system` is forced
                to the hint afterwards.
        """
        from sage.observability.spans import sage_span
        from sage.runtime.event_log import (
            EventLogUnavailable,
            RuntimeEventLog,
            current_event_log,
            install_event_log,
        )
        from sage.runtime.event_log.redaction import _hash_text
        from sage.runtime.run_frame.builder import _RunFrameBuilder

        # Cycle-13 E Tier 2.1 smoke discovery 2026-05-05: when called via
        # `sage run --jsonl` (cycle-12 prelude `d09bed4d`), the CLI installs
        # its own RuntimeEventLog with a stdout-mirror tee BEFORE calling
        # pipeline.run(). The previous unconditional construction here
        # shadowed the CLI's eventlog with a fresh-disabled one (no
        # trace_dir kwarg + SAGE_TRACE_JSONL_DIR env unset =>
        # writer.py:162 sets disabled=True, all emit_* become no-ops),
        # so no runtime events ever reached the CLI's stdout. Prefer
        # the externally-installed eventlog when present; fall back to
        # creating a fresh one for direct-Python callers (the
        # historical default).
        event_log = current_event_log()
        if event_log is None:
            event_log = RuntimeEventLog(run_id=_new_runtime_run_id())
        run_frame_builder = _RunFrameBuilder(
            run_id=event_log.run_id,
            task_id=event_log.run_id,
            task_hash=_hash_text(task),
        )
        run_frame_builder.capture_feature_flags()
        token = install_event_log(event_log)
        final_emitted = False
        ctx: PipelineContext | None = None
        t0 = time.monotonic()
        _span_attrs: dict[str, Any] = {"gen_ai.request.model": ""}
        try:
            with sage_span("sage.pipeline.run", op="invoke_agent", **_span_attrs):
                effective_budget_usd = (
                    self.budget_usd
                    if budget_usd is None
                    else _resolve_task_budget_usd(budget_usd)
                )
                ctx = PipelineContext(task=task, budget=effective_budget_usd)
                if effective_budget_usd > 0:
                    ctx.cost_tracker = CostTracker(budget_usd=effective_budget_usd)

                self._last_routing_decision = None
                self._last_runtime_routing_source = "default"
                self._last_runtime_routing_confidence = None
                self._last_runtime_routing_model_id = ""
                event_log.emit_task_started(ctx.task)

                # G-series (2026-04-19): rebuild write gate per task so entries from a
                # previous task don't persist as novelty penalties or exact-dedup hits
                # on content in THIS task. Rust gate has no in-place reset yet.
                self.write_gate = self._build_write_gate()

                # Stage 0: CLASSIFY
                ctx = self._stage_classify(ctx)
                if system_hint in (1, 2, 3) and ctx.system != system_hint:
                    log.info(
                        "Stage 0: system_hint=S%d overrides router S%d",
                        system_hint, ctx.system,
                    )
                    ctx.system = system_hint
                routing_source = getattr(self, "_last_runtime_routing_source", "default")
                routing_confidence = getattr(self, "_last_runtime_routing_confidence", None)
                routing_model_id = getattr(self, "_last_runtime_routing_model_id", "")
                routing_seq = event_log.emit_routing_decision(
                    routing_source=routing_source,
                    system=ctx.system,
                    domain=ctx.domain,
                    confidence=routing_confidence,
                    model_id=routing_model_id,
                )
                run_frame_builder.record_routing_decision(
                    seq=routing_seq,
                    routing_source=routing_source,
                    system=ctx.system,
                    domain=ctx.domain,
                    confidence=routing_confidence,
                    model_id=routing_model_id,
                )
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
                self._runtime_emit_topology_selected(
                    ctx,
                    event_log,
                    run_frame_builder,
                    reason="initial",
                )
                self._emit("SELECT_TOPOLOGY", {"node_count": topo_nodes})

                # Stage 3: ASSIGN MODELS
                ctx = self._stage_assign_models(ctx)
                self._runtime_emit_model_assigned(ctx, event_log, run_frame_builder)
                self._emit(
                    "ASSIGN_MODELS", {"assignments": ctx.assignments, "domain": ctx.domain}
                )

                # Stage 4: EXECUTE
                ctx = await self._stage_execute(
                    ctx,
                    event_log=event_log,
                    run_frame_builder=run_frame_builder,
                )
                ctx.latency_ms = (time.monotonic() - t0) * 1000

                # cgpro 2026-04-29 R6.1a verify Path E: bench-result feedback
                # seam. Synchronous-eval benches (BigCodeBench, EvalPlus, etc.)
                # attach an evaluator via run_with_bench_evaluator(); we call
                # it on the executed output BEFORE final_result + oracle so
                # _exact_oracle has bench_result["passed"] available. Fail-
                # closed: any exception leaves ctx.bench_result=None and the
                # oracle abstains via _exact_oracle's None-guard.
                if bench_evaluator is not None and ctx.bench_result is None:
                    try:
                        candidate = bench_evaluator(ctx.result or "")
                        import inspect as _inspect
                        if _inspect.isawaitable(candidate):
                            candidate = await candidate
                        if isinstance(candidate, Mapping):
                            ctx.bench_result = candidate
                        else:
                            log.warning(
                                "bench_evaluator returned %r; expected Mapping. "
                                "Oracle will abstain.",
                                type(candidate).__name__,
                            )
                    except Exception as _eval_exc:  # noqa: BLE001 - fail-closed
                        log.warning(
                            "bench_evaluator raised %s: %s; oracle will abstain.",
                            type(_eval_exc).__name__,
                            _eval_exc,
                        )

                oracle_on = oracle_enabled()
                final_status = self._runtime_final_status(ctx)

                if oracle_on:
                    final_seq = event_log.emit_final_result(
                        status=final_status,
                        output=ctx.result or "",
                        total_cost_usd=float(ctx.cost or 0.0),
                        total_latency_ms=ctx.latency_ms,
                        node_count=self._runtime_final_node_count(ctx),
                    )
                    run_frame_builder.record_final_result(
                        seq=final_seq,
                        status=final_status,
                    )
                    final_emitted = True
                    try:
                        from sage.runtime import oracle as oracle_stack

                        verdict = oracle_stack.evaluate(
                            run_frame_builder.snapshot_view(),
                            final_output=ctx.result or "",
                            bench_result=ctx.bench_result,
                            config=self._oracle_config,
                        )
                    except Exception as exc:  # noqa: BLE001 - oracle must fail closed
                        log.warning("OracleStack failed; collapsing to Abstain: %s", exc)
                        verdict = OracleVerdict(
                            trainable=False,
                            verdict_source="abstain",
                            quality_label="unknown",
                            score=None,
                            confidence=1.0,
                            reason_codes=("oracle_exception", type(exc).__name__),
                            evidence=(EvidenceRef(run_id=event_log.run_id),),
                        )
                    oracle_seq = event_log.emit_oracle_verdict(
                        parent_event_id=final_seq,
                        verdict=verdict,
                    )
                    run_frame_builder.record_oracle_verdict(
                        seq=oracle_seq,
                        verdict=verdict,
                    )
                    ctx.oracle_verdict = verdict

                    self._record_to_memory(
                        ctx,
                        is_training_evidence=verdict.trainable,
                    )
                    await self._stage_learn(ctx)
                    self._emit("LEARN", {"latency_ms": ctx.latency_ms})
                else:
                    # Legacy OFF mode: keep the R7 execution/learn/final order.
                    self._record_to_memory(ctx)
                    await self._stage_learn(ctx)
                    self._emit("LEARN", {"latency_ms": ctx.latency_ms})

                    # Expose full context before final_result, preserving R7 order.
                    self.last_context = ctx

                    if self._agent_loop is not None and ctx.cost:
                        self._agent_loop.total_cost_usd = float(ctx.cost)

                    final_seq = event_log.emit_final_result(
                        status=final_status,
                        output=ctx.result or "",
                        total_cost_usd=float(ctx.cost or 0.0),
                        total_latency_ms=ctx.latency_ms,
                        node_count=self._runtime_final_node_count(ctx),
                    )
                    run_frame_builder.record_final_result(
                        seq=final_seq,
                        status=final_status,
                    )
                    final_emitted = True

                if oracle_on:
                    self.last_context = ctx
                    if self._agent_loop is not None and ctx.cost:
                        self._agent_loop.total_cost_usd = float(ctx.cost)

                frame = run_frame_builder.finalize()
                if emit_run_frame_summary and frame.final_result_seq is not None:
                    try:
                        event_log.emit_run_frame_summary(
                            parent_event_id=frame.final_result_seq,
                            summary=frame.to_summary_dict(redacted=True),
                        )
                    except (EventLogUnavailable, OSError, IOError, ValueError):
                        pass
                return ctx.result, frame
        except Exception:
            if not final_emitted:
                latency_ms = (time.monotonic() - t0) * 1000
                if ctx is not None:
                    ctx.latency_ms = latency_ms
                final_seq = event_log.emit_final_result(
                    status="failure",
                    output=(ctx.result if ctx is not None else "") or "",
                    total_cost_usd=float((ctx.cost if ctx is not None else 0.0) or 0.0),
                    total_latency_ms=latency_ms,
                    node_count=self._runtime_final_node_count(ctx),
                )
                run_frame_builder.record_final_result(seq=final_seq, status="failure")
            raise
        finally:
            token.var.reset(token)
            event_log.close()

    # ── Stage 0: Classify ───────────────────────────────────────────────────

    def _stage_classify(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 0: Classify task complexity, domain, and select integrated routing.

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.classify.classify`. This is now a 1-line
        delegator. LOCAL import per cgpro DESIGN trap #4.
        """
        from sage.pipeline_v2.classify import classify as _v2_classify
        return _v2_classify(self, ctx)

    # ── Stage 1: Decompose ──────────────────────────────────────────────────

    async def _stage_decompose(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Decompose task into sub-tasks (S2/S3 only).

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.decompose.decompose`. This is now a 1-line
        delegator. LOCAL import (NOT top-level) to avoid the
        circular-init partial-load risk: `pipeline.py` is loaded
        before `pipeline_v2/` in many call paths. See
        `cgpro_pi_mono_pivot_20260505` DESIGN lock trap #4.
        """
        from sage.pipeline_v2.decompose import decompose as _v2_decompose
        return await _v2_decompose(self, ctx)

    # ── Structure-driven topology selection ──────────────────────────────

    def _build_topology_from_hint(self, hint: str) -> Any | None:
        """Create a topology from a template hint using Rust TemplateStore.

        Cycle-13 K Phase 2.1 Step A1 (2026-05-06): body moved to
        `sage.pipeline_v2.topology_helpers.build_topology_from_hint`.
        This is now a 1-line delegator. The method form is preserved
        on the class so existing callers (`pipeline_v2/select_topology.py`
        invoking `self._build_topology_from_hint(...)`) continue to
        function AND so the ~6 test files that mock this method via
        `pipeline._build_topology_from_hint = MagicMock(...)` keep
        working byte-identical. LOCAL import per cgpro DESIGN trap
        on circular-import risk in pipeline_v2 package.
        """
        from sage.pipeline_v2.topology_helpers import build_topology_from_hint
        return build_topology_from_hint(hint)

    # ── Stage 2: Select Topology ────────────────────────────────────────────

    def _stage_select_topology(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 2: Select optimal topology.

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.select_topology.select_topology`. This is now a
        1-line LOCAL-import delegator. Topology helper methods stay on
        this class per cgpro DESIGN (helper ownership migration is Phase C).
        """
        from sage.pipeline_v2.select_topology import select_topology as _v2_select_topology
        return _v2_select_topology(self, ctx)

    # ── Topology helpers (Phase 2.1 Step B5, 2026-05-06) ───────────────────
    # Bodies in `sage.pipeline_v2.topology_helpers`. Methods preserved as
    # delegators for mockability. LOCAL imports.

    def _topology_candidate_items(self, result: Any) -> list[Any]:
        from sage.pipeline_v2.topology_helpers import topology_candidate_items
        return topology_candidate_items(self, result)

    def _log_topology_candidates(self, candidates: list[Any]) -> None:
        from sage.pipeline_v2.topology_helpers import log_topology_candidates
        log_topology_candidates(self, candidates)

    @staticmethod
    def _candidate_text_attr(
        obj: Any,
        names: tuple[str, ...],
        default: str,
    ) -> str:
        from sage.pipeline_v2.topology_helpers import candidate_text_attr
        return candidate_text_attr(obj, names, default)

    @staticmethod
    def _candidate_float_attr(
        obj: Any,
        names: tuple[str, ...],
        default: float,
    ) -> float:
        from sage.pipeline_v2.topology_helpers import candidate_float_attr
        return candidate_float_attr(obj, names, default)

    @staticmethod
    def _candidate_node_count(obj: Any) -> int:
        from sage.pipeline_v2.topology_helpers import candidate_node_count
        return candidate_node_count(obj)

    def _log_topology_structure(
        self,
        topology: Any,
        source: str,
        confidence: float | None,
    ) -> None:
        from sage.pipeline_v2.topology_helpers import log_topology_structure
        log_topology_structure(self, topology, source, confidence)

    def _apply_topology_budget_and_cache(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.topology_helpers import apply_topology_budget_and_cache
        apply_topology_budget_and_cache(self, ctx)

    def _check_topology_budget(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.topology_helpers import check_topology_budget
        check_topology_budget(self, ctx)

    def _make_single_node_topology(self, ctx: PipelineContext) -> Any:
        from sage.pipeline_v2.topology_helpers import make_single_node_topology
        return make_single_node_topology(self, ctx)

    # ── Costing helpers (Phase 2.1 Step B5, 2026-05-06) ────────────────────
    # Bodies in `sage.pipeline_v2.costing` — costing-transverse, NOT
    # assign-side per cgpro Q3 garde-fou.

    def _estimate_topology_cost(self, ctx: PipelineContext) -> float:
        from sage.pipeline_v2.costing import estimate_topology_cost
        return estimate_topology_cost(self, ctx)

    def _load_model_catalog(self) -> Any:
        from sage.pipeline_v2.costing import load_model_catalog
        return load_model_catalog(self)

    # ── Stage 3: Assign Models ──────────────────────────────────────────────

    def _stage_assign_models(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 3: Assign model_id to each topology node.

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.assign_models.assign_models`. 1-line LOCAL-import
        delegator. Helpers `_log_model_assigner_chosen_fallback` and
        `_verify_assignment_formal` STAY on this class per cgpro DESIGN
        trap #6 (helper ownership migration is Phase C).
        """
        from sage.pipeline_v2.assign_models import assign_models as _v2_assign_models
        return _v2_assign_models(self, ctx)

    # ── Assign-side helpers (Phase 2.1 Step B2, 2026-05-06) ────────────────
    # Bodies live in `sage.pipeline_v2.assign_models`. Methods preserved as
    # delegators for mockability. LOCAL imports.

    def _log_model_assigner_chosen_fallback(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.assign_models import log_model_assigner_chosen_fallback
        log_model_assigner_chosen_fallback(self, ctx)

    def _verify_assignment_formal(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.assign_models import verify_assignment_formal
        verify_assignment_formal(self, ctx)

    # ── Stage 4: Execute ────────────────────────────────────────────────────
    #
    # Cycle-13 K Phase 2.1 Step A2 (2026-05-06): the 5 bandit-attribution
    # lifecycle helpers below now delegate their bodies to
    # `sage.pipeline_v2.bandit_attribution`. Method form preserved so:
    #   - existing call sites in pipeline_v2/{learn,execute,classify}.py
    #     calling `self._<method>(ctx)` continue working unchanged
    #   - test mocks (`pipeline._record_bandit_outcome_checked = MagicMock(...)`,
    #     direct calls `pipeline._is_single_agent_execution(ctx)` in
    #     test_pipeline_topology_skip_guardrails_decoupling, etc.) keep
    #     working byte-identical
    # LOCAL imports per cgpro DESIGN trap on circular-import risk.

    def _bandit_task_context(self, ctx: PipelineContext) -> list[float]:
        from sage.pipeline_v2.bandit_attribution import bandit_task_context
        return bandit_task_context(self, ctx)

    def _is_single_agent_execution(self, ctx: PipelineContext) -> bool:
        from sage.pipeline_v2.bandit_attribution import is_single_agent_execution
        return is_single_agent_execution(self, ctx)

    def _clear_bandit_decision(self, ctx: PipelineContext) -> None:
        from sage.pipeline_v2.bandit_attribution import clear_bandit_decision
        clear_bandit_decision(self, ctx)

    def _cancel_bandit_decision(self, ctx: PipelineContext, *, force: bool = False) -> bool:
        from sage.pipeline_v2.bandit_attribution import cancel_bandit_decision
        return cancel_bandit_decision(self, ctx, force=force)

    def _record_bandit_outcome_checked(self, ctx: PipelineContext, quality: float) -> None:
        from sage.pipeline_v2.bandit_attribution import record_bandit_outcome_checked
        record_bandit_outcome_checked(self, ctx, quality)

    def _pick_fallback_provider(self):
        """Return (provider, config) for a healthy fallback.

        Cycle-13 K Phase 2.1 Step B3 (2026-05-06): body moved to
        `sage.pipeline_v2.execute.pick_fallback_provider`. Method
        preserved as 1-line LOCAL-import delegator so the 7 tests
        in test_pipeline_fallback_provider.py + the call site in
        pipeline_v2/execute.py:505 continue working byte-identical.
        """
        from sage.pipeline_v2.execute import pick_fallback_provider
        return pick_fallback_provider(self)

    async def _stage_execute(
        self,
        ctx: PipelineContext,
        event_log: Any | None = None,
        run_frame_builder: Any | None = None,
    ) -> PipelineContext:
        """Stage 4 execution moved to ``sage.pipeline_v2.execute``."""
        # Fix C moved with the body:
        # _effective_controller = (
        #     None if self._llm_tier == "budget" else self.controller
        # )
        from sage.pipeline_v2.execute import execute as _v2_execute

        return await _v2_execute(
            self,
            ctx,
            event_log=event_log,
            run_frame_builder=run_frame_builder,
        )

    # ── Stage 5: Learn ──────────────────────────────────────────────────────

    async def _stage_learn(self, ctx: PipelineContext) -> None:
        """Stage 5: Record outcome for learning.

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.learn.learn`. This is now a 1-line
        LOCAL-import async delegator. Learning helper methods stay on
        this class per cgpro DESIGN (helper ownership migration is Phase C).
        """
        from sage.pipeline_v2.learn import learn as _v2_learn
        return await _v2_learn(self, ctx)
