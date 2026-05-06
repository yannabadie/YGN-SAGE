"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import hashlib
import json
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

    def _emit_bandit_attribution_mismatch(
        self,
        ctx: PipelineContext,
        reason_code: BanditAttributionReasonCode,
    ) -> None:
        payload = {
            "decision_id": str(getattr(ctx, "bandit_decision_id", "") or ""),
            "selected_model_id": str(getattr(ctx, "bandit_model_id", "") or ""),
            "selected_template": str(getattr(ctx, "bandit_template", "") or ""),
            "executed_model_id": str(getattr(ctx, "executed_model_id", "") or ""),
            "executed_template": str(getattr(ctx, "executed_template", "") or ""),
            "reason_code": reason_code,
        }
        try:
            from sage.runtime.event_log import current_event_log
            from sage.runtime.event_log.events import _EventCore

            event_log = current_event_log()
            if event_log is not None:
                event_log._emit(  # noqa: SLF001 - no public generic emit exists for new event types.
                    _EventCore,
                    "bandit_attribution_mismatch",
                    "pipeline",
                    payload=payload,
                    _force_payload=True,
                )
        except Exception:  # noqa: BLE001 - telemetry must not mask execution or learning.
            pass
        self._emit("BANDIT_ATTRIBUTION_MISMATCH", payload)

    @staticmethod
    def _bandit_reason_from_exception(exc: Exception) -> BanditAttributionReasonCode:
        text = str(exc).lower()
        for reason in _BANDIT_ATTRIBUTION_REASON_CODES:
            if reason in text:
                return reason
        if "unknown" in text and "decision" in text:
            return "decision_unknown"
        if "template" in text and "mismatch" in text:
            return "template_mismatch"
        if "model" in text and "mismatch" in text:
            return "model_mismatch"
        return "recorder_instance_mismatch"

    @staticmethod
    def _runtime_node_count(topology: Any) -> int:
        if topology is None or not hasattr(topology, "node_count"):
            return 0
        try:
            node_count = topology.node_count()
            return int(node_count() if callable(node_count) else node_count)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0

    @staticmethod
    def _runtime_edge_type(value: Any) -> str:
        if value == 0:
            return "control"
        if value == 1:
            return "message"
        if value == 2:
            return "state"
        return str(value or "")

    def _runtime_edge_summary(self, topology: Any) -> tuple[int, list[dict[str, Any]]]:
        if topology is None:
            return 0, []
        try:
            if hasattr(topology, "get_edges"):
                raw_edges = list(topology.get_edges() or [])
                summaries: list[dict[str, Any]] = []
                for idx, edge in enumerate(raw_edges):
                    source_id = edge[0] if len(edge) > 0 else ""
                    target_id = edge[1] if len(edge) > 1 else ""
                    edge_type = self._runtime_edge_type(edge[2] if len(edge) > 2 else "")
                    summaries.append(
                        {
                            "edge_id": f"{source_id}->{target_id}:{idx}",
                            "source_id": str(source_id),
                            "target_id": str(target_id),
                            "edge_type": edge_type,
                            "channel": edge_type,
                        }
                    )
                return len(raw_edges), summaries
            if hasattr(topology, "edge_count"):
                edge_count = topology.edge_count()
                return int(edge_count() if callable(edge_count) else edge_count), []
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0, []
        return 0, []

    def _runtime_node_summary(self, ctx: PipelineContext) -> list[dict[str, Any]]:
        topology = ctx.topology
        summaries: list[dict[str, Any]] = []
        for idx in range(self._runtime_node_count(topology)):
            try:
                node = topology.get_node(idx)
            except (AttributeError, RuntimeError, TypeError):
                continue
            summaries.append(
                {
                    "node_id": str(idx),
                    "node_role": getattr(node, "role", "") or f"node-{idx}",
                    "node_type": getattr(node, "node_type", "") or "",
                    "model_id": getattr(node, "model_id", "") or "",
                    "provider_hint": ctx.provider_hints.get(idx, ""),
                }
            )
        return summaries

    def _runtime_provider_id_for_model(self, model_id: str, ctx: PipelineContext) -> str:
        for node_idx, assigned_model in ctx.assignments.items():
            if assigned_model == model_id and node_idx in ctx.provider_hints:
                return str(ctx.provider_hints[node_idx])
        if self.provider_pool is not None and hasattr(self.provider_pool, "infer_provider"):
            try:
                provider_id = self.provider_pool.infer_provider(model_id)
                if provider_id:
                    return str(provider_id)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        return str(getattr(self.llm_config, "provider", "") if self.llm_config else "")

    @staticmethod
    def _runtime_node_capabilities(node: Any) -> tuple[str, ...]:
        for attr in ("required_capabilities", "capabilities_required", "capabilities"):
            raw = getattr(node, attr, None)
            if raw:
                return tuple(str(item) for item in raw)
        return ()

    def _runtime_graph_digest(
        self,
        *,
        nodes_summary: list[dict[str, Any]],
        edges_summary: list[dict[str, Any]],
    ) -> str:
        canonical = json.dumps(
            {"nodes": nodes_summary, "edges": edges_summary},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _runtime_emit_topology_selected(
        self,
        ctx: PipelineContext,
        event_log: Any,
        run_frame_builder: Any | None = None,
        *,
        reason: str = "initial",
    ) -> None:
        if ctx.topology is None:
            return
        edge_count, edges_summary = self._runtime_edge_summary(ctx.topology)
        nodes_summary = self._runtime_node_summary(ctx)
        topology_id = ctx.topology_id or getattr(ctx.topology, "id", "") or ""
        seq = None
        if event_log is not None:
            seq = event_log.emit_topology_selected(
                topology_id=topology_id,
                template_type=getattr(ctx.topology, "template_type", "") or "",
                node_count=self._runtime_node_count(ctx.topology),
                edge_count=edge_count,
                nodes_summary=nodes_summary,
                edges_summary=edges_summary,
            )
        if run_frame_builder is not None:
            run_frame_builder.record_topology_selected(
                seq=seq,
                topology_id=topology_id,
                graph_digest=self._runtime_graph_digest(
                    nodes_summary=nodes_summary,
                    edges_summary=edges_summary,
                ),
                reason=reason,
            )

    def _runtime_emit_model_assigned(
        self,
        ctx: PipelineContext,
        event_log: Any,
        run_frame_builder: Any | None = None,
    ) -> None:
        if ctx.topology is None:
            return
        for idx in range(self._runtime_node_count(ctx.topology)):
            try:
                node = ctx.topology.get_node(idx)
            except (AttributeError, RuntimeError, TypeError):
                continue
            model_id = ctx.assignments.get(idx, getattr(node, "model_id", "") or "")
            node_role = getattr(node, "role", "") or f"node-{idx}"
            provider_id = self._runtime_provider_id_for_model(model_id, ctx)
            capabilities = self._runtime_node_capabilities(node)
            seq = None
            if event_log is not None:
                seq = event_log.emit_model_assigned(
                    node_id=str(idx),
                    node_role=node_role,
                    model_id=model_id,
                    provider_id=provider_id,
                    required_capabilities=capabilities,
                )
            if run_frame_builder is not None:
                run_frame_builder.record_model_assigned(
                    seq=seq,
                    node_id=str(idx),
                    node_role=node_role,
                    model_id=model_id,
                    provider_id=provider_id,
                    required_capabilities=capabilities,
                )

    def _runtime_final_status(self, ctx: PipelineContext | None) -> RunStatus:
        if ctx is None:
            return "failure"
        if ctx.result == BUDGET_EXCEEDED_RESULT:
            return "budget_exceeded"
        return "success" if ctx.result else "failure"

    def _runtime_final_node_count(self, ctx: PipelineContext | None) -> int:
        if ctx is None or ctx.topology is None:
            return 0
        return self._runtime_node_count(ctx.topology)

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

    def _topology_candidate_items(self, result: Any) -> list[Any]:
        if result is None:
            return []
        if isinstance(result, (list, tuple)):
            return list(result)
        candidates_attr = getattr(result, "candidates", None)
        if callable(candidates_attr):
            try:
                candidates_attr = candidates_attr()
            except Exception:
                candidates_attr = None
        if candidates_attr is not None:
            try:
                return list(candidates_attr)
            except TypeError:
                pass
        return [result]

    def _log_topology_candidates(self, candidates: list[Any]) -> None:
        for path, candidate in enumerate(candidates, start=1):
            source = self._candidate_text_attr(candidate, ("source",), "unknown")
            score = self._candidate_float_attr(
                candidate,
                ("score", "confidence", "quality"),
                0.0,
            )
            topology = getattr(candidate, "topology", None)
            template = self._candidate_text_attr(
                topology if topology is not None else candidate,
                ("template_type", "template"),
                "unknown",
            )
            nodes = self._candidate_node_count(topology if topology is not None else candidate)
            log.info(
                "topology.candidate path=%d source=%s archive_hit=%s "
                "score=%.3f template_type=%s nodes=%d",
                path,
                source,
                "true" if source in ("archive", "archive_hit") else "false",
                score,
                template,
                nodes,
            )

    @staticmethod
    def _candidate_text_attr(
        obj: Any,
        names: tuple[str, ...],
        default: str,
    ) -> str:
        if obj is None:
            return default
        for name in names:
            value = getattr(obj, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if value is not None:
                text = str(value)
                if text:
                    return text
        return default

    @staticmethod
    def _candidate_float_attr(
        obj: Any,
        names: tuple[str, ...],
        default: float,
    ) -> float:
        if obj is None:
            return default
        for name in names:
            value = getattr(obj, name, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return default

    @staticmethod
    def _candidate_node_count(obj: Any) -> int:
        if obj is None:
            return 0
        try:
            node_count = getattr(obj, "node_count", None)
            if callable(node_count):
                return int(node_count())
            if node_count is not None:
                return int(node_count)
        except Exception:
            return 0
        return 0

    def _log_topology_structure(
        self,
        topology: Any,
        source: str,
        confidence: float | None,
    ) -> None:
        """Gap 1+2 (2026-04-21): emit two INFO lines describing the DAG shape.

        Called from each Stage-2 branch right before the final cache step so
        every selected topology gets attribution regardless of which of the
        6 paths (smmu_hit / archive_hit / llm_synthesis / mutation /
        mcts_search / template_fallback) or fallback branches produced it.

        topology.edges — adjacency list. Truncated at 20 tuples; when the
        graph has >20 edges, include `total=N` so readers can tell the
        truncation happened. When the graph exposes no `get_edges`, we
        emit count-only so post-run analysis still sees the structure.

        topology.source — 6-path attribution with confidence. "dag_template"
        and "template_fallback" are Python-side branches (no engine call);
        all Rust-side sources are the canonical string from
        PyGenerateResult.source() (sage-core/src/topology/pyo3_wrappers.rs).
        """
        if topology is None:
            return

        nodes = 0
        try:
            if hasattr(topology, "node_count"):
                nc = topology.node_count()
                nodes = nc() if callable(nc) else int(nc)
        except Exception:
            nodes = 0

        template = getattr(topology, "template_type", None) or "unknown"
        topo_id = getattr(topology, "id", "") or ""

        # --- edges line ---
        edges_render: str = "[]"
        total_edges = 0
        truncated = False
        try:
            if hasattr(topology, "get_edges"):
                raw_edges = topology.get_edges()
                edges_iter = list(raw_edges) if raw_edges is not None else []
                total_edges = len(edges_iter)
                # Keep only (from, to) tuples; flow_type (3rd field) omitted
                # to keep the line short. Flow type is dominated by "control"
                # for DAG templates and not load-bearing for grep.
                pairs = [(int(e[0]), int(e[1])) for e in edges_iter[:20]]
                edges_render = repr(pairs)
                if total_edges > 20:
                    truncated = True
            elif hasattr(topology, "edge_count"):
                ec = topology.edge_count()
                total_edges = ec() if callable(ec) else int(ec)
                edges_render = "<count-only>"
        except Exception:
            edges_render = "<unreadable>"

        if truncated:
            log.info(
                "topology.edges nodes=%d template=%s id=%s edges=%s total=%d",
                nodes, template, (topo_id[:8] if topo_id else "none"),
                edges_render, total_edges,
            )
        else:
            log.info(
                "topology.edges nodes=%d template=%s id=%s edges=%s",
                nodes, template, (topo_id[:8] if topo_id else "none"),
                edges_render,
            )

        # --- source line ---
        conf_str = (
            f"{float(confidence):.3f}"
            if confidence is not None
            else "n/a"
        )
        # archive_hit flag (boolean) distinguishes the fast archive path from
        # every other 6-path source — useful for MAP-Elites growth attribution.
        archive_hit = (source == "archive_hit")
        log.info(
            "topology.source source=%s confidence=%s archive_hit=%s template=%s id=%s",
            source, conf_str, "true" if archive_hit else "false",
            template, (topo_id[:8] if topo_id else "none"),
        )

    def _apply_topology_budget_and_cache(self, ctx: PipelineContext) -> None:
        """Plan item 1.4a (2026-04-20): apply budget check + cache the final topology.

        _check_topology_budget may replace ctx.topology with a degraded
        single-node fallback; we cache AFTER that replacement so the id
        stored in topology_cache matches whatever record_outcome will
        reference in Stage 5. Before this helper existed, cache_topology
        was only wired on the engine branch (H4, commit dc51976), leaving
        three production paths silently uncached:
          - template branch (line ~502, dominant production path)
          - engine-branch budget degrade (_make_single_node_topology)
          - fallback TopologyGraph + TopologyNode path
        Empirically verified by plan-1.4 smoke: template branch → 0 cells
        after 10 pipeline.run() calls; with this helper → archive grows.
        """
        self._check_topology_budget(ctx)
        if (ctx.topology is not None
                and self.engine is not None
                and hasattr(self.engine, "cache_topology")):
            try:
                self.engine.cache_topology(ctx.topology)
            except (RuntimeError, TypeError) as exc:
                log.debug("cache_topology failed: %s", exc)

    def _check_topology_budget(self, ctx: PipelineContext) -> None:
        """Pre-validate budget feasibility — degrade to single-node if over budget."""
        if ctx.budget <= 0:
            return
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
        """PREDICTIVE cost estimate (pre-execution budget check only).

        P1.6 clarification (2026-04-22 audit remediation): this function is
        a PREDICTION used by the topology-budget gate BEFORE execution. It
        is NOT the actual cost tracker. The real runtime cost comes from
        `AgentLoop.total_cost_usd`, which is computed by
        `sage.phases.think._extract_step_cost` using the provider-reported
        token usage (`response.usage.prompt_tokens` +
        `response.usage.completion_tokens`) times the per-model rate in
        cards.toml. That path is the truth; ctx.cost gets updated from it
        post-execution (see lines 1270, 1353, 1398 in this file).

        This predictor uses a fixed ~500 input / ~300 output token budget
        per node — a known imprecise heuristic acceptable for a pre-run
        budget-gate. The audit (AUDIT4 bug #1 "cost estimation fictive")
        correctly flagged this as fiction for POST-EXEC reporting, but
        pre-exec gating is a distinct problem from per-token accounting.
        A token-count PREDICTOR (rather than a fixed 500/300) is a separate
        research task; the $0.001 fallback fires only when cards.toml has
        no entry for the assigned model.

        Loads the model catalog once (cached) and looks up cost_input_per_m /
        cost_output_per_m for each node's assigned model.  Falls back to
        $0.001 per node when the catalog or model is unavailable.
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
        """Stage 3: Assign model_id to each topology node.

        Cycle-12 Phase B (2026-05-05): body moved to
        `sage.pipeline_v2.assign_models.assign_models`. 1-line LOCAL-import
        delegator. Helpers `_log_model_assigner_chosen_fallback` and
        `_verify_assignment_formal` STAY on this class per cgpro DESIGN
        trap #6 (helper ownership migration is Phase C).
        """
        from sage.pipeline_v2.assign_models import assign_models as _v2_assign_models
        return _v2_assign_models(self, ctx)

    def _log_model_assigner_chosen_fallback(self, ctx: PipelineContext) -> None:
        """T5 diagnostic fallback for opaque Rust assigners.

        The Python fallback logs true top-3 candidates from its scoring table.
        Rust's PyO3 assigner does not expose that table yet, so this branch
        emits only the chosen model as rank 1, with unknown score components.
        """
        if os.environ.get("SAGE_ASSIGNER_LOG_TOP3") != "1":
            return
        if self.assigner is not None and hasattr(self.assigner, "_score_candidates"):
            return
        if ctx.topology is None:
            return

        node_count = (
            ctx.topology.node_count()
            if hasattr(ctx.topology, "node_count")
            else 0
        )
        for node_idx in range(node_count):
            model_id = ctx.assignments.get(node_idx, "")
            if not model_id and hasattr(ctx.topology, "get_node"):
                node = ctx.topology.get_node(node_idx)
                model_id = getattr(node, "model_id", "") if node else ""
            if not model_id:
                continue
            log.info(
                "model_assigner.candidates node_id=%d rank=1 model=%s "
                "source=wrapper_fallback reason_code=non_finite_score "
                "score=%.6f affinity=%.6f domain=%.6f cost_norm=%.6f "
                "hint_bonus=%.6f diversity_penalty=%.6f",
                node_idx,
                model_id,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

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
            node_caps: set[str] = set(node.capabilities_required) if node else set()
            if model_id not in model_caps:
                model_caps[model_id] = set()
            model_caps[model_id].update(node_caps)

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
        """Return (provider, config) for a healthy fallback, or (None, None).

        Preference order (first match wins):
          1. ``self.llm_provider`` if its provider name is alive in the
             pool (i.e. the boot default is not currently dead).
          2. Any provider in ``self.provider_pool._providers`` whose
             circuit is closed and whose TTL'd exclusion hasn't fired.
          3. ``self.llm_provider`` as a last resort (better than nothing).

        Used by Stage 4 single-agent fallback after multi-agent execution
        failed. The previous implementation used ``self.llm_provider``
        unconditionally, which on a provider-outage (minimax 529 storm
        2026-04-21 morning) meant the fallback hit the same dead provider
        and returned empty content. Routing to a different healthy
        provider recovers 3-5/10 tasks that would otherwise be EMPTY.
        """
        pool = getattr(self, "provider_pool", None)

        # Helper: is a provider alive?
        def _alive(pname: str) -> bool:
            if pool is None:
                return True  # No pool → assume alive
            # Dead if TTL'd exclusion or circuit-open.
            if pname in getattr(pool, "_dead_at", {}):
                return False
            if hasattr(pool, "is_available") and not pool.is_available(pname):
                return False
            return True

        # 1. Try the default provider first if it's alive.
        default = self.llm_provider
        default_name = ""
        if default is not None:
            default_name = getattr(default, "name", "") or getattr(default, "provider_name", "")
            if default_name and _alive(default_name):
                return default, self.llm_config

        # 2. Iterate the pool for any alive provider that's not the dead default.
        if pool is not None:
            providers = getattr(pool, "_providers", {}) or {}
            for pname, prov in providers.items():
                if pname == default_name:
                    continue
                if not _alive(pname):
                    continue
                model_id = getattr(prov, "model_id", "") or getattr(prov, "model_string", "")
                from sage.llm.base import LLMConfig
                cfg = LLMConfig(
                    provider=pname,
                    model=model_id,
                    context_window=getattr(self.llm_config, "context_window", 128000) if self.llm_config else 128000,
                )
                log.info(
                    "Stage 4 fallback: rerouting from dead default=%s to healthy %s",
                    default_name or "(none)", pname,
                )
                return prov, cfg

        # 3. Last resort: default even if marked dead (better than nothing).
        return default, self.llm_config

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
