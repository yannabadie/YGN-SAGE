"""CognitiveOrchestrationPipeline — 5-stage cognitive orchestration.

Replaces the inline routing+topology+execution logic in AgentSystem.run()
with a clean, staged pipeline driven by ModelCards and TopologyGraph.
"""
from __future__ import annotations

import asyncio
import contextvars
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
    EXECUTE_BUDGET_EXCEEDED,
    EXECUTE_HALTED_UNVERIFIED,
    EXECUTE_UNVERIFIED,
)

from sage.pipeline_stages import (
    _infer_domain,
    compute_dag_features,
    select_macro_topology,
    DAGFeatures,
)
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
_MULTI_NODE_ATTRIBUTION_TEMPLATES = {"parallel", "parallel_fanout", "debate"}

# P6-B (cycle-11 entry, cgpro round-4 review 2026-05-04): re-entry guard
# for the AgentLoop bypass singleton mutation block. The bypass path
# (CognitiveOrchestrationPipeline._stage_execute, around the historical
# "B9 deferred" comment) snapshots+mutates 12 fields on the boot
# singleton AgentLoop, runs `await self._agent_loop.run(...)`, then
# restores from the snapshot. Two same-event-loop concurrent runs would
# clobber each other's mutations because the snapshot is per-call but
# the singleton is shared.
#
# A naive asyncio.Lock around the block would fix the basic race but
# would deadlock on `sage_recurse` — a tool registered at boot
# (sage/boot.py) that re-invokes `system.run` from inside an AgentLoop
# step. Re-entering pipeline.run() while the bypass lock is already held
# in the same task would attempt to re-acquire the lock and never
# return.
#
# This ContextVar is set inside the lock and checked at the head of the
# bypass path. Nested entry fails fast with a controlled RuntimeError
# instead of silent deadlock. The ContextVar is task-local, so two
# unrelated concurrent runs each see their own False default and
# serialize cleanly through the lock.
_BYPASS_AGENT_LOOP_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sage_pipeline_bypass_agent_loop_active",
    default=False,
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
        # P6-B (cycle-11, cgpro round-4 review 2026-05-04): the boot
        # singleton AgentLoop is shared across all pipeline runs that
        # reach the bypass-mutation block. We serialize concurrent
        # bypass entries on a per-event-loop asyncio.Lock to prevent
        # snapshot/restore interleaving from clobbering the singleton's
        # config across overlapping tasks. Lazy because asyncio.Lock()
        # binds to the running loop on first await; constructing it in
        # __init__ would bind to whatever loop (if any) builds the
        # pipeline. The _loop tracker rebuilds the lock if a fresh
        # event loop is observed (e.g. between two pytest cases that
        # each create-and-tear-down a loop).
        self._agent_loop_bypass_lock: asyncio.Lock | None = None
        self._agent_loop_bypass_lock_loop: asyncio.AbstractEventLoop | None = None
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

    def _emit_budget_exceeded(self, ctx: PipelineContext) -> None:
        cost_tracker = getattr(ctx, "cost_tracker", None)
        data: dict[str, Any] = {"reason": "task budget exceeded"}
        if cost_tracker is not None and hasattr(cost_tracker, "stats"):
            try:
                data.update(cost_tracker.stats())
            except Exception:  # noqa: BLE001 - telemetry must not mask abort
                pass
        self._emit(EXECUTE_BUDGET_EXCEEDED, data)

    def _build_write_gate(self) -> Any:
        """Construct a fresh CompositeWriteGate (Rust if available, Python fallback).

        Governance (A0b, 2026-04-23, ALIRE2 §6): in normal mode a failure
        here logs-and-returns-None (memory writes continue ungated — the
        pre-A0b default that keeps dev smoke runs resilient to a broken
        Rust build). Under ``SAGE_STRICT_GOVERNANCE=1`` the failure is
        re-raised so the caller aborts the pipeline — the posture we
        want in production / audit runs where "continuing ungated"
        silently falsifies the governance claim.
        """
        from sage.memory.write_gate import create_composite_write_gate
        try:
            return create_composite_write_gate(**self._gate_config)
        except Exception as exc:
            if _is_strict_governance():
                log.error(
                    "CompositeWriteGate init failed under SAGE_STRICT_GOVERNANCE=1; "
                    "aborting pipeline: %s", exc,
                )
                raise
            log.debug("CompositeWriteGate init failed, memory writes ungated: %s", exc)
            return None

    def _get_agent_loop_bypass_lock(self) -> asyncio.Lock:
        """Return the per-event-loop asyncio.Lock guarding the bypass mutation block.

        P6-B (cycle-11, cgpro round-4 review 2026-05-04). Constructed on first
        use because asyncio.Lock() binds to the running loop. Reconstructed if a
        different running loop is observed — pytest-asyncio creates a fresh loop
        per test, and a stale lock from a closed loop would either deadlock or
        raise. Single-task per pipeline = no race here; the rebuild only happens
        between tests.
        """
        loop = asyncio.get_running_loop()
        lock = self._agent_loop_bypass_lock
        if lock is None or self._agent_loop_bypass_lock_loop is not loop:
            lock = asyncio.Lock()
            self._agent_loop_bypass_lock = lock
            self._agent_loop_bypass_lock_loop = loop
        return lock

    def _record_to_memory(
        self,
        ctx: PipelineContext,
        *,
        is_training_evidence: bool | None = None,
    ) -> None:
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
                # In-run observability: log the STM → Arrow tier transition so
                # smoke runs can track S-MMU paging frequency per task. The 4
                # events added above (TASK/TOPOLOGY/RESULT/METRICS) guarantee
                # the threshold is met on every pipeline.run().
                events_before = self.working_memory.event_count()
                if events_before >= 4:
                    self.working_memory.compact_to_arrow()
                    log.info(
                        "memory.smmu.tier_transition from=stm to=arrow "
                        "events=%d task_id=%s",
                        events_before, self._task_count,
                    )
            except (RuntimeError, IOError) as exc:
                log.debug("Memory write (Tier 0) failed: %s", exc)

        # Tier 1: Episodic memory (persistent SQLite)
        if self.episodic_memory:
            try:
                entry = {
                    "task": ctx.task[:200],
                    "system": ctx.system,
                    "topology_nodes": ctx.topology.node_count() if ctx.topology and hasattr(ctx.topology, 'node_count') else 0,
                    "assignments": str(ctx.assignments),
                    "result_len": len(ctx.result) if ctx.result else 0,
                    "latency_ms": ctx.latency_ms,
                    "cost": ctx.cost,
                }
                metadata: dict[str, Any] = {}
                if is_training_evidence is not None:
                    entry["is_training_evidence"] = is_training_evidence
                    metadata["is_training_evidence"] = is_training_evidence
                    if ctx.oracle_verdict is not None:
                        metadata["oracle_verdict"] = ctx.oracle_verdict.to_dict()
                import json
                content = json.dumps(entry, default=str)
                # Use sync add if available, else async
                if hasattr(self.episodic_memory, 'add'):
                    try:
                        if metadata:
                            self.episodic_memory.add(
                                key=f"pipeline-{self._task_count}",
                                content=content,
                                metadata=metadata,
                            )
                        else:
                            self.episodic_memory.add(
                                key=f"pipeline-{self._task_count}",
                                content=content,
                            )
                    except TypeError:
                        self.episodic_memory.add(
                            key=f"pipeline-{self._task_count}",
                            content=content,
                        )
                elif hasattr(self.episodic_memory, 'add_episode'):
                    try:
                        if metadata:
                            self.episodic_memory.add_episode(
                                key=f"pipeline-{self._task_count}",
                                content=content,
                                metadata=metadata,
                            )
                        else:
                            self.episodic_memory.add_episode(
                                key=f"pipeline-{self._task_count}",
                                content=content,
                            )
                    except TypeError:
                        self.episodic_memory.add_episode(
                            key=f"pipeline-{self._task_count}",
                            content=content,
                        )
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
            install_event_log,
        )
        from sage.runtime.event_log.redaction import _hash_text
        from sage.runtime.run_frame.builder import _RunFrameBuilder

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
        """Stage 0: Classify task complexity, domain, and select an integrated routing model.

        Priority: Rust SystemRouter route_integrated > Python kNN fallback > heuristic.
        A14b makes the SystemRouter-owned bandit decision here and records it
        only after Stage 4 proves which model/template actually executed.

        Returns: system (S1/S2/S3), domain, and stores RoutingDecision for
        model selection in Stage 3 (ModelAssigner) and Stage 5 telemetry.
        """
        from sage.observability.spans import sage_span
        with sage_span("sage.classify", op="sage.classify"):
            # Priority 1: Rust SystemRouter (full integrated routing)
            if self._rust_router:
                try:
                    ctx.domain = _infer_domain(ctx.task)
                    import importlib

                    RoutingConstraints = getattr(
                        importlib.import_module("sage_core"),
                        "RoutingConstraints",
                    )

                    constraints = RoutingConstraints(
                        max_cost_usd=float(ctx.budget or 0.0),
                        max_latency_ms=0.0,
                        min_quality=0.0,
                        required_capabilities=[],
                        security_label="",
                        exploration_budget=0.1,
                        domain_hint=ctx.domain,
                    )
                    decision = self._rust_router.route_integrated(ctx.task, constraints, "")
                    ctx.system = int(decision.system)
                    decision_id = str(getattr(decision, "decision_id", "") or "")
                    selected_template = str(
                        getattr(decision, "selected_template", "")
                        or getattr(decision, "template", "")
                        or ""
                    )
                    if selected_template:
                        ctx.bandit_decision_id = decision_id
                        ctx.bandit_model_id = str(getattr(decision, "model_id", "") or "")
                        ctx.bandit_template = selected_template
                        ctx.bandit_attribution_state = "pending" if decision_id else "skipped"
                    else:
                        self._clear_bandit_decision(ctx)
                    # Store decision for model selection + telemetry
                    self._last_routing_decision = decision
                    self._last_runtime_routing_source = "rust_system_router"
                    self._last_runtime_routing_confidence = getattr(
                        decision,
                        "confidence",
                        None,
                    )
                    self._last_runtime_routing_model_id = getattr(decision, "model_id", "") or ""
                    log.info(
                        "Stage 0: Rust routing → S%d model=%s (conf=%.2f, cost=%.4f)",
                        ctx.system, decision.model_id, decision.confidence, decision.estimated_cost,
                    )
                    return ctx
                except Exception as exc:
                    log.warning("Stage 0: Rust SystemRouter failed (%s), falling back to Python", exc)
                    self._cancel_bandit_decision(ctx, force=True)
                    ctx.bandit_attribution_state = "skipped"
                    self._emit_bandit_attribution_mismatch(ctx, "router_fallback_degraded")
                    self._clear_bandit_decision(ctx)

            # Priority 2: Python kNN (93.3% accuracy, Rust-accelerated embedding)
            if self.router and hasattr(self.router, '_knn') and self.router._knn is not None:
                try:
                    knn_result = self.router._knn.route(ctx.task)
                    if knn_result is not None:
                        ctx.system = knn_result.system
                        log.info("Stage 0: kNN routing → S%d (conf=%.2f, %s)",
                                 knn_result.system, knn_result.confidence, knn_result.method)
                        ctx.domain = _infer_domain(ctx.task)
                        self._last_runtime_routing_source = "knn"
                        self._last_runtime_routing_confidence = getattr(
                            knn_result,
                            "confidence",
                            None,
                        )
                        self._last_runtime_routing_model_id = ""
                        return ctx
                except (ImportError, RuntimeError) as exc:
                    log.debug("Stage 0: kNN failed (%s), falling back", exc)

            # Priority 3: AdaptiveRouter heuristic
            if self.router:
                try:
                    profile = self.router.assess_complexity(ctx.task)
                    decision = self.router.route(profile)
                    ctx.system = getattr(decision, "system", 2)
                    self._last_runtime_routing_source = "adaptive_router"
                    self._last_runtime_routing_confidence = getattr(decision, "confidence", None)
                    self._last_runtime_routing_model_id = getattr(decision, "model_id", "") or ""
                except (ImportError, RuntimeError) as exc:
                    log.warning("Stage 0 classify failed: %s, defaulting to S2", exc)
                    ctx.system = 2
                    self._last_runtime_routing_source = "default"
                    self._last_runtime_routing_confidence = None
                    self._last_runtime_routing_model_id = ""
            else:
                ctx.system = 2
                self._last_runtime_routing_source = "default"
                self._last_runtime_routing_confidence = None
                self._last_runtime_routing_model_id = ""

            ctx.domain = _infer_domain(ctx.task)
            return ctx

    # ── Stage 1: Decompose ──────────────────────────────────────────────────

    async def _stage_decompose(self, ctx: PipelineContext) -> PipelineContext:
        """Stage 1: Decompose task into sub-tasks (S2/S3 only)."""
        from sage.observability.spans import sage_span
        with sage_span("sage.decompose", op="sage.decompose"):
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
        from sage.observability.spans import sage_span
        with sage_span("sage.topology_select", op="sage.topology_select"):
            skip_dag_template = (
                os.environ.get("SAGE_TOPOLOGY_SKIP_DAG_TEMPLATE") == "1"
                or os.environ.get("SAGE_TOPOLOGY_FORCE_ENGINE") == "1"
            )
            log_all_candidates = (
                os.environ.get("SAGE_TOPOLOGY_LOG_ALL_CANDIDATES") == "1"
            )

            # Sprint 5 ablation: force bypass to measure framework delta.
            if os.environ.get("SAGE_ABLATION_NO_TOPOLOGY") == "1":
                ctx.topology = None
                log.info("Stage 2: topology disabled by SAGE_ABLATION_NO_TOPOLOGY=1 (ablation)")
                return ctx

            # S1 fast path: skip topology for non-math tasks.
            # Math tasks use formal_solver (SatLM NeurIPS 2023): LLM formalizes,
            # Rust solves exactly. Falls back to single-agent if solver fails.
            if ctx.system == 1 and not skip_dag_template:
                if ctx.domain == "math":
                    topo = self._build_topology_from_hint("formal_solver")
                    if topo:
                        ctx.topology = topo
                        log.info("S1 math: formal_solver (formalizer → Rust solver, fallback to CoT)")
                        self._log_topology_structure(topo, source="dag_template", confidence=None)
                        self._apply_topology_budget_and_cache(ctx)
                        return ctx
                ctx.topology = None
                log.debug("S1 task: skipping topology (direct single-agent)")
                return ctx

            # Structure-driven template selection from DAG decomposition
            # Uses omega (parallelism), delta (depth), gamma (coupling) —
            # no regex heuristics, purely structural signals from Stage 1.
            hint = "sequential"
            if ctx.dag_features and not skip_dag_template:
                hint = select_macro_topology(ctx.dag_features, ctx.system, ctx.domain)

            # S2+sequential: use the sequential topology template instead of bypass.
            # Research (AdaptOrch 2602.16873, MASS 2502.02533) shows sequential
            # planner→coder→synthesizer pipeline beats single-agent by 12-23%.
            # The old bypass (SAGE_BYPASS_S2_SEQUENTIAL=1) is available for A/B testing.
            if (
                ctx.system == 2
                and hint == "sequential"
                and not skip_dag_template
            ):
                # `os` is imported at module level; a redundant local `import os`
                # here would shadow it and break the earlier SAGE_ABLATION_NO_TOPOLOGY
                # check (UnboundLocalError when Python treats `os` as local).
                if os.environ.get("SAGE_BYPASS_S2_SEQUENTIAL") == "1":
                    ctx.topology = None
                    log.info("Stage 2: BYPASS topology (SAGE_BYPASS_S2_SEQUENTIAL=1)")
                    return ctx

            # Build topology from template hint. All DAG-selected templates go
            # through TemplateStore which creates multi-node topologies.
            if (
                not skip_dag_template
                and hint in (
                    "sequential",
                    "avr",
                    "parallel",
                    "robust",
                    "horizon_pipeline",
                    "parallel_fanout",
                )
            ):
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
                    # Gap 1+2 (2026-04-21): emit structure log alongside template
                    # name so post-run analysis can attribute pass-rate by DAG
                    # shape (edges) and 6-path source, not just template name.
                    self._log_topology_structure(topo, source="dag_template", confidence=None)
                    self._apply_topology_budget_and_cache(ctx)
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
                    raw_result = self.engine.generate(
                        ctx.task, task_embedding, ctx.system, ctx.budget
                    )
                    candidates = self._topology_candidate_items(raw_result)
                    if log_all_candidates:
                        self._log_topology_candidates(candidates)
                    result = (
                        candidates[0]
                        if isinstance(raw_result, (list, tuple)) and candidates
                        else raw_result
                    )
                    if result and hasattr(result, "topology"):
                        ctx.topology = result.topology
                        # Plan item 1.4a (2026-04-20): always use ctx.topology.id
                        # (full ULID) — the engine's topology_cache / archive lookup
                        # is keyed by this ULID. result.topology_id() returns a
                        # descriptor-keyed semantic ID (e.g. "avr:n3:01KPN3XZ")
                        # which is NOT cache-compatible; using it caused record_outcome
                        # cache misses → archive never grew on the engine branch.
                        ctx.topology_id = getattr(ctx.topology, "id", "")
                    elif result:
                        ctx.topology = result

                    # Gap 1+2 (2026-04-21): log DAG edges + 6-path source
                    # (smmu_hit / archive_hit / llm_synthesis / mutation /
                    # mcts_search / template_fallback) with confidence. The
                    # source is exposed by PyGenerateResult.source() per
                    # sage-core/src/topology/pyo3_wrappers.rs.
                    _src = None
                    _conf = None
                    if result is not None:
                        _src_attr = getattr(result, "source", None)
                        if callable(_src_attr):
                            try:
                                _src = _src_attr()
                            except Exception:
                                _src = None
                        else:
                            _src = _src_attr
                        _conf_attr = getattr(result, "confidence", None)
                        if callable(_conf_attr):
                            try:
                                _conf = _conf_attr()
                            except Exception:
                                _conf = None
                        else:
                            _conf = _conf_attr
                    self._log_topology_structure(
                        ctx.topology, source=_src or "engine_unknown", confidence=_conf,
                    )
                    self._apply_topology_budget_and_cache(ctx)
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

            # Gap 1+2 (2026-04-21): log structure for the fallback path too.
            if ctx.topology is not None:
                self._log_topology_structure(
                    ctx.topology, source="template_fallback", confidence=None,
                )
            self._apply_topology_budget_and_cache(ctx)
            return ctx

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
        """Stage 3: Assign model_id to each topology node."""
        from sage.observability.spans import sage_span
        with sage_span("sage.assign_models", op="sage.assign_models"):
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
                self._log_model_assigner_chosen_fallback(ctx)
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

    def _bandit_task_context(self, ctx: PipelineContext) -> list[float]:
        return [
            float(ctx.system),
            float(len(ctx.task)),
            float(
                ctx.topology.node_count()
                if ctx.topology and hasattr(ctx.topology, "node_count")
                else 0
            ),
        ]

    def _is_single_agent_execution(self, ctx: PipelineContext) -> bool:
        return ctx.topology is None or (
            hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
        )

    def _clear_bandit_decision(self, ctx: PipelineContext) -> None:
        ctx.bandit_decision_id = ""
        ctx.bandit_model_id = ""
        ctx.bandit_template = ""
        ctx.bandit_context = []
        ctx.bandit_attribution_state = "skipped"

    def _cancel_bandit_decision(self, ctx: PipelineContext, *, force: bool = False) -> bool:
        decision_id = str(getattr(ctx, "bandit_decision_id", "") or "")
        if not decision_id and not force:
            return False
        if self._rust_router and hasattr(self._rust_router, "cancel_bandit_decision"):
            try:
                return bool(self._rust_router.cancel_bandit_decision(decision_id))
            except (ImportError, RuntimeError, ValueError) as exc:
                log.warning("Bandit decision cancel failed: %s", exc)
                return False
        log.warning("Bandit decision cancel unavailable: SystemRouter wrapper missing")
        return False

    def _record_bandit_outcome_checked(self, ctx: PipelineContext, quality: float) -> None:
        if oracle_enabled():
            verdict = getattr(ctx, "oracle_verdict", None)
            if verdict is None or not verdict.trainable:
                self._cancel_bandit_decision(ctx)
                self._clear_bandit_decision(ctx)
                return

        if not getattr(ctx, "bandit_decision_id", ""):
            return

        raw_executed_model_ids = getattr(ctx, "executed_model_ids", [])
        if isinstance(raw_executed_model_ids, Mapping):
            raw_executed_model_ids = raw_executed_model_ids.values()
        executed_model_ids = [model_id for model_id in raw_executed_model_ids if model_id]
        if (
            len(set(executed_model_ids)) > 1
            or getattr(ctx, "executed_template", "") in _MULTI_NODE_ATTRIBUTION_TEMPLATES
        ):
            ctx.bandit_attribution_state = "skipped"
            self._cancel_bandit_decision(ctx)
            self._emit_bandit_attribution_mismatch(ctx, "multi_node_ambiguous")
            self._clear_bandit_decision(ctx)
            return

        if not self._rust_router or not hasattr(self._rust_router, "record_outcome_checked"):
            log.warning(
                "Bandit outcome skipped: SystemRouter record_outcome_checked unavailable"
            )
            ctx.bandit_attribution_state = "mismatch"
            self._cancel_bandit_decision(ctx)
            self._emit_bandit_attribution_mismatch(ctx, "recorder_instance_mismatch")
            return

        try:
            self._rust_router.record_outcome_checked(
                ctx.bandit_decision_id,
                ctx.executed_model_id,
                ctx.executed_template,
                quality,
                ctx.cost,
                ctx.latency_ms,
            )
            ctx.bandit_attribution_state = "verified"
        except (ImportError, RuntimeError, ValueError) as exc:
            reason_code = self._bandit_reason_from_exception(exc)
            ctx.bandit_attribution_state = "mismatch"
            self._cancel_bandit_decision(ctx)
            self._emit_bandit_attribution_mismatch(ctx, reason_code)

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
        """Stage 4: Execute topology with per-node model resolution."""
        from sage.observability.spans import sage_span
        with sage_span("sage.execute", op="sage.execute"):
            cost_tracker = getattr(ctx, "cost_tracker", None)
            if cost_tracker is not None and cost_tracker.is_over_budget:
                self._emit_budget_exceeded(ctx)
                ctx.result = BUDGET_EXCEEDED_RESULT
                return ctx

            if not ctx.verification_passed:
                # A0b (2026-04-23, ALIRE2 §6): strict mode aborts here instead
                # of falling through to EXECUTE_UNVERIFIED. The default keeps
                # the historical "log and continue" behaviour so dev smokes
                # don't break on a Z3 unsat that would normally be a soft
                # signal. Production / audit runs set SAGE_STRICT_GOVERNANCE=1.
                if _is_strict_governance():
                    log.error(
                        "Stage 4: aborting under SAGE_STRICT_GOVERNANCE=1 — "
                        "verification failed on provider assignment (SAT check)."
                    )
                    self._emit(
                        EXECUTE_HALTED_UNVERIFIED,
                        {"reason": "SAT check failed in Stage 3"},
                    )
                    raise RuntimeError(
                        "SAGE_STRICT_GOVERNANCE: pipeline aborted — provider "
                        "assignment failed verification (SAT check)."
                    )
                log.warning("Stage 4: executing with unverified provider assignment (SAT check failed)")
                self._emit(EXECUTE_UNVERIFIED, {"reason": "SAT check failed in Stage 3"})

            # Single-agent mode (no topology or single node)
            if self._is_single_agent_execution(ctx):
                ctx.executed_template = "single_agent"
                decision = None
                bandit_provider = None
                bandit_config = None

                if self._agent_loop:
                    # P6-B (cycle-11, cgpro round-4 review 2026-05-04):
                    # serialize concurrent bypass entries on the boot
                    # singleton AgentLoop and fail fast on re-entry. The
                    # snapshot/mutate/run/restore block below mutates 12
                    # fields on the shared singleton; two same-event-loop
                    # concurrent calls would interleave and clobber each
                    # other's restoration. The reentry guard catches the
                    # `sage_recurse` deadlock case (a tool registered at
                    # boot can call back into pipeline.run from inside
                    # this very block — re-acquiring the lock from the
                    # same task hangs forever). Per-run AgentLoop factory
                    # (P6-A) is the structural fix and is deferred to a
                    # later cycle (ADR-015 characterization tests first).
                    if _BYPASS_AGENT_LOOP_ACTIVE.get():
                        raise RuntimeError(
                            "Recursive AgentLoop bypass disabled: "
                            "pipeline.run() was re-entered from inside the "
                            "single-agent bypass mutation block (likely "
                            "via the sage_recurse tool). The shared "
                            "singleton AgentLoop cannot be safely re-used "
                            "while its config snapshot is held — use the "
                            "topology path or the per-run AgentLoop "
                            "factory (P6-A, deferred)."
                        )

                    bypass_lock = self._get_agent_loop_bypass_lock()
                    async with bypass_lock:
                        _bypass_token = _BYPASS_AGENT_LOOP_ACTIVE.set(True)
                        try:
                            # Phase 1: agent_loop.run() provides tools + S2/S3 validation +
                            # guardrails + memory. Replaces the raw provider.generate() loop.

                            # A0a (2026-04-23, ALIRE2 §4 "shared mutable state"):
                            # snapshot EVERY field we are about to mutate before touching
                            # any of them. The prior code snapshotted only `_llm` and
                            # `config.llm`, leaving 8 others (write_gate, gate_*, _on_drift,
                            # validation_level, max_steps, stall_after_tool_steps,
                            # _current_topology) dirty for the next caller after this
                            # bypass path returned. The `finally` block below restores
                            # every one of these; concurrency-safe restoration is now
                            # handled by P6-B (lock + ContextVar reentry guard above).
                            _orig_bypass_state = {
                                "_skip_routing": getattr(self._agent_loop, "_skip_routing", False),
                                "_current_topology": self._agent_loop._current_topology,
                                "write_gate": getattr(self._agent_loop, "write_gate", None),
                                "gate_current_task": getattr(self._agent_loop, "gate_current_task", None),
                                "gate_source_tier": getattr(self._agent_loop, "gate_source_tier", None),
                                "_on_drift": getattr(self._agent_loop, "_on_drift", None),
                                "_run_frame_builder": getattr(
                                    self._agent_loop,
                                    "_run_frame_builder",
                                    None,
                                ),
                                "_runtime_node_run_id": getattr(
                                    self._agent_loop,
                                    "_runtime_node_run_id",
                                    None,
                                ),
                                "validation_level": self._agent_loop.config.validation_level,
                                "max_steps": self._agent_loop.config.max_steps,
                                "stall_after_tool_steps": self._agent_loop.config.stall_after_tool_steps,
                            }

                            # H1: Skip routing in agent_loop (pipeline already routed in Stage 0)
                            self._agent_loop._skip_routing = True
                            # H4: Clear topology (pipeline owns topology, not agent_loop)
                            self._agent_loop._current_topology = None

                            # H5 audit fix (2026-04-19): wire the pipeline-scoped write gate
                            # onto the shared AgentLoop for the single-agent bypass path.
                            # The G-series fix (commit c905d06) only wired the gate through
                            # `agent_loop_factory.create_node_agent_loop` for multi-node
                            # topology traversal. This code path reuses a pre-existing
                            # `self._agent_loop` singleton built at boot — it never saw the
                            # factory wiring, so `loop.write_gate is None` and phases/act.py
                            # fell through to ungated writes. Same silent-bypass class as
                            # H4 (cache_topology) — fix perfectly wired, never fires.
                            self._agent_loop.write_gate = self.write_gate
                            self._agent_loop.gate_current_task = ctx.task
                            self._agent_loop._run_frame_builder = run_frame_builder
                            self._agent_loop._runtime_node_run_id = None
                            try:
                                from sage.memory.write_gate import infer_source_tier
                                model_id = getattr(
                                    getattr(self._agent_loop.config, "llm", None),
                                    "model", None,
                                )
                                self._agent_loop.gate_source_tier = infer_source_tier(model_id)
                            except (ImportError, AttributeError):
                                self._agent_loop.gate_source_tier = "unknown"

                            # H6 audit fix (2026-04-19): wire the drift callback on the
                            # bypass path. The multi-node path sets `_on_drift` via the
                            # factory (topology/runner.py:502-521) so SWITCH_MODEL /
                            # RESET_AGENT classifications forward to
                            # `ProviderPool.record_failure` — tripping the provider's
                            # circuit breaker so subsequent resolve() picks a different
                            # provider. On the bypass path this was never wired; drift
                            # events on S1 tasks logged but had zero effect on routing.
                            # Same silent-bypass class as H5 (write_gate).
                            if (self.provider_pool is not None
                                    and hasattr(self.provider_pool, "record_failure")):
                                _pool_ref = self.provider_pool
                                _bypass_model_id = getattr(
                                    getattr(self._agent_loop.config, "llm", None),
                                    "model", "",
                                ) or "default"

                                def _on_drift_bypass(
                                    provider_hint: str,
                                    action: str,
                                    details: dict[str, Any],
                                    _pool: Any = _pool_ref,
                                    _model: str = _bypass_model_id,
                                ) -> None:
                                    if action not in ("SWITCH_MODEL", "RESET_AGENT"):
                                        return
                                    _key = (provider_hint or _model or "unknown")
                                    try:
                                        _pool.record_failure(
                                            _key,
                                            RuntimeError(
                                                f"drift_{action.lower()} "
                                                f"latency={details.get('latency', '?')}"
                                            ),
                                        )
                                    except Exception:  # noqa: BLE001
                                        pass

                                self._agent_loop._on_drift = _on_drift_bypass

                            # Set validation level from system classification
                            if ctx.system >= 3:
                                self._agent_loop.config.validation_level = 3
                            elif ctx.system >= 2 and self._agent_loop.sandbox_manager:
                                self._agent_loop.config.validation_level = 2
                            else:
                                self._agent_loop.config.validation_level = 1

                            # Plan item 1.1 (2026-04-20): scale singleton max_steps by
                            # ctx.system — close the H5-class bypass extending the
                            # singleton-vs-factory asymmetry. boot.py:279 built the
                            # singleton with max_steps=MAX_AGENT_STEPS=20; the factory
                            # (agent_loop_factory.py:132-137) scales 5/10/20 per system
                            # tier for per-node AgentLoops. Without this line, S1 tasks
                            # on the bypass path run at 4x the factory-intended budget.
                            # agent_loop.py:424 reads self.config.max_steps directly in
                            # the step loop — mutation takes effect on the next .run().
                            self._agent_loop.config.max_steps = {1: 5, 2: 10, 3: 20}.get(ctx.system, 10)

                            # Plan item 1.2 (2026-04-20): scale singleton D8 stall cap
                            # to match the factory (agent_loop_factory.py:151-154).
                            # AgentConfig.stall_after_tool_steps defaults to 0 (D8
                            # disabled), so the singleton never broke out of a tool-step
                            # thrash on S2/S3 bypass. Factory formula:
                            #   stall_cap = (max_steps - 1) if max_steps > 5 else 0
                            # (S1 budget too tight for any window — D8 off; S2→9, S3→19.)
                            # agent_loop.py:511 live-reads config.stall_after_tool_steps
                            # each step → mutation takes effect on next .run().
                            _new_max = self._agent_loop.config.max_steps
                            self._agent_loop.config.stall_after_tool_steps = (
                                _new_max - 1 if _new_max > 5 else 0
                            )

                            _original_llm = self._agent_loop._llm
                            _original_config = self._agent_loop.config.llm
                            active_model_id = ""
                            if decision is not None and bandit_provider is not None:
                                self._agent_loop._llm = bandit_provider
                                self._agent_loop.config.llm = bandit_config
                                active_model_id = decision.model_id
                                log.info(
                                    "Stage 4 bypass: agent_loop using bandit-selected %s (S%d)",
                                    decision.model_id, ctx.system,
                                )
                            else:
                                # Resolve model from Rust routing decision (preserve legacy selection)
                                routing_decision = getattr(self, '_last_routing_decision', None)
                                if routing_decision and routing_decision.model_id and self.provider_pool:
                                    try:
                                        if self.provider_pool.is_model_available(routing_decision.model_id):
                                            resolved_provider, resolved_config = self.provider_pool.resolve(
                                                routing_decision.model_id
                                            )
                                            self._agent_loop._llm = resolved_provider
                                            self._agent_loop.config.llm = resolved_config
                                            active_model_id = routing_decision.model_id
                                            log.info(
                                                "Stage 4 bypass: agent_loop using Rust-selected %s (S%d)",
                                                routing_decision.model_id, ctx.system,
                                            )
                                    except Exception:
                                        pass  # Keep default provider
                            if not active_model_id:
                                active_model_id = (
                                    getattr(self._agent_loop.config.llm, "model", "")
                                    or getattr(self._agent_loop._llm, "model_id", "")
                                    or getattr(self._agent_loop._llm, "model_string", "")
                                    or getattr(self._agent_loop._llm, "name", "")
                                )
                            ctx.executed_model_id = active_model_id
                            ctx.executed_template = "single_agent"

                            try:
                                ctx.result = await self._agent_loop.run(ctx.task)
                                ctx.cost = self._agent_loop.total_cost_usd
                                # Forward tool-use telemetry from the agent loop so bench
                                # manifests reflect actual usage, not dead zeros.
                                ctx.tool_call_count = getattr(self._agent_loop, "tool_call_count", 0)
                                ctx.tool_turn_count = getattr(self._agent_loop, "tool_turn_count", 0)
                                ctx.executed_commands = list(getattr(self._agent_loop, "executed_commands", []))
                            finally:
                                # A0a restoration — complete (12 fields, matches the
                                # snapshot taken before the first mutation above).
                                # Prior to 2026-04-23 this restored only 3 of the 10
                                # mutated fields, leaving write_gate / _on_drift /
                                # validation_level / max_steps / stall_after_tool_steps
                                # / _current_topology dirty for the next caller.
                                # P6-B (2026-05-04) wraps this entire block in a
                                # serializing lock — restoration still runs on
                                # exception/cancellation paths because the outer
                                # try/finally below releases the lock and the
                                # ContextVar token regardless of how we exit.
                                self._agent_loop._skip_routing = _orig_bypass_state["_skip_routing"]
                                self._agent_loop._current_topology = _orig_bypass_state["_current_topology"]
                                self._agent_loop.write_gate = _orig_bypass_state["write_gate"]
                                self._agent_loop.gate_current_task = _orig_bypass_state["gate_current_task"]
                                self._agent_loop.gate_source_tier = _orig_bypass_state["gate_source_tier"]
                                self._agent_loop._on_drift = _orig_bypass_state["_on_drift"]
                                self._agent_loop._run_frame_builder = _orig_bypass_state[
                                    "_run_frame_builder"
                                ]
                                self._agent_loop._runtime_node_run_id = _orig_bypass_state[
                                    "_runtime_node_run_id"
                                ]
                                self._agent_loop.config.validation_level = _orig_bypass_state["validation_level"]
                                self._agent_loop.config.max_steps = _orig_bypass_state["max_steps"]
                                self._agent_loop.config.stall_after_tool_steps = _orig_bypass_state["stall_after_tool_steps"]
                                self._agent_loop._llm = _original_llm
                                self._agent_loop.config.llm = _original_config
                        finally:
                            _BYPASS_AGENT_LOOP_ACTIVE.reset(_bypass_token)

                elif self.llm_provider or bandit_provider is not None:
                    # Simple fallback: single provider.generate() call (no tool loop).
                    # Used only when pipeline is created without agent_loop (e.g., tests).
                    from sage.llm.base import Message, Role

                    active_provider = self.llm_provider
                    active_config = self.llm_config
                    active_model_id = (
                        getattr(self.llm_config, "model", "")
                        if self.llm_config is not None
                        else ""
                    )
                    if decision is not None and bandit_provider is not None:
                        active_provider = bandit_provider
                        active_config = bandit_config
                        active_model_id = decision.model_id
                    else:
                        routing_decision = getattr(self, '_last_routing_decision', None)
                        if routing_decision and routing_decision.model_id and self.provider_pool:
                            try:
                                if self.provider_pool.is_model_available(routing_decision.model_id):
                                    active_provider, active_config = self.provider_pool.resolve(
                                        routing_decision.model_id
                                    )
                                    active_model_id = routing_decision.model_id
                            except Exception:
                                pass
                    if active_provider is not None and not active_model_id:
                        active_model_id = (
                            getattr(active_config, "model", "")
                            or getattr(active_provider, "model_id", "")
                            or getattr(active_provider, "model_string", "")
                            or getattr(active_provider, "name", "")
                        )
                    ctx.executed_model_id = active_model_id
                    ctx.executed_template = "single_agent"

                    messages = [Message(role=Role.USER, content=ctx.task)]
                    try:
                        response = await active_provider.generate(
                            messages=messages, config=active_config,
                        )
                        ctx.result = response.content or ""
                    except (RuntimeError, TimeoutError) as exc:
                        log.error("Stage 4 fallback failed: %s", exc)
                        ctx.result = f"Error: {exc}"
                return ctx

            # Multi-agent mode: use TopologyRunner with ProviderPool
            ctx.executed_model_ids = [
                model_id for _, model_id in sorted(ctx.assignments.items())
            ]
            ctx.executed_template = getattr(ctx.topology, "template_type", "") or "multi_agent"
            try:
                from sage.topology.runner import TopologyRunner

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
                        task_domain=ctx.domain or "",  # F7-symmetric domain gate for PRM
                        on_event=(
                            self.event_bus.emit
                            if self.event_bus and hasattr(self.event_bus, "emit")
                            else None
                        ),
                        # G-series: pipeline-scoped gate + task text for relevance
                        write_gate=self.write_gate,
                        task_text=ctx.task,
                        # T2 phase 0/1 (cgpro 2026-04-29): forward memory
                        # backends so per-node loops can write to real
                        # episodic/semantic/causal stores instead of
                        # always hitting memory_backend_unwired.
                        episodic_memory=self.episodic_memory,
                        semantic_memory=self.semantic_memory,
                        memory_agent=self.memory_agent,
                        causal_memory=self.causal_memory,
                    )

                # Fix C (2026-05-03): adaptive controller adds ~30-50s overhead
                # per task on budget tier (model upgrades + reroutes push tasks
                # over 120s cap). v7 ablation: no-guardrails 7/10 vs full 4/10.
                _effective_controller = (
                    None if self._llm_tier == "budget" else self.controller
                )
                runner = TopologyRunner(
                    graph=ctx.topology,
                    executor=executor,
                    llm_provider=self.llm_provider,
                    llm_config=self.llm_config,
                    provider_pool=self.provider_pool,
                    controller=_effective_controller,
                    axis_hint=ctx.axis_hint,
                    agent_loop_factory=_agent_loop_factory,
                    cost_tracker=getattr(ctx, "cost_tracker", None),
                    assigner=self.assigner,
                    task_domain=getattr(ctx, "domain", "") or "",
                    budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                    event_log=event_log,
                    run_frame_builder=run_frame_builder,
                )
                result = await runner.run(ctx.task)
                # Roll up tool-use telemetry from TopologyRunner → ctx. Without
                # this the bench manifest sees zero even on multi-agent paths
                # (Codex 2026-04-18 review flagged this gap at pipeline.py:963).
                ctx.tool_call_count = getattr(runner, "tool_call_count", 0)
                ctx.tool_turn_count = getattr(runner, "tool_turn_count", 0)
                ctx.executed_commands = list(getattr(runner, "executed_commands", []))
                # Same roll-up for cost. Before Apr 18 2026 ctx.cost came only
                # from the single-loop bypass path, so multi-agent topology runs
                # reported _cost_usd=0 even when each node had metered cost.
                ctx.cost = float(getattr(runner, "total_cost_usd", 0.0) or 0.0)
                if result == BUDGET_EXCEEDED_RESULT:
                    self._emit_budget_exceeded(ctx)
                    ctx.result = result
                    return ctx
                if result == "__REROUTE__" and self.engine:
                    log.info("Topology reroute triggered — REBUILDING full topology (not in-place mutation)")
                    self._emit("REROUTE_REBUILD", {"reason": "controller_triggered"})
                    ctx = self._stage_select_topology(ctx)  # new topology
                    ctx = self._stage_assign_models(ctx)    # re-assign models
                    self._runtime_emit_topology_selected(
                        ctx,
                        event_log,
                        run_frame_builder,
                        reason="reroute",
                    )
                    self._runtime_emit_model_assigned(ctx, event_log, run_frame_builder)
                    ctx.executed_model_ids = [
                        model_id for _, model_id in sorted(ctx.assignments.items())
                    ]
                    ctx.executed_template = (
                        getattr(ctx.topology, "template_type", "") or "multi_agent"
                    )
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
                        cost_tracker=getattr(ctx, "cost_tracker", None),
                        assigner=self.assigner,
                        task_domain=getattr(ctx, "domain", "") or "",
                        budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                        event_log=event_log,
                        run_frame_builder=run_frame_builder,
                    )
                    result = await runner2.run(ctx.task)
                    # Prefer the post-reroute telemetry (it's the attempt that
                    # actually produced the final output).
                    ctx.tool_call_count = getattr(runner2, "tool_call_count", 0)
                    ctx.tool_turn_count = getattr(runner2, "tool_turn_count", 0)
                    ctx.executed_commands = list(getattr(runner2, "executed_commands", []))
                    ctx.cost = float(getattr(runner2, "total_cost_usd", 0.0) or 0.0)
                    if result == BUDGET_EXCEEDED_RESULT:
                        self._emit_budget_exceeded(ctx)
                        ctx.result = result
                        return ctx

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
                            self._runtime_emit_model_assigned(
                                ctx,
                                event_log,
                                run_frame_builder,
                            )
                            from sage_core import TopologyExecutor as _TE  # type: ignore[import-not-found]
                            executor2 = _TE(ctx.topology)
                            runner3 = TopologyRunner(
                                graph=ctx.topology, executor=executor2,
                                llm_provider=self.llm_provider, llm_config=self.llm_config,
                                provider_pool=self.provider_pool,
                                agent_loop_factory=_agent_loop_factory,
                                cost_tracker=getattr(ctx, "cost_tracker", None),
                                assigner=self.assigner,
                                task_domain=getattr(ctx, "domain", "") or "",
                                budget_usd=float(getattr(ctx, "budget", 0.0) or 0.0),
                                event_log=event_log,
                                run_frame_builder=run_frame_builder,
                            )
                            retry_result = await runner3.run(ctx.task)
                            if retry_result:
                                result = retry_result
                                log.info("Stage 4: FrugalGPT cascade succeeded on retry")
                        except (RuntimeError, TimeoutError) as exc:
                            log.debug("Stage 4: FrugalGPT cascade retry failed: %s", exc)

                if result == BUDGET_EXCEEDED_RESULT:
                    self._emit_budget_exceeded(ctx)
                    ctx.result = result
                    return ctx

                ctx.result = result
                # Prefer the runner's aggregated real cost (summed from per-node
                # AgentLoop.total_cost_usd, which now prefers provider-reported
                # cost_usd). Fall back to the 500-in/300-out per-node estimate
                # only when no node reported a real cost (e.g. fully-mocked
                # tests). Before Apr 18 2026 this was always the estimate, so
                # benches never saw real provider metering even when LiteLLM
                # populated it correctly.
                if not ctx.cost:
                    ctx.cost = self._estimate_topology_cost(ctx)
            except (ImportError, RuntimeError, TimeoutError) as exc:
                log.error("Stage 4 multi-agent execution failed: %s — falling back to single-agent", exc)
                # Fallback: run task directly on a healthy provider.
                #
                # 2026-04-21 v17 fix: previously used self.llm_provider
                # unconditionally, which is typically the boot-default — often
                # the same provider the multi-agent stage just failed on (e.g.
                # minimax 529 storm). Result: fallback hits the same 529
                # immediately, or returns "" as "success" and SAGE emits an
                # EMPTY patch (5/10 tasks on 2026-04-21 v13 smoke). Now we
                # prefer a healthy provider from the pool — if the default is
                # dead we try the first alive one instead. If the provider
                # returns empty content, we RAISE (not silently emit "") so
                # the bench classifier records an honest error.
                fallback_provider, fallback_config = self._pick_fallback_provider()
                if fallback_provider is not None:
                    try:
                        from sage.llm.base import Message, Role
                        response = await fallback_provider.generate(
                            messages=[Message(role=Role.USER, content=ctx.task)],
                            config=fallback_config or self.llm_config,
                        )
                        content = (response.content or "").strip()
                        if not content:
                            raise RuntimeError(
                                "Stage 4 fallback returned empty content — "
                                "treating as failure rather than emitting empty patch"
                            )
                        ctx.result = response.content or ""
                        log.info(
                            "Stage 4 fallback single-agent succeeded (%d chars, provider=%s)",
                            len(ctx.result),
                            getattr(fallback_provider, "name", type(fallback_provider).__name__),
                        )
                    except (RuntimeError, TimeoutError) as fallback_exc:
                        log.error("Stage 4 fallback also failed: %s", fallback_exc)
                        ctx.result = ""
                else:
                    log.error("Stage 4 fallback: no healthy provider available")
                    ctx.result = ""

            return ctx

    # ── Stage 5: Learn ──────────────────────────────────────────────────────

    async def _stage_learn(self, ctx: PipelineContext) -> None:
        """Stage 5: Record outcome for learning.

        Quality signal for bandit feedback. Research background: ETH-SRI's
        Cascade Routing (arxiv 2410.10347, ICLR 2025) established that
        quality estimation — not the routing algorithm — is the bottleneck
        in LLM-routing bandits. A "PILOT 2508.21141" citation previously
        sat here but that arxiv ID is dead (see 2026-04-22 audit
        verification D.1); removed rather than kept as a ghost reference.

        - Empty result: quality = 0.0 (definitively bad, bandit learns from it)
        - QualityEstimator returns float: use it
        - QualityEstimator returns None: abstain — bandit does NOT record
        - No estimator: abstain — bandit does NOT record
        """
        from sage.observability.spans import sage_span
        with sage_span("sage.learn", op="sage.learn"):
            import re

            quality: float | None = None
            oracle_on = oracle_enabled()
            oracle_trainable = False

            if oracle_on:
                verdict = getattr(ctx, "oracle_verdict", None)
                if verdict is not None and verdict.trainable:
                    quality = verdict.score
                    oracle_trainable = quality is not None
            # Empty result => total failure, bandit must learn from it
            elif not ctx.result or not ctx.result.strip():
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
            if (
                not oracle_on
                and self.prm
                and quality is not None
                and ctx.result
                and _STRUCTURED.search(ctx.result)
            ):
                try:
                    r_path, _ = self.prm.calculate_r_path(ctx.result)
                    if r_path >= 0.0:  # valid score (negative = penalty for no reasoning)
                        quality = 0.8 * quality + 0.2 * r_path
                        log.debug("PRM blended quality: %.2f (estimator + PRM)", quality)
                except (RuntimeError, ValueError) as exc:
                    log.warning("PRM scoring failed in LEARN: %s", exc)

            # Only record to bandit when quality is known and attribution is causal.
            if quality is not None:
                self._record_bandit_outcome_checked(ctx, quality)
            else:
                self._cancel_bandit_decision(ctx)
                self._clear_bandit_decision(ctx)

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

                        # In-run observability: read archive cell count before +
                        # after record_outcome so we can attribute MAP-Elites growth
                        # to a specific topology_id. The H4 incident (dc51976)
                        # proved this delta is frequently 0 even when record_outcome
                        # returns cleanly — the log would have caught that bypass
                        # immediately, so we keep it structured going forward.
                        cells_before = 0
                        if hasattr(self.engine, 'archive_cell_count'):
                            try:
                                cells_before = int(self.engine.archive_cell_count())
                            except (RuntimeError, TypeError):
                                cells_before = 0
                        self.engine.record_outcome(
                            topology_id,
                            ctx.task[:200],
                            keywords,
                            task_embedding,  # real embedding instead of None
                            quality,
                            ctx.cost,
                            ctx.latency_ms,
                        )
                        cells_after = cells_before
                        if hasattr(self.engine, 'archive_cell_count'):
                            try:
                                cells_after = int(self.engine.archive_cell_count())
                            except (RuntimeError, TypeError):
                                cells_after = cells_before
                        if cells_after > cells_before:
                            # Get descriptor shape for the new cell when available
                            coverage = 0.0
                            if hasattr(self.engine, 'archive_coverage'):
                                try:
                                    coverage = float(self.engine.archive_coverage())
                                except (RuntimeError, TypeError):
                                    coverage = 0.0
                            log.info(
                                "memory.archive.grow cells=%d delta=%d "
                                "coverage=%.3f topology=%s quality=%.2f",
                                cells_after, cells_after - cells_before,
                                coverage, topology_id[:8] if topology_id else "unknown",
                                quality,
                            )
                        log.debug(
                            "Evolution: recorded outcome for topology %s (quality=%.2f)",
                            topology_id[:8], quality,
                        )
                except (ImportError, RuntimeError) as exc:
                    log.debug("Evolution feedback failed: %s", exc)

            # H1 audit fix (2026-04-19): Online evolution gate.
            #
            # Architecture/architecture.md and evolution/README.md both claim
            # "Online evolution: Rust should_evolve() gates evolve() in agent
            # loop (SA-3 complete)". The Rust impl exists at
            # `sage-core/src/topology/engine.rs:644 (should_evolve)` and
            # `:668 (evolve)`, both exposed via PyO3, but no Python call site
            # invoked them — same class of bypass as the G-series write-gate
            # (built, tested in isolation, never wired). `_auto_evolve=True`
            # is set on AgentLoop in boot.py:332 but no code reads that flag.
            #
            # Wiring at the end of LEARN is the correct hook: by here we've
            # just appended an outcome to the archive, so should_evolve()
            # has fresh data. Constants come from sage.constants — single
            # source of truth, already covered by test_online_evolution.py.
            allow_training_updates = (not oracle_on) or oracle_trainable
            if allow_training_updates and self.engine and hasattr(self.engine, "should_evolve"):
                try:
                    from sage.constants import (
                        EVOLUTION_MIN_OUTCOMES,
                        EVOLUTION_COOLDOWN_OUTCOMES,
                        EVOLUTION_ONLINE_POP_SIZE,
                        EVOLUTION_ONLINE_GENERATIONS,
                    )
                    decision = self.engine.should_evolve(
                        EVOLUTION_MIN_OUTCOMES, EVOLUTION_COOLDOWN_OUTCOMES,
                    )
                    # In-run observability: log every should_evolve() call, not
                    # just the True branch. Without this, False branches are
                    # invisible -- we can't tell if the gate was even evaluated
                    # on a given task, which is exactly the class of silent-bypass
                    # that H1 (2cd840e) and H4 (dc51976) were. Archive context
                    # travels with the log so post-run analysis can correlate.
                    cells = 0
                    coverage = 0.0
                    if hasattr(self.engine, "archive_cell_count"):
                        try:
                            cells = int(self.engine.archive_cell_count())
                        except (RuntimeError, TypeError):
                            cells = 0
                    if hasattr(self.engine, "archive_coverage"):
                        try:
                            coverage = float(self.engine.archive_coverage())
                        except (RuntimeError, TypeError):
                            coverage = 0.0
                    log.info(
                        "evolution.should_evolve.decision decision=%s cells=%d "
                        "coverage=%.3f min_outcomes=%d cooldown=%d",
                        "true" if decision else "false", cells, coverage,
                        EVOLUTION_MIN_OUTCOMES, EVOLUTION_COOLDOWN_OUTCOMES,
                    )
                    if decision:
                        # Read cells before evolve to show the effect of the call
                        cells_pre_evolve = cells
                        self.engine.evolve(
                            pop_size=EVOLUTION_ONLINE_POP_SIZE,
                            generations=EVOLUTION_ONLINE_GENERATIONS,
                        )
                        cells_post_evolve = cells_pre_evolve
                        if hasattr(self.engine, "archive_cell_count"):
                            try:
                                cells_post_evolve = int(self.engine.archive_cell_count())
                            except (RuntimeError, TypeError):
                                cells_post_evolve = cells_pre_evolve
                        log.info(
                            "evolution.evolve.called pop_size=%d generations=%d "
                            "cells_before=%d cells_after=%d",
                            EVOLUTION_ONLINE_POP_SIZE, EVOLUTION_ONLINE_GENERATIONS,
                            cells_pre_evolve, cells_post_evolve,
                        )
                        # Per-child operator logs. Rust's TopologyEngine tracks
                        # the Thompson-sampled operator name for every mutation
                        # attempt inside evolve() (see pyo3_wrappers.rs
                        # PyTopologyEngine.drain_last_applied_ops). We drain the
                        # buffer here and emit one `evolution.mutation.applied`
                        # line per attempt. Closes gap #3 from the 0bcb92b
                        # pillar-logging pass (per-mutation-operator observability).
                        #
                        # tier: best-effort — the pipeline's most recent routing
                        # decision. evolve() mutations aren't bound to a specific
                        # routing tier (they operate on cached topologies), so we
                        # emit the current-task tier purely for correlation in
                        # post-run analysis.
                        if hasattr(self.engine, "drain_last_applied_ops"):
                            try:
                                applied_ops = list(self.engine.drain_last_applied_ops())
                            except (RuntimeError, TypeError):
                                applied_ops = []
                            if applied_ops:
                                import hashlib as _hashlib
                                tier = ""
                                rd = getattr(self, "_last_routing_decision", None)
                                if rd is not None:
                                    tier = getattr(rd, "llm_tier", "") or ""
                                topo_id = ctx.topology_id or getattr(
                                    ctx.topology, "id", "",
                                )
                                parent_cell = cells_pre_evolve
                                for child_idx, op_name in enumerate(applied_ops):
                                    # child_hash: stable short hash mixing the
                                    # topology id, child index, and op name.
                                    # Opaque but reproducible per-run identifier.
                                    h_src = f"{topo_id}:{child_idx}:{op_name}".encode(
                                        "utf-8",
                                    )
                                    child_hash = _hashlib.blake2b(
                                        h_src, digest_size=4,
                                    ).hexdigest()
                                    log.info(
                                        "evolution.mutation.applied op=%s "
                                        "parent_cell=%d child_hash=%s tier=%s",
                                        op_name, parent_cell, child_hash,
                                        tier or "unknown",
                                    )
                except (ImportError, RuntimeError, AttributeError) as exc:
                    # Engine without these methods (e.g. test stub) → silent skip.
                    log.debug("Online evolution gate skipped: %s", exc)

            # ── Periodic maintenance ───────────────────────────────────────────
            self._task_count += 1

            # Inter-tier consolidation: episodic → semantic → causal (MAGMA 2601.03236)
            from sage.constants import CONSOLIDATION_INTERVAL_STEPS
            if (allow_training_updates
                    and self._task_count % CONSOLIDATION_INTERVAL_STEPS == 0
                    and self.consolidator is not None):
                try:
                    consolidation_result = await self.consolidator.consolidate()
                    # In-run observability: emit a structured log regardless of
                    # whether any entries were processed, so smoke runs can see
                    # consolidation ran on the scheduled interval. The old DEBUG
                    # log was silent for zero-processed passes — now every firing
                    # is accounted for.
                    processed = getattr(consolidation_result, 'processed', 0)
                    entities = getattr(consolidation_result, 'entities_added', 0)
                    edges = getattr(consolidation_result, 'causal_edges_added', 0)
                    log.info(
                        "memory.consolidation.fired processed=%d entities=%d "
                        "causal_edges=%d task_id=%d",
                        processed, entities, edges, self._task_count,
                    )
                except (RuntimeError, IOError):
                    pass  # Best-effort, never blocks pipeline

            # Bandit + MAP-Elites state persistence (crash-safe, WAL write ~5ms)
            #
            # Cycle-11 follow-up (2026-05-05, advisor-flagged 2026-05-04):
            # symmetric with the atexit handler at boot_topology.py:185.
            # Both call sites preflight epoch consistency before writing
            # state so directive #8 (A14 posterior epoch guard is fail-
            # closed) holds across the run lifecycle, not just at session
            # boundaries. Without this preflight, the periodic flush
            # would (a) keep writing state under SAGE_BOOT_BYPASS_EPOCH_GUARD=1
            # while atexit correctly skips, (b) overwrite a contaminated
            # marker and erase forensic evidence, (c) deepen a manifest-
            # less state dir making the next boot fail-close.
            from sage.constants import BANDIT_FLUSH_INTERVAL
            if (self._task_count % BANDIT_FLUSH_INTERVAL == 0
                    and self.engine and hasattr(self.engine, 'save_state')):
                try:
                    from pathlib import Path
                    from sage.posterior_epoch import (
                        ensure_clean_epoch_before_save,
                        is_a14_epoch_guard_error,
                    )
                    state_dir = Path.home() / ".sage"
                    # Preflight: matches boot_topology.py:185 atexit
                    # handler ordering. Raises on bypass active /
                    # contaminated marker / missing manifest / mismatch.
                    ensure_clean_epoch_before_save(state_dir)
                    self.engine.save_state(str(state_dir))
                    log.debug("Periodic state flush (%d tasks)", self._task_count)
                except (RuntimeError, IOError) as exc:
                    if is_a14_epoch_guard_error(exc):
                        # Visible warning: A14 guard fired. Operator
                        # needs this signal to know periodic saves are
                        # skipping. Atexit logs at .info; periodic at
                        # .warning since it can recur many times in a
                        # session and ops should investigate sooner.
                        log.warning(
                            "Periodic state flush blocked by A14 epoch "
                            "guard: %s", exc,
                        )
                    # else: silent best-effort behaviour preserved
                    # (e.g. transient I/O errors, manifest verification
                    # failures, malformed state on disk).
