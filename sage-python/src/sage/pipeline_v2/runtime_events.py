"""Runtime-event helpers for `CognitiveOrchestrationPipeline`.

EventBus emission helpers + the 11 `runtime_emit_*` builders that the
orchestrator and the Stage 4 reroute / rebuild paths consume:

  - `emit` — generic EventBus emission entry point (also retained as
    `pipeline._emit` instance method per Q3a EventBus seam lock).
  - `emit_bandit_attribution_mismatch` — invariant 6 mismatch event.
  - `bandit_reason_from_exception` — pure exception-to-reason-code mapper.
  - `runtime_node_count` / `runtime_edge_type` /
    `runtime_node_capabilities` / `runtime_graph_digest` — pure
    helpers (no `pipeline` arg).
  - `runtime_edge_summary` / `runtime_node_summary` — read pipeline
    + topology / ctx state.
  - `runtime_provider_id_for_model` — uses `pipeline.provider_pool`
    + `pipeline.llm_config`.
  - `runtime_emit_topology_selected` / `runtime_emit_model_assigned` —
    canonical event-emission entry points consumed by
    `pipeline_v2/orchestrator.py:run_internal` and the topology
    controller's reroute / rebuild paths.
  - `runtime_final_status` / `runtime_final_node_count` — final-result
    classification.

No top-level imports of `sage.pipeline` (TYPE_CHECKING only). The
`BUDGET_EXCEEDED_RESULT` and `_BANDIT_ATTRIBUTION_REASON_CODES`
constants are imported LAZILY inside the functions that need them so
the `pipeline_v2/__init__.py` PEP 562 lazy resolver stays acyclic.

Logger uses ``sage.pipeline`` so trace-grep continuity is preserved
across the refactor.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import (
        BanditAttributionReasonCode,
        CognitiveOrchestrationPipeline,
        PipelineContext,
    )
    from sage.runtime.run_frame import RunStatus


log = logging.getLogger("sage.pipeline")


def emit(
    pipeline: "CognitiveOrchestrationPipeline",
    stage: str,
    data: dict[str, Any],
) -> None:
    """Emit a PIPELINE-tagged AgentEvent on the pipeline's EventBus, if available.

    No-op when the pipeline has no event_bus or when AgentEvent
    construction fails (defensive ImportError / RuntimeError).
    """
    event_bus = pipeline.event_bus
    if event_bus and hasattr(event_bus, "emit"):
        try:
            from sage.agent_loop import AgentEvent

            event_bus.emit(
                AgentEvent(
                    type="PIPELINE",
                    step=0,
                    timestamp=time.time(),
                    meta={"stage": stage, **data},
                )
            )
        except (ImportError, RuntimeError):
            pass


def emit_bandit_attribution_mismatch(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    reason_code: "BanditAttributionReasonCode",
) -> None:
    """Emit `bandit_attribution_mismatch` event + PIPELINE EventBus mirror."""
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
    pipeline._emit("BANDIT_ATTRIBUTION_MISMATCH", payload)


def bandit_reason_from_exception(exc: Exception) -> "BanditAttributionReasonCode":
    """Map an exception's message text to a canonical bandit-attribution reason code."""
    from sage.pipeline import _BANDIT_ATTRIBUTION_REASON_CODES

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


def runtime_node_count(topology: Any) -> int:
    """Pure helper — return the topology's node count or 0 on any failure."""
    if topology is None or not hasattr(topology, "node_count"):
        return 0
    try:
        node_count = topology.node_count()
        return int(node_count() if callable(node_count) else node_count)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def runtime_edge_type(value: Any) -> str:
    """Pure helper — map an integer/value edge code to its canonical string label."""
    if value == 0:
        return "control"
    if value == 1:
        return "message"
    if value == 2:
        return "state"
    return str(value or "")


def runtime_edge_summary(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    topology: Any,
) -> tuple[int, list[dict[str, Any]]]:
    """Return (edge_count, list of edge summaries) from a topology graph."""
    if topology is None:
        return 0, []
    try:
        if hasattr(topology, "get_edges"):
            raw_edges = list(topology.get_edges() or [])
            summaries: list[dict[str, Any]] = []
            for idx, edge in enumerate(raw_edges):
                source_id = edge[0] if len(edge) > 0 else ""
                target_id = edge[1] if len(edge) > 1 else ""
                edge_type = runtime_edge_type(edge[2] if len(edge) > 2 else "")
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


def runtime_node_summary(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext",
) -> list[dict[str, Any]]:
    """Return per-node summary dicts for the canonical TopologySelected payload."""
    topology = ctx.topology
    summaries: list[dict[str, Any]] = []
    for idx in range(runtime_node_count(topology)):
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


def runtime_provider_id_for_model(
    pipeline: "CognitiveOrchestrationPipeline",
    model_id: str,
    ctx: "PipelineContext",
) -> str:
    """Resolve a model_id to its authoritative provider_id.

    ``ctx.provider_hints`` is diagnostic input to assignment, not an
    authorization source. It must not relabel audit events.
    """
    if pipeline.provider_pool is not None and hasattr(pipeline.provider_pool, "infer_provider"):
        try:
            provider_id = pipeline.provider_pool.infer_provider(model_id)
            if provider_id:
                return str(provider_id)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return str(getattr(pipeline.llm_config, "provider", "") if pipeline.llm_config else "")


def runtime_node_capabilities(node: Any) -> tuple[str, ...]:
    """Pure helper — read declared capabilities on a TopologyNode."""
    for attr in ("required_capabilities", "capabilities_required", "capabilities"):
        raw = getattr(node, attr, None)
        if raw:
            return tuple(str(item) for item in raw)
    return ()


def runtime_graph_digest(
    *,
    nodes_summary: list[dict[str, Any]],
    edges_summary: list[dict[str, Any]],
) -> str:
    """Pure helper — SHA-256 hex digest of a canonical (nodes, edges) JSON dump."""
    canonical = json.dumps(
        {"nodes": nodes_summary, "edges": edges_summary},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def runtime_emit_topology_selected(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any,
    run_frame_builder: Any | None = None,
    *,
    reason: str = "initial",
) -> None:
    """Emit `TopologySelected` event + RunFrame record entry."""
    if ctx.topology is None:
        return
    edge_count, edges_summary = runtime_edge_summary(pipeline, ctx.topology)
    nodes_summary = runtime_node_summary(pipeline, ctx)
    topology_id = ctx.topology_id or getattr(ctx.topology, "id", "") or ""
    seq = None
    if event_log is not None:
        seq = event_log.emit_topology_selected(
            topology_id=topology_id,
            template_type=getattr(ctx.topology, "template_type", "") or "",
            node_count=runtime_node_count(ctx.topology),
            edge_count=edge_count,
            nodes_summary=nodes_summary,
            edges_summary=edges_summary,
        )
        try:
            from sage.pipeline_v2 import learning_side_effects as lse_mod

            lse_mod.store_event_ref(ctx, "topology_selected", event_log)
        except Exception:  # noqa: BLE001 - audit sidecar refs are best-effort
            pass
    if run_frame_builder is not None:
        run_frame_builder.record_topology_selected(
            seq=seq,
            topology_id=topology_id,
            graph_digest=runtime_graph_digest(
                nodes_summary=nodes_summary,
                edges_summary=edges_summary,
            ),
            reason=reason,
        )


def runtime_emit_model_assigned(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any,
    run_frame_builder: Any | None = None,
) -> None:
    """Emit `ModelAssigned` events + RunFrame records, one per topology node."""
    if ctx.topology is None:
        return
    for idx in range(runtime_node_count(ctx.topology)):
        try:
            node = ctx.topology.get_node(idx)
        except (AttributeError, RuntimeError, TypeError):
            continue
        model_id = ctx.assignments.get(idx, getattr(node, "model_id", "") or "")
        node_role = getattr(node, "role", "") or f"node-{idx}"
        provider_id = runtime_provider_id_for_model(pipeline, model_id, ctx)
        capabilities = runtime_node_capabilities(node)
        seq = None
        if event_log is not None:
            seq = event_log.emit_model_assigned(
                node_id=str(idx),
                node_role=node_role,
                model_id=model_id,
                provider_id=provider_id,
                required_capabilities=capabilities,
            )
            try:
                from sage.pipeline_v2 import learning_side_effects as lse_mod

                lse_mod.store_event_ref(ctx, "model_assigned", event_log)
            except Exception:  # noqa: BLE001 - audit sidecar refs are best-effort
                pass
        if run_frame_builder is not None:
            run_frame_builder.record_model_assigned(
                seq=seq,
                node_id=str(idx),
                node_role=node_role,
                model_id=model_id,
                provider_id=provider_id,
                required_capabilities=capabilities,
            )


# Slice 10D (cgpro DESIGN_LOCK 2026-05-11 Route A, v0):
# `provider_execution_witness` reason-code enum. Compact strings so the
# event payload doesn't carry long free-form text. Keep in sync with
# the doc in docs/superpowers/plans/2026-05-10-handoff-recovery-plan.md.
_WITNESS_REASON_PASSES_POLICY = "passes_policy"
_WITNESS_REASON_NO_POLICY_ACTIVE = "no_policy_active"
_WITNESS_REASON_PROVIDER_IN_DENYLIST = "provider_in_denylist"
_WITNESS_REASON_PROVIDER_OUTSIDE_ALLOWLIST = "provider_outside_allowlist"
_WITNESS_REASON_UNKNOWN_PROVIDER = "unknown_provider"
_WITNESS_REASON_ROUTING_PROVIDER_UNRESOLVED = "routing_provider_unresolved"
_WITNESS_REASON_ASSIGNMENT_PROVIDER_UNRESOLVED = "assignment_provider_unresolved"

_WITNESS_REASON_CODES = (
    _WITNESS_REASON_PASSES_POLICY,
    _WITNESS_REASON_NO_POLICY_ACTIVE,
    _WITNESS_REASON_PROVIDER_IN_DENYLIST,
    _WITNESS_REASON_PROVIDER_OUTSIDE_ALLOWLIST,
    _WITNESS_REASON_UNKNOWN_PROVIDER,
    _WITNESS_REASON_ROUTING_PROVIDER_UNRESOLVED,
    _WITNESS_REASON_ASSIGNMENT_PROVIDER_UNRESOLVED,
)


# Rust filter rejection reason codes (cgpro DESIGN_LOCKED v1_1,
# 2026-05-11). Map 1:1 to predicates in
# `sage-core/src/routing/model_assigner.rs`. Order is significant:
# first-match wins per Rust predicate order (cgpro Q2 lock).
_RUST_FILTER_REASON_CARD_INACTIVE = "card_inactive"
_RUST_FILTER_REASON_PROVIDER_UNKNOWN = "provider_excluded_policy_unknown_provider"
_RUST_FILTER_REASON_PROVIDER_DENYLIST = "provider_excluded_policy_denylist"
_RUST_FILTER_REASON_PROVIDER_ALLOWLIST = "provider_excluded_policy_allowlist"
_RUST_FILTER_REASON_PROVIDER_DEAD = "provider_excluded_dead"
_RUST_FILTER_REASON_EXCLUDED_BY_CALLER = "excluded_by_caller"
_RUST_FILTER_REASON_CAPABILITY_MISMATCH = "capability_mismatch"
_RUST_FILTER_REASON_COST_ABOVE_BUDGET = "cost_above_budget"

_RUST_FILTER_REASON_CODES = (
    _RUST_FILTER_REASON_CARD_INACTIVE,
    _RUST_FILTER_REASON_PROVIDER_UNKNOWN,
    _RUST_FILTER_REASON_PROVIDER_DENYLIST,
    _RUST_FILTER_REASON_PROVIDER_ALLOWLIST,
    _RUST_FILTER_REASON_PROVIDER_DEAD,
    _RUST_FILTER_REASON_EXCLUDED_BY_CALLER,
    _RUST_FILTER_REASON_CAPABILITY_MISMATCH,
    _RUST_FILTER_REASON_COST_ABOVE_BUDGET,
)

# Cap per cgpro DESIGN_LOCK Q3 + substitution_summary 8 KB UTF-8 budget.
_RUST_FILTER_REJECTIONS_CAP = 20


def _runtime_read_rust_filter_rejections(
    pipeline: Any,
) -> tuple[list[dict[str, str]] | None, bool]:
    """Read structured filter rejections from the Rust ModelAssigner.

    Returns ``(rejections, truncated)`` where:
    - ``rejections`` is ``None`` if Rust didn't run a recording-aware
      path (legacy wheel without the v1_1 methods, or assigner absent),
      ``[]`` if it ran and rejected nothing, or a non-empty list of
      ``{model_id, reason_code}`` dicts.
    - ``truncated`` is ``True`` iff the Rust side truncated the list to
      the 20-entry cap.

    Defensive: any unexpected shape or AttributeError returns
    ``(None, False)`` so observability never breaks the run.
    """
    assigner = getattr(pipeline, "assigner", None)
    if assigner is None:
        return None, False
    raw_rejections = None
    raw_truncated = False
    try:
        getter = getattr(assigner, "last_filter_rejections", None)
        if getter is None:
            return None, False
        raw_rejections = getter()
        trunc_getter = getattr(
            assigner, "last_filter_rejections_truncated", None
        )
        if trunc_getter is not None:
            raw_truncated = bool(trunc_getter())
    except (AttributeError, RuntimeError, TypeError):
        return None, False
    if raw_rejections is None:
        return None, False
    if not isinstance(raw_rejections, (list, tuple)):
        return None, False
    rejections: list[dict[str, str]] = []
    for entry in raw_rejections:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            # Unrecognized shape — fail-closed to None so callers don't
            # half-record. Better silent than half-true.
            return None, False
        model_id, reason_code = entry
        model_id = str(model_id or "")
        reason_code = str(reason_code or "")
        if reason_code not in _RUST_FILTER_REASON_CODES:
            # Drop entries with unknown reason codes — could be a Rust
            # version that uses a code we don't recognise. Don't fail
            # the run; just don't surface unknown labels.
            continue
        rejections.append(
            {"model_id": model_id, "reason_code": reason_code}
        )
        if len(rejections) >= _RUST_FILTER_REJECTIONS_CAP:
            break
    return rejections, raw_truncated


def _classify_provider_against_policy(
    provider_id: str,
    *,
    allowlist: tuple[str, ...],
    denylist: tuple[str, ...],
    policy_active: bool,
) -> tuple[str, str]:
    """Decide whether ``provider_id`` is allowed by the current policy.

    Returns ``(decision, reason_code)`` where ``decision`` is
    ``"allowed" | "blocked" | "unresolved"``.

    Pure function: no side effects, no event emission, no I/O.
    """
    if not provider_id:
        return "unresolved", (
            _WITNESS_REASON_ROUTING_PROVIDER_UNRESOLVED
            if not policy_active
            else _WITNESS_REASON_ROUTING_PROVIDER_UNRESOLVED
        )
    if not policy_active:
        return "allowed", _WITNESS_REASON_NO_POLICY_ACTIVE
    if denylist and provider_id in denylist:
        return "blocked", _WITNESS_REASON_PROVIDER_IN_DENYLIST
    if allowlist and provider_id not in allowlist:
        return "blocked", _WITNESS_REASON_PROVIDER_OUTSIDE_ALLOWLIST
    return "allowed", _WITNESS_REASON_PASSES_POLICY


def runtime_emit_provider_execution_witness(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    event_log: Any,
    *,
    routing_model_id: str = "",
    assignment_phase: str = "initial",
) -> int | None:
    """Emit a ``provider_execution_witness`` event (slice 10D Route A v0).

    cgpro DESIGN_LOCK 2026-05-11: this event sits between the per-node
    ``model_assigned`` events and the first ``node_started`` to make the
    chain ``routing_chosen_model → policy_decision → per_node_assignments``
    visible in the runtime event log.

    Inputs:
    - ``routing_model_id`` — the model id the router originally chose
      (from the ``routing_decision`` event payload). The caller is
      responsible for plumbing this through; we don't re-read events.
    - ``ctx`` — must have ``assignments`` (idx → model_id) and
      ``topology`` set. ``ctx.provider_allowlist``/``ctx.provider_denylist``
      are read if present.
    - ``assignment_phase`` — ``"initial"`` (default) or ``"reroute"``.

    Returns the event sequence number or ``None`` if no event_log was
    provided.
    """
    if event_log is None:
        return None

    # Resolve allowlist / denylist from ctx if present. The CLI
    # adapter populates these as tuples; tests may pass lists.
    allowlist_raw = getattr(ctx, "provider_allowlist", None) or ()
    denylist_raw = getattr(ctx, "provider_denylist", None) or ()
    allowlist = tuple(str(p) for p in allowlist_raw if p)
    denylist = tuple(str(p) for p in denylist_raw if p)
    policy_active = bool(allowlist or denylist)

    # Resolve routing provider
    routing_provider_id = ""
    if routing_model_id:
        routing_provider_id = runtime_provider_id_for_model(
            pipeline, routing_model_id, ctx
        )
    routing_decision, routing_reason = _classify_provider_against_policy(
        routing_provider_id,
        allowlist=allowlist,
        denylist=denylist,
        policy_active=policy_active,
    )

    # Per-node assignments
    per_node: list[dict[str, Any]] = []
    if ctx.topology is not None:
        for idx in range(runtime_node_count(ctx.topology)):
            try:
                node = ctx.topology.get_node(idx)
            except (AttributeError, RuntimeError, TypeError):
                continue
            model_id = ctx.assignments.get(idx, getattr(node, "model_id", "") or "")
            node_role = getattr(node, "role", "") or f"node-{idx}"
            capabilities = list(runtime_node_capabilities(node))
            assigned_provider_id = ""
            if model_id:
                assigned_provider_id = runtime_provider_id_for_model(
                    pipeline, model_id, ctx
                )
            decision, reason = _classify_provider_against_policy(
                assigned_provider_id,
                allowlist=allowlist,
                denylist=denylist,
                policy_active=policy_active,
            )
            if not assigned_provider_id:
                reason = _WITNESS_REASON_ASSIGNMENT_PROVIDER_UNRESOLVED
            per_node.append(
                {
                    "node_id": str(idx),
                    "node_role": node_role,
                    "assigned_model_id": str(model_id or ""),
                    "assigned_provider_id": str(assigned_provider_id or ""),
                    "required_capabilities": capabilities,
                    "assignment_policy_decision": decision,
                    "assignment_policy_reason_code": reason,
                }
            )

    # Substitution summary — high-signal booleans + counts
    routing_model_distinct = bool(
        routing_model_id
        and any(
            assignment.get("assigned_model_id") != routing_model_id
            for assignment in per_node
        )
    )
    blocked_count = sum(
        1 for a in per_node if a.get("assignment_policy_decision") == "blocked"
    )
    allowed_count = sum(
        1 for a in per_node if a.get("assignment_policy_decision") == "allowed"
    )
    # Rust filter details (cgpro DESIGN_LOCKED v1_1, 2026-05-11):
    # ModelAssigner may expose a structured rejection list via
    # `last_filter_rejections()` (returns list of `(model_id,
    # reason_code)` tuples) and `last_filter_rejections_truncated()`
    # (bool). Read defensively — older Rust wheels don't have these
    # methods yet. Null/empty semantics:
    #   None       — Rust did not run a recording-aware path
    #   []         — Rust ran and rejected nothing
    #   non-empty  — Rust recorded the listed rejections
    rust_rejections, rust_truncated = _runtime_read_rust_filter_rejections(
        pipeline
    )
    substitution_summary = {
        "routing_model_distinct_from_assignments": routing_model_distinct,
        "routing_candidate_blocked_by_policy": routing_decision == "blocked",
        "executed_models_distinct_from_routing": routing_model_distinct,
        "assignment_count": len(per_node),
        "allowed_assignment_count": allowed_count,
        "blocked_assignment_count": blocked_count,
        "rust_filter_details_observed": rust_rejections is not None,
        "rust_filter_rejections": rust_rejections,
        "rust_filter_rejections_truncated": rust_truncated,
    }

    routing_payload = {
        "routing_source": str(getattr(ctx, "routing_source", "") or ""),
        "routing_model_id": str(routing_model_id or ""),
        "routing_provider_id": str(routing_provider_id or ""),
        "system": int(getattr(ctx, "system", 0) or 0),
        "domain": str(getattr(ctx, "domain", "") or ""),
        "confidence": getattr(ctx, "confidence", None),
    }

    policy_payload = {
        "active": policy_active,
        "source": ("cli" if policy_active else "none"),
        "allowlist": list(allowlist),
        "denylist": list(denylist),
        "routing_candidate_decision": routing_decision,
        "routing_candidate_reason_code": routing_reason,
    }

    seq = event_log.emit_provider_execution_witness(
        witness_schema_version="v0",
        assignment_phase=assignment_phase,
        routing=routing_payload,
        policy=policy_payload,
        per_node_assignments=per_node,
        substitution_summary=substitution_summary,
    )

    # Audit-sidecar ref (best-effort)
    try:
        from sage.pipeline_v2 import learning_side_effects as lse_mod

        lse_mod.store_event_ref(ctx, "provider_execution_witness", event_log)
    except Exception:  # noqa: BLE001 - audit sidecar refs are best-effort
        pass

    return seq


def runtime_final_status(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext | None",
) -> "RunStatus":
    """Classify a run's final status from `ctx.result` ↔ BUDGET_EXCEEDED_RESULT."""
    from sage.pipeline import BUDGET_EXCEEDED_RESULT

    if ctx is None:
        return "failure"
    if ctx.result == BUDGET_EXCEEDED_RESULT:
        return "budget_exceeded"
    return "success" if ctx.result else "failure"


def runtime_final_node_count(
    pipeline: "CognitiveOrchestrationPipeline",  # noqa: ARG001 - signature symmetry
    ctx: "PipelineContext | None",
) -> int:
    """Return the topology node count for the final-result event payload."""
    if ctx is None or ctx.topology is None:
        return 0
    return runtime_node_count(ctx.topology)


__all__ = [
    "bandit_reason_from_exception",
    "emit",
    "emit_bandit_attribution_mismatch",
    "runtime_edge_summary",
    "runtime_edge_type",
    "runtime_emit_model_assigned",
    "runtime_emit_provider_execution_witness",
    "runtime_emit_topology_selected",
    "runtime_final_node_count",
    "runtime_final_status",
    "runtime_graph_digest",
    "runtime_node_capabilities",
    "runtime_node_count",
    "runtime_node_summary",
    "runtime_provider_id_for_model",
]
