"""Runtime-event helpers for `CognitiveOrchestrationPipeline`.

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
real home for the EventBus emission helpers and the 11 `_runtime_emit_*`
builders carved out of `CognitiveOrchestrationPipeline`.

Step A3 landed `emit` (basic). Step B4 (this commit) lands the rest:

  - `emit_bandit_attribution_mismatch` — invariant 6 mismatch event
  - `bandit_reason_from_exception` — string-parsing static (pure)
  - `runtime_node_count` / `runtime_edge_type` /
    `runtime_node_capabilities` / `runtime_graph_digest` — pure
    helpers (no `pipeline` arg)
  - `runtime_edge_summary` / `runtime_node_summary` — read pipeline
    + topology / ctx state
  - `runtime_provider_id_for_model` — uses `pipeline.provider_pool`
    + `pipeline.llm_config`
  - `runtime_emit_topology_selected` / `runtime_emit_model_assigned` —
    canonical event-emission entry points consumed by `_run_internal`
    and the topology controller's reroute / rebuild paths
  - `runtime_final_status` / `runtime_final_node_count` — final-result
    classification

cgpro Q4 garde-fou: NO top-level imports of `sage.pipeline` from this
module — TYPE_CHECKING only. The `BUDGET_EXCEEDED_RESULT` and
`_BANDIT_ATTRIBUTION_REASON_CODES` constants are imported LAZILY
inside the functions that need them so the `pipeline_v2/__init__.py`
PEP 562 lazy conversion (Step E0) does not regress.

cgpro Q7 garde-fou: order in `_run_internal` (final_result →
oracle_verdict → learn → run_frame_summary) is NOT touched by this
module. Step B4 only carves out the helpers — `_run_internal` body
relocation is Step D.

Logger uses ``sage.pipeline`` per cgpro Q7 trap "logger name drift" —
modules carved out of `pipeline.py` keep the legacy logger name so
trace-grep continuity is preserved across the refactor.
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
    """Resolve a model_id to its provider_id using assignments + provider_pool + default config."""
    for node_idx, assigned_model in ctx.assignments.items():
        if assigned_model == model_id and node_idx in ctx.provider_hints:
            return str(ctx.provider_hints[node_idx])
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
        if run_frame_builder is not None:
            run_frame_builder.record_model_assigned(
                seq=seq,
                node_id=str(idx),
                node_role=node_role,
                model_id=model_id,
                provider_id=provider_id,
                required_capabilities=capabilities,
            )


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
    "runtime_emit_topology_selected",
    "runtime_final_node_count",
    "runtime_final_status",
    "runtime_graph_digest",
    "runtime_node_capabilities",
    "runtime_node_count",
    "runtime_node_summary",
    "runtime_provider_id_for_model",
]
