"""Audit helpers for Learning Side-Effect Ledger v0."""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from sage.runtime.credit_assignment import emit_learning_side_effect
from sage.runtime.oracle import oracle_enabled

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def store_event_ref(ctx: "PipelineContext", event_type: str, event_log: Any) -> None:
    """Store the last RuntimeEventLog ref on ctx for sidecar linking."""
    ref_getter = getattr(event_log, "last_event_ref", None)
    if ref_getter is None:
        return
    ref = ref_getter()
    if ref is not None and ref.event_type == event_type:
        ctx.runtime_event_refs[event_type] = ref.to_dict()


def emit_decision(
    pipeline: "CognitiveOrchestrationPipeline",
    ctx: "PipelineContext",
    *,
    side_effect: str,
    decision: str,
    reason_code: str,
    attempted: bool,
    quality: float | None = None,
    result_summary: dict[str, Any] | None = None,
) -> None:
    """Emit one audit-only learning side-effect decision."""
    oracle_on = oracle_enabled()
    verdict = getattr(ctx, "oracle_verdict", None)
    oracle_trainable = bool(verdict.trainable) if verdict is not None else False
    record = {
        "parent_event_refs": _parent_refs(ctx),
        "oracle_verdict_ref": _oracle_ref(ctx),
        "policy_ref": _policy_ref(ctx),
        "subject": _subject(ctx),
        "side_effect": side_effect,
        "decision": decision,
        "reason_code": reason_code,
        "attempted": attempted,
        "gate": {
            "oracle_enabled": oracle_on,
            "oracle_trainable": oracle_trainable,
            "allow_training_updates": (not oracle_on) or oracle_trainable,
            "quality_source": _quality_source(oracle_on, verdict, quality),
        },
        "metrics": {
            "quality": quality,
            "cost_usd": _finite_or_none(getattr(ctx, "cost", None)),
            "latency_ms": _finite_or_none(getattr(ctx, "latency_ms", None)),
        },
        "result_summary": result_summary or {"status": "observed", "redacted": True},
    }
    emit_learning_side_effect(record)


def reason_for_blocked_or_skipped(ctx: "PipelineContext", quality: float | None) -> str:
    oracle_on = oracle_enabled()
    verdict = getattr(ctx, "oracle_verdict", None)
    if oracle_on and verdict is None:
        return "oracle_missing"
    if oracle_on and verdict is not None and not verdict.trainable:
        return "oracle_untrainable"
    if not oracle_on and quality is not None:
        return "oracle_disabled_legacy_quality_path"
    if quality is None:
        return "quality_unavailable"
    return "not_applicable"


def _parent_refs(ctx: "PipelineContext") -> list[dict[str, Any]]:
    ordered = (
        "routing_decision",
        "topology_selected",
        "model_assigned",
        "final_result",
        "oracle_verdict",
    )
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for event_type in ordered:
        ref = ctx.runtime_event_refs.get(event_type)
        if not ref:
            continue
        key = (str(ref.get("event_type")), int(ref.get("seq", -1)))
        if key not in seen:
            refs.append(dict(ref))
            seen.add(key)
    return refs


def _oracle_ref(ctx: "PipelineContext") -> dict[str, Any] | None:
    verdict = getattr(ctx, "oracle_verdict", None)
    ref = ctx.runtime_event_refs.get("oracle_verdict")
    if verdict is None or ref is None:
        return None
    return {
        "seq": ref["seq"],
        "payload_hash": ref["payload_hash"],
        "trainable": verdict.trainable,
        "verdict_source": verdict.verdict_source,
        "quality_label": verdict.quality_label,
        "score": verdict.score,
        "evidence_hashes": [
            item.evidence_hash for item in verdict.evidence if item.evidence_hash
        ],
    }


def _policy_ref(ctx: "PipelineContext") -> dict[str, Any]:
    routing_ref = ctx.runtime_event_refs.get("routing_decision")
    candidate_basis = {
        "bandit_model_id": getattr(ctx, "bandit_model_id", ""),
        "bandit_template": getattr(ctx, "bandit_template", ""),
        "executed_model_id": getattr(ctx, "executed_model_id", ""),
        "executed_template": getattr(ctx, "executed_template", ""),
    }
    return {
        "decision_id": getattr(ctx, "bandit_decision_id", "") or None,
        "routing_decision_ref": routing_ref,
        "policy_snapshot_hash": _hash_json(candidate_basis),
        "candidate_set_hash": _hash_json(
            {
                "assignments": getattr(ctx, "assignments", {}),
                "provider_hints": getattr(ctx, "provider_hints", {}),
            }
        ),
        "selection_probability": None,
        "selection_probability_reason": "thompson_sampling_propensity_not_logged",
    }


def _subject(ctx: "PipelineContext") -> dict[str, Any]:
    provider_id = None
    provider_hints = getattr(ctx, "provider_hints", {})
    if provider_hints:
        provider_id = ",".join(str(value) for value in sorted(set(provider_hints.values())))
    return {
        "model_id": getattr(ctx, "executed_model_id", "")
        or getattr(ctx, "bandit_model_id", "")
        or None,
        "provider_id": provider_id,
        "template": getattr(ctx, "executed_template", "")
        or getattr(ctx, "bandit_template", "")
        or None,
        "topology_id": getattr(ctx, "topology_id", "") or None,
        "node_id": None,
        "tool_path_hash": None,
    }


def _quality_source(oracle_on: bool, verdict: Any, quality: float | None) -> str:
    if oracle_on and verdict is not None:
        return "oracle"
    if not oracle_on and quality is not None:
        return "legacy_quality"
    return "none"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
