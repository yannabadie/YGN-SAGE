"""Learning Side-Effect Ledger v0 schema.

This sidecar is audit output. It records learning side-effect decisions made by
the existing runtime gates and must never authorize learning by itself.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "learning_side_effect.v0"

SIDE_EFFECTS = frozenset(
    {
        "bandit_record_outcome",
        "bandit_cancel_pending",
        "map_elites_record_outcome",
        "online_evolution_should_evolve",
        "online_evolution_evolve",
        "training_memory_consolidate",
    }
)

DECISIONS = frozenset({"allowed", "blocked", "skipped", "failed"})
QUALITY_SOURCES = frozenset({"oracle", "legacy_quality", "none"})

REASON_CODES = frozenset(
    {
        "oracle_trainable",
        "oracle_untrainable",
        "oracle_missing",
        "oracle_disabled_legacy_quality_path",
        "quality_unavailable",
        "no_pending_decision",
        "multi_node_ambiguous",
        "model_mismatch",
        "template_mismatch",
        "decision_unknown",
        "recorder_instance_mismatch",
        "record_failed",
        "record_succeeded",
        "safety_cancel_untrainable",
        "safety_cancel_ambiguous",
        "safety_cancel_mismatch",
        "safety_cancel_fallback",
        "not_applicable",
        "engine_missing",
        "topology_missing",
        "engine_method_missing",
        "should_evolve_false",
        "not_scheduled",
        "call_failed",
    }
)

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "seq",
        "timestamp_ns",
        "run_id",
        "trace_id",
        "task_hash",
        "parent_event_refs",
        "oracle_verdict_ref",
        "policy_ref",
        "subject",
        "side_effect",
        "decision",
        "reason_code",
        "attempted",
        "gate",
        "metrics",
        "result_summary",
        "redaction_state",
        "prev_record_hash",
        "record_hash",
    }
)

FORBIDDEN_RESULT_KEYS = frozenset(
    {
        "raw_prompt",
        "prompt",
        "task",
        "task_text",
        "raw_output",
        "output",
        "result",
    }
)

MAX_RESULT_SUMMARY_BYTES = 2048
POLICY_REF_FIELDS = frozenset(
    {
        "decision_id",
        "routing_decision_ref",
        "policy_snapshot_hash",
        "candidate_set_hash",
        "selection_probability",
        "selection_probability_reason",
    }
)
SUBJECT_FIELDS = frozenset(
    {
        "model_id",
        "provider_id",
        "template",
        "topology_id",
        "node_id",
        "tool_path_hash",
    }
)
GATE_FIELDS = frozenset(
    {
        "oracle_enabled",
        "oracle_trainable",
        "allow_training_updates",
        "quality_source",
    }
)
METRICS_FIELDS = frozenset({"quality", "cost_usd", "latency_ms"})


class LearningSideEffectSchemaError(ValueError):
    """Raised when a ledger record fails schema or hash validation."""


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def validate_record_shape(record: Mapping[str, Any]) -> None:
    keys = set(record)
    missing = sorted(REQUIRED_FIELDS - keys)
    extras = sorted(keys - REQUIRED_FIELDS)
    if missing:
        raise LearningSideEffectSchemaError(f"missing required field(s): {missing}")
    if extras:
        raise LearningSideEffectSchemaError(f"unknown field(s): {extras}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise LearningSideEffectSchemaError("invalid schema_version")
    if not isinstance(record["seq"], int) or record["seq"] < 0:
        raise LearningSideEffectSchemaError("seq must be a non-negative int")
    if not isinstance(record["timestamp_ns"], int) or record["timestamp_ns"] <= 0:
        raise LearningSideEffectSchemaError("timestamp_ns must be a positive int")
    for key in ("run_id", "trace_id", "task_hash", "redaction_state"):
        if not isinstance(record[key], str):
            raise LearningSideEffectSchemaError(f"{key} must be a string")
    if record["redaction_state"] != "redacted":
        raise LearningSideEffectSchemaError("redaction_state must be redacted")
    if record["side_effect"] not in SIDE_EFFECTS:
        raise LearningSideEffectSchemaError("invalid side_effect")
    if record["decision"] not in DECISIONS:
        raise LearningSideEffectSchemaError("invalid decision")
    if record["reason_code"] not in REASON_CODES:
        raise LearningSideEffectSchemaError("invalid reason_code")
    if not isinstance(record["attempted"], bool):
        raise LearningSideEffectSchemaError("attempted must be bool")
    if not isinstance(record["parent_event_refs"], list):
        raise LearningSideEffectSchemaError("parent_event_refs must be a list")
    for parent in record["parent_event_refs"]:
        _validate_event_ref(parent, "parent_event_refs[]")
    if record["oracle_verdict_ref"] is not None:
        if not isinstance(record["oracle_verdict_ref"], Mapping):
            raise LearningSideEffectSchemaError("oracle_verdict_ref must be object/null")
        _validate_oracle_ref(record["oracle_verdict_ref"])
    for key in ("policy_ref", "subject", "gate", "metrics", "result_summary"):
        if not isinstance(record[key], Mapping):
            raise LearningSideEffectSchemaError(f"{key} must be an object")
    _validate_policy_ref(record["policy_ref"])
    _validate_subject(record["subject"])
    _validate_gate(record["gate"])
    _validate_metrics(record["metrics"])
    _validate_result_summary(record["result_summary"])
    prev_hash = record["prev_record_hash"]
    if prev_hash is not None and not isinstance(prev_hash, str):
        raise LearningSideEffectSchemaError("prev_record_hash must be string/null")
    if not isinstance(record["record_hash"], str):
        raise LearningSideEffectSchemaError("record_hash must be a string")


def record_hash_input(record: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(record)
    data.pop("record_hash", None)
    return data


def _validate_event_ref(value: Any, prefix: str) -> None:
    if not isinstance(value, Mapping):
        raise LearningSideEffectSchemaError(f"{prefix} must be an object")
    allowed = {"event_type", "seq", "payload_hash"}
    extras = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if extras or missing:
        raise LearningSideEffectSchemaError(
            f"{prefix} invalid keys missing={missing} extras={extras}"
        )
    if not isinstance(value["event_type"], str):
        raise LearningSideEffectSchemaError(f"{prefix}.event_type must be string")
    if not isinstance(value["seq"], int) or value["seq"] < 0:
        raise LearningSideEffectSchemaError(f"{prefix}.seq must be non-negative int")
    if not isinstance(value["payload_hash"], str):
        raise LearningSideEffectSchemaError(f"{prefix}.payload_hash must be string")


def _validate_oracle_ref(value: Mapping[str, Any]) -> None:
    for key in (
        "seq",
        "payload_hash",
        "trainable",
        "verdict_source",
        "quality_label",
        "score",
        "evidence_hashes",
    ):
        if key not in value:
            raise LearningSideEffectSchemaError(f"oracle_verdict_ref missing {key}")
    if not isinstance(value["seq"], int) or value["seq"] < 0:
        raise LearningSideEffectSchemaError("oracle_verdict_ref.seq invalid")
    if not isinstance(value["payload_hash"], str):
        raise LearningSideEffectSchemaError("oracle_verdict_ref.payload_hash invalid")
    if not isinstance(value["trainable"], bool):
        raise LearningSideEffectSchemaError("oracle_verdict_ref.trainable invalid")
    if value["score"] is not None and not _finite_number(value["score"]):
        raise LearningSideEffectSchemaError("oracle_verdict_ref.score invalid")
    if not isinstance(value["evidence_hashes"], list) or any(
        not isinstance(item, str) for item in value["evidence_hashes"]
    ):
        raise LearningSideEffectSchemaError("oracle_verdict_ref.evidence_hashes invalid")


def _validate_policy_ref(value: Mapping[str, Any]) -> None:
    _validate_exact_keys(value, POLICY_REF_FIELDS, "policy_ref")
    if value["decision_id"] is not None and not isinstance(value["decision_id"], str):
        raise LearningSideEffectSchemaError("policy_ref.decision_id invalid")
    if value["routing_decision_ref"] is not None:
        _validate_event_ref(value["routing_decision_ref"], "policy_ref.routing_decision_ref")
    for key in ("policy_snapshot_hash", "candidate_set_hash"):
        if not _is_sha256(value[key]):
            raise LearningSideEffectSchemaError(f"policy_ref.{key} invalid")
    probability = value["selection_probability"]
    if probability is not None and not _finite_number(probability):
        raise LearningSideEffectSchemaError("policy_ref.selection_probability invalid")
    if not isinstance(value["selection_probability_reason"], str):
        raise LearningSideEffectSchemaError(
            "policy_ref.selection_probability_reason invalid"
        )


def _validate_subject(value: Mapping[str, Any]) -> None:
    _validate_exact_keys(value, SUBJECT_FIELDS, "subject")
    for key in SUBJECT_FIELDS:
        if value[key] is not None and not isinstance(value[key], str):
            raise LearningSideEffectSchemaError(f"subject.{key} invalid")


def _validate_gate(value: Mapping[str, Any]) -> None:
    _validate_exact_keys(value, GATE_FIELDS, "gate")
    for key in ("oracle_enabled", "oracle_trainable", "allow_training_updates"):
        if not isinstance(value[key], bool):
            raise LearningSideEffectSchemaError(f"gate.{key} must be bool")
    if value["quality_source"] not in QUALITY_SOURCES:
        raise LearningSideEffectSchemaError("gate.quality_source invalid")


def _validate_metrics(value: Mapping[str, Any]) -> None:
    _validate_exact_keys(value, METRICS_FIELDS, "metrics")
    for key in ("quality", "cost_usd", "latency_ms"):
        item = value.get(key)
        if item is not None and not _finite_number(item):
            raise LearningSideEffectSchemaError(f"metrics.{key} must be finite/null")


def _validate_result_summary(value: Mapping[str, Any]) -> None:
    forbidden = sorted(set(value) & FORBIDDEN_RESULT_KEYS)
    if forbidden:
        raise LearningSideEffectSchemaError(
            f"result_summary contains forbidden key(s): {forbidden}"
        )
    size = len(canonical_json(value).encode("utf-8", errors="replace"))
    if size > MAX_RESULT_SUMMARY_BYTES:
        raise LearningSideEffectSchemaError("result_summary exceeds max bytes")


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _validate_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    prefix: str,
) -> None:
    missing = sorted(expected - set(value))
    extras = sorted(set(value) - expected)
    if missing or extras:
        raise LearningSideEffectSchemaError(
            f"{prefix} invalid keys missing={missing} extras={extras}"
        )
