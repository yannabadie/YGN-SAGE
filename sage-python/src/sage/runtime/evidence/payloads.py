from __future__ import annotations

import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from sage.runtime.evidence.errors import EvidenceError


FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "stdout",
        "stderr",
        "raw_stdout",
        "raw_stderr",
        "output",
        "raw_output",
        "content",
        "raw_content",
        "patch",
        "raw_patch",
        "diff",
        "raw_diff",
        "final_answer",
        "message",
        "traceback",
    }
)

DEFAULT_MAX_STRING_LENGTH = 256

PAYLOAD_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "tool_execution": frozenset(
        {
            "exit_code",
            "timeout_sec",
            "timed_out",
            "tool_error_class",
            "fatal_scope",
            "artifact_hash",
            "duration_ms",
        }
    ),
    "test_parser": frozenset(
        {
            "framework",
            "parser_id",
            "suite_id",
            "passed_count",
            "failed_count",
            "skipped_count",
            "error_count",
            "duration_ms",
        }
    ),
    "diff_verifier": frozenset(
        {
            "patch_apply_ok",
            "mismatch_count",
            "mismatch_kinds",
            "repair_stage",
            "patch_hash",
        }
    ),
    "formal_verifier": frozenset(
        {
            "obligation_id",
            "obligation_type",
            "verifier_id",
            "solver_status",
            "encoding",
            "model_hash",
            "counterexample_hash",
        }
    ),
    "code_node_return": frozenset(
        {
            "schema_id",
            "valid",
            "return_hash",
            "declared_fields",
        }
    ),
    "planner_decision": frozenset(
        {
            "decision_type",
            "topology_id",
            "graph_digest",
            "node_count",
        }
    ),
}

PAYLOAD_MAX_STRING_LENGTHS: dict[str, int] = {
    "framework": 64,
    "parser_id": 96,
    "suite_id": 128,
    "tool_error_class": 128,
    "fatal_scope": 64,
    "artifact_hash": 128,
    "patch_hash": 128,
    "repair_stage": 96,
    "obligation_id": 128,
    "obligation_type": 96,
    "verifier_id": 96,
    "solver_status": 64,
    "encoding": 128,
    "model_hash": 128,
    "counterexample_hash": 128,
    "schema_id": 128,
    "return_hash": 128,
    "decision_type": 96,
    "topology_id": 128,
    "graph_digest": 128,
}


def validate_payload(
    producer: str,
    delta_kind: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a producer payload before it can enter RuntimeDelta."""
    del delta_kind
    if not isinstance(payload, Mapping):
        raise EvidenceError("payload must be a mapping")
    allowed = PAYLOAD_ALLOWED_KEYS.get(producer)
    if allowed is None:
        raise EvidenceError(f"unknown producer payload schema: {producer!r}")
    for key, value in payload.items():
        if not isinstance(key, str):
            raise EvidenceError("payload keys must be strings")
        if key in FORBIDDEN_KEYS:
            raise EvidenceError(f"payload key {key!r} is forbidden")
        if key not in allowed:
            raise EvidenceError(
                f"payload key {key!r} is not allowed for producer {producer!r}"
            )
        _validate_json_value(value, top_key=key, path=f"payload.{key}")
    return payload


def deep_freeze_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _deep_freeze_value(value) for key, value in payload.items()}
    )


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for RuntimeDelta hashing."""
    return json.dumps(
        _plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_evidence_hash(
    *,
    schema_version: str,
    producer: str,
    delta_kind: str,
    polarity: str,
    source_id: str,
    payload: Mapping[str, Any],
) -> str:
    envelope = {
        "schema_version": schema_version,
        "producer": producer,
        "delta_kind": delta_kind,
        "polarity": polarity,
        "source_id": source_id,
        "payload": payload,
    }
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


def _validate_json_value(value: Any, *, top_key: str, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            max_len = PAYLOAD_MAX_STRING_LENGTHS.get(
                top_key,
                DEFAULT_MAX_STRING_LENGTH,
            )
            if len(value) > max_len:
                raise EvidenceError(
                    f"payload string {path} exceeds max length {max_len}"
                )
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"payload float {path} must be finite")
        return
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise EvidenceError(f"payload mapping key {path} must be a string")
            if nested_key in FORBIDDEN_KEYS:
                raise EvidenceError(f"payload key {nested_key!r} is forbidden")
            if len(nested_key) > DEFAULT_MAX_STRING_LENGTH:
                raise EvidenceError(f"payload key {path}.{nested_key} is too long")
            _validate_json_value(
                nested_value,
                top_key=top_key,
                path=f"{path}.{nested_key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _validate_json_value(item, top_key=top_key, path=f"{path}[{idx}]")
        return
    raise EvidenceError(
        f"payload value {path} has unsupported JSON type {type(value).__name__}"
    )


def _deep_freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_value(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value
