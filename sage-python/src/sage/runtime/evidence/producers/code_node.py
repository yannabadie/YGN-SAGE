from __future__ import annotations

import hashlib
from typing import Any, Mapping

from sage.runtime.evidence.delta import (
    RUNTIME_DELTA_SCHEMA_VERSION,
    DeltaProducerResult,
    RuntimeDelta,
    _DELTA_KIND_TABLE,
    _POLARITY_RULES,
)
from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import PAYLOAD_ALLOWED_KEYS, canonical_json


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["code_node_return"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["code_node_return"]
POLARITY_RULES = _POLARITY_RULES["code_node_return"]


def produce_code_node_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    return_value: Any,
    return_schema: Mapping[str, Any] | None,
) -> DeltaProducerResult:
    """Produce code-node return facts only when an explicit schema exists."""
    try:
        if not return_schema:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason="return_schema is required",
            )
        schema_id = str(return_schema.get("schema_id", "") or "")
        declared_fields = _declared_fields(return_schema)
        if not schema_id or not declared_fields:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason="return_schema must declare schema_id and fields",
            )
        valid = isinstance(return_value, Mapping) and all(
            field in return_value for field in declared_fields
        )
        return_hash = hashlib.sha256(
            canonical_json(return_value).encode("utf-8")
        ).hexdigest()
        return DeltaProducerResult(
            deltas=(
                RuntimeDelta(
                    schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                    producer="code_node_return",
                    delta_kind=(
                        "structured_return_valid"
                        if valid
                        else "structured_return_invalid"
                    ),
                    polarity="positive" if valid else "negative",
                    confidence=1.0,
                    run_id=run_id,
                    node_run_id=node_run_id,
                    event_seq=event_seq,
                    source_id=source_id,
                    payload={
                        "schema_id": schema_id,
                        "valid": valid,
                        "return_hash": return_hash,
                        "declared_fields": tuple(declared_fields),
                    },
                ),
            )
        )
    except (EvidenceError, TypeError, ValueError) as exc:
        return DeltaProducerResult(deltas=(), rejected_reason=str(exc))


def _declared_fields(return_schema: Mapping[str, Any]) -> tuple[str, ...]:
    fields = return_schema.get("fields")
    if isinstance(fields, Mapping):
        return tuple(str(key) for key in fields)
    if isinstance(fields, (list, tuple)):
        return tuple(str(item) for item in fields)
    properties = return_schema.get("properties")
    if isinstance(properties, Mapping):
        return tuple(str(key) for key in properties)
    return ()
