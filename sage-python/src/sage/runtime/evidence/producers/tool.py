from __future__ import annotations

from typing import Any

from sage.runtime.evidence.delta import (
    RUNTIME_DELTA_SCHEMA_VERSION,
    DeltaPolarity,
    DeltaProducerResult,
    RuntimeDelta,
    _DELTA_KIND_TABLE,
    _POLARITY_RULES,
)
from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import PAYLOAD_ALLOWED_KEYS


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["tool_execution"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["tool_execution"]
POLARITY_RULES = _POLARITY_RULES["tool_execution"]

# cgpro 2026-04-29 R6.1a verify push-back: ToolOracle may train fail on a
# `fatal_failure` delta only when the failure deterministically invalidates
# the claimed task output. Generic agent-loop tool exceptions (incidental,
# not tied to the artifact under test) must NOT become trainable fail.
# Producers MUST tag every fatal_failure with one of the values below so
# the oracle can gate on `fatal_scope == "claimed_task_output"`.
FATAL_SCOPES: frozenset[str] = frozenset(
    {"claimed_task_output", "incidental_tool_call", "unknown"}
)


def produce_tool_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    exit_code: int,
    timed_out: bool,
    duration_ms: float,
    tool_error_class: str = "",
    fatal_scope: str = "unknown",
    artifact_hash: str | None = None,
    timeout_sec: float | None = None,
) -> DeltaProducerResult:
    """Produce tool_execution facts from structured metadata only."""
    try:
        normalized_scope = (fatal_scope or "unknown").strip().lower()
        if normalized_scope not in FATAL_SCOPES:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason=(
                    f"fatal_scope {fatal_scope!r} is not one of "
                    f"{sorted(FATAL_SCOPES)}"
                ),
            )
        payload: dict[str, Any] = {
            "exit_code": int(exit_code),
            "timed_out": bool(timed_out),
            "tool_error_class": str(tool_error_class or ""),
            "duration_ms": float(duration_ms),
        }
        if artifact_hash is not None:
            payload["artifact_hash"] = str(artifact_hash)
        if timeout_sec is not None:
            payload["timeout_sec"] = float(timeout_sec)

        if timed_out:
            delta_kind = "timed_out"
            polarity: DeltaPolarity = "neutral"
        elif _is_unavailable(exit_code=exit_code, tool_error_class=tool_error_class):
            delta_kind = "unavailable"
            polarity = "neutral"
        elif int(exit_code) == 0:
            delta_kind = "exit_zero"
            polarity = "positive"
        elif tool_error_class:
            delta_kind = "fatal_failure"
            polarity = "negative"
            payload["fatal_scope"] = normalized_scope
        else:
            delta_kind = "exit_nonzero"
            polarity = "neutral"

        return DeltaProducerResult(
            deltas=(
                RuntimeDelta(
                    schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                    producer="tool_execution",
                    delta_kind=delta_kind,
                    polarity=polarity,
                    confidence=1.0,
                    run_id=run_id,
                    node_run_id=node_run_id,
                    event_seq=event_seq,
                    source_id=source_id,
                    payload=payload,
                ),
            )
        )
    except (EvidenceError, TypeError, ValueError) as exc:
        return DeltaProducerResult(deltas=(), rejected_reason=str(exc))


def _is_unavailable(*, exit_code: int, tool_error_class: str) -> bool:
    lowered = (tool_error_class or "").lower().replace(" ", "")
    return int(exit_code) == 127 or lowered in {
        "toolunavailable",
        "toolnotfound",
        "unknown_tool",
        "unknowntool",
    }
