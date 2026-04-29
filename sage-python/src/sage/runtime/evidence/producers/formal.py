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


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["formal_verifier"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["formal_verifier"]
POLARITY_RULES = _POLARITY_RULES["formal_verifier"]

_DEFAULT_POLARITY: dict[str, DeltaPolarity] = {
    "obligation_proved": "positive",
    "obligation_refuted": "negative",
    "counterexample_found": "negative",
    "obligation_unknown": "unknown",
    "verifier_unavailable": "neutral",
    "assumption_invalidated": "negative",
}


def produce_formal_verifier_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    delta_kind: str,
    obligation_id: str = "",
    obligation_type: str = "",
    verifier_id: str = "",
    solver_status: str = "",
    encoding: str = "",
    model_hash: str | None = None,
    counterexample_hash: str | None = None,
    polarity: DeltaPolarity | None = None,
) -> DeltaProducerResult:
    """Emit obligation-semantic formal facts, not raw SAT/UNSAT rewards."""
    try:
        if delta_kind not in ALLOWED_DELTA_KINDS:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason=f"unknown formal delta_kind: {delta_kind!r}",
            )
        if delta_kind != "verifier_unavailable" and not obligation_id:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason="obligation_id is required for formal facts",
            )
        payload: dict[str, Any] = {}
        for key, value in (
            ("obligation_id", obligation_id),
            ("obligation_type", obligation_type),
            ("verifier_id", verifier_id),
            ("solver_status", solver_status),
            ("encoding", encoding),
            ("model_hash", model_hash),
            ("counterexample_hash", counterexample_hash),
        ):
            if value is not None and value != "":
                payload[key] = value
        return DeltaProducerResult(
            deltas=(
                RuntimeDelta(
                    schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                    producer="formal_verifier",
                    delta_kind=delta_kind,
                    polarity=polarity or _DEFAULT_POLARITY[delta_kind],
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
