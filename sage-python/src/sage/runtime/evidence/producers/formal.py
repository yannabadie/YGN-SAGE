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

# cgpro 2026-04-29 R6.1a verify push-back: trainable formal kinds must carry
# complete obligation semantics (obligation_id + verifier_id + encoding +
# solver_status), and solver_status must match the kind's proof/refutation
# direction. Incomplete or inconsistent deltas are rejected by the producer
# and re-checked by the oracle as defense-in-depth.
_TRAINABLE_KINDS: frozenset[str] = frozenset(
    {"obligation_proved", "obligation_refuted", "counterexample_found"}
)
_KIND_TO_REQUIRED_SOLVER_STATUS: dict[str, frozenset[str]] = {
    "obligation_proved": frozenset({"unsat"}),
    "obligation_refuted": frozenset({"sat"}),
    "counterexample_found": frozenset({"sat"}),
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
        # Trainable formal kinds (obligation_proved/refuted/counterexample_found)
        # must carry complete obligation semantics: verifier_id, encoding,
        # solver_status, and solver_status must match the kind's direction.
        if delta_kind in _TRAINABLE_KINDS:
            normalized_status = (solver_status or "").strip().lower()
            if not verifier_id:
                return DeltaProducerResult(
                    deltas=(),
                    rejected_reason=(
                        f"verifier_id is required for trainable formal kind "
                        f"{delta_kind!r}"
                    ),
                )
            if not encoding:
                return DeltaProducerResult(
                    deltas=(),
                    rejected_reason=(
                        f"encoding is required for trainable formal kind "
                        f"{delta_kind!r}"
                    ),
                )
            if not normalized_status:
                return DeltaProducerResult(
                    deltas=(),
                    rejected_reason=(
                        f"solver_status is required for trainable formal kind "
                        f"{delta_kind!r}"
                    ),
                )
            allowed_status = _KIND_TO_REQUIRED_SOLVER_STATUS[delta_kind]
            if normalized_status not in allowed_status:
                return DeltaProducerResult(
                    deltas=(),
                    rejected_reason=(
                        f"solver_status {solver_status!r} is inconsistent with "
                        f"delta_kind {delta_kind!r} (expected one of "
                        f"{sorted(allowed_status)})"
                    ),
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
