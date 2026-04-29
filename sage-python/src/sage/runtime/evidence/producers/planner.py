from __future__ import annotations

from sage.runtime.evidence.delta import (
    RUNTIME_DELTA_SCHEMA_VERSION,
    DeltaProducerResult,
    RuntimeDelta,
    _DELTA_KIND_TABLE,
    _POLARITY_RULES,
)
from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import PAYLOAD_ALLOWED_KEYS


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["planner_decision"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["planner_decision"]
POLARITY_RULES = _POLARITY_RULES["planner_decision"]


def produce_planner_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    decision_type: str,
    topology_id: str | None = None,
    graph_digest: str | None = None,
    node_count: int | None = None,
) -> DeltaProducerResult:
    """Produce planner structural facts only; never planner free text."""
    try:
        if decision_type not in ALLOWED_DELTA_KINDS:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason=f"unknown planner decision_type: {decision_type!r}",
            )
        payload: dict[str, object] = {"decision_type": decision_type}
        if topology_id is not None:
            payload["topology_id"] = topology_id
        if graph_digest is not None:
            payload["graph_digest"] = graph_digest
        if node_count is not None:
            payload["node_count"] = int(node_count)
        return DeltaProducerResult(
            deltas=(
                RuntimeDelta(
                    schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                    producer="planner_decision",
                    delta_kind=decision_type,
                    polarity="neutral",
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
