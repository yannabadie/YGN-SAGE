from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, cast

from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import (
    PAYLOAD_ALLOWED_KEYS,
    compute_evidence_hash,
    deep_freeze_payload,
    validate_payload,
)


RUNTIME_DELTA_SCHEMA_VERSION: Literal["0"] = "0"

ProducerName = Literal[
    "tool_execution",
    "test_parser",
    "diff_verifier",
    "formal_verifier",
    "code_node_return",
    "planner_decision",
]

DeltaPolarity = Literal["positive", "negative", "neutral", "unknown"]

PRODUCERS: tuple[ProducerName, ...] = (
    "tool_execution",
    "test_parser",
    "diff_verifier",
    "formal_verifier",
    "code_node_return",
    "planner_decision",
)

POLARITIES: tuple[DeltaPolarity, ...] = (
    "positive",
    "negative",
    "neutral",
    "unknown",
)

_DELTA_KIND_TABLE: dict[ProducerName, frozenset[str]] = {
    "tool_execution": frozenset(
        {
            "exit_zero",
            "exit_nonzero",
            "fatal_failure",
            "timed_out",
            "unavailable",
        }
    ),
    "test_parser": frozenset(
        {
            "tests_passed",
            "tests_failed",
            "tests_partial",
            "parse_failed",
        }
    ),
    "diff_verifier": frozenset(
        {
            "patch_applied",
            "patch_failed",
            "context_mismatch",
            "hunk_header_mismatch",
            "repair_accepted",
            "repair_rejected",
        }
    ),
    "formal_verifier": frozenset(
        {
            "obligation_proved",
            "obligation_refuted",
            "counterexample_found",
            "obligation_unknown",
            "verifier_unavailable",
            "assumption_invalidated",
        }
    ),
    "code_node_return": frozenset(
        {
            "structured_return_valid",
            "structured_return_invalid",
        }
    ),
    "planner_decision": frozenset(
        {
            "topology_selected",
            "decomposition_applied",
        }
    ),
}

_POLARITY_RULES: dict[ProducerName, dict[str, frozenset[DeltaPolarity]]] = {
    "tool_execution": {
        "exit_zero": frozenset({"positive", "neutral"}),
        "exit_nonzero": frozenset({"neutral", "unknown"}),
        "fatal_failure": frozenset({"negative"}),
        "timed_out": frozenset({"neutral", "unknown"}),
        "unavailable": frozenset({"neutral"}),
    },
    "test_parser": {
        "tests_passed": frozenset({"positive"}),
        "tests_failed": frozenset({"negative"}),
        "tests_partial": frozenset({"positive", "negative", "neutral"}),
        "parse_failed": frozenset({"unknown"}),
    },
    "diff_verifier": {
        "patch_applied": frozenset({"positive", "neutral"}),
        "patch_failed": frozenset({"negative", "neutral", "unknown"}),
        "context_mismatch": frozenset({"negative"}),
        "hunk_header_mismatch": frozenset({"negative"}),
        "repair_accepted": frozenset({"positive"}),
        "repair_rejected": frozenset({"negative", "neutral"}),
    },
    "formal_verifier": {
        "obligation_proved": frozenset({"positive"}),
        "obligation_refuted": frozenset({"negative"}),
        "counterexample_found": frozenset({"negative"}),
        "obligation_unknown": frozenset({"unknown"}),
        "verifier_unavailable": frozenset({"neutral"}),
        "assumption_invalidated": frozenset({"negative", "neutral"}),
    },
    "code_node_return": {
        "structured_return_valid": frozenset({"positive"}),
        "structured_return_invalid": frozenset({"negative", "neutral"}),
    },
    "planner_decision": {
        "topology_selected": frozenset({"neutral"}),
        "decomposition_applied": frozenset({"neutral"}),
    },
}

_PAYLOAD_SCHEMA_TABLE = PAYLOAD_ALLOWED_KEYS


@dataclass(frozen=True, slots=True)
class DeltaProducerResult:
    deltas: tuple["RuntimeDelta", ...]
    rejected_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeDelta:
    schema_version: Literal["0"]
    producer: ProducerName
    delta_kind: str
    polarity: DeltaPolarity
    confidence: float
    run_id: str
    node_run_id: str | None = None
    event_seq: int | None = None
    source_id: str = ""
    evidence_hash: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_DELTA_SCHEMA_VERSION:
            raise EvidenceError(
                f"schema_version must be {RUNTIME_DELTA_SCHEMA_VERSION!r}"
            )
        if self.producer not in PRODUCERS:
            raise EvidenceError(f"unknown producer: {self.producer!r}")
        producer = cast(ProducerName, self.producer)
        if self.delta_kind not in _DELTA_KIND_TABLE[producer]:
            raise EvidenceError(
                f"unknown delta_kind {self.delta_kind!r} for producer {producer!r}"
            )
        if self.polarity not in POLARITIES:
            raise EvidenceError(f"unknown polarity: {self.polarity!r}")
        allowed_polarities = _POLARITY_RULES[producer][self.delta_kind]
        if self.polarity not in allowed_polarities:
            raise EvidenceError(
                f"polarity {self.polarity!r} is illegal for "
                f"{producer}/{self.delta_kind}"
            )
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise EvidenceError(f"confidence must be in [0, 1], got {self.confidence}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise EvidenceError("run_id must be a non-empty string")
        if self.event_seq is not None and (
            not isinstance(self.event_seq, int) or self.event_seq < 0
        ):
            raise EvidenceError("event_seq must be a non-negative int or None")

        validate_payload(producer, self.delta_kind, self.payload)
        frozen_payload = deep_freeze_payload(self.payload)
        object.__setattr__(self, "payload", frozen_payload)

        expected_hash = compute_evidence_hash(
            schema_version=self.schema_version,
            producer=producer,
            delta_kind=self.delta_kind,
            polarity=self.polarity,
            source_id=self.source_id,
            payload=frozen_payload,
        )
        if self.evidence_hash is None:
            object.__setattr__(self, "evidence_hash", expected_hash)
        elif self.evidence_hash != expected_hash:
            raise EvidenceError(
                "evidence_hash does not match stable RuntimeDelta envelope"
            )
