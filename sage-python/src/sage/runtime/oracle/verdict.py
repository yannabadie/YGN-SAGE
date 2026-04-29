from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ORACLE_VERDICT_SCHEMA_VERSION: Literal["0"] = "0"

VerdictSource = Literal["exact", "tool", "formal", "spec", "llm_judge", "abstain"]
VERDICT_SOURCES: tuple[str, ...] = (
    "exact",
    "tool",
    "formal",
    "spec",
    "llm_judge",
    "abstain",
)

QualityLabel = Literal["pass", "fail", "partial", "unknown"]
QUALITY_LABELS: tuple[str, ...] = ("pass", "fail", "partial", "unknown")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    run_id: str
    node_run_id: str | None = None
    event_seq: int | None = None
    output_sha256: str | None = None
    tool_call_id: str | None = None
    verifier_id: str | None = None


@dataclass(frozen=True, slots=True)
class OracleVerdict:
    trainable: bool
    verdict_source: VerdictSource
    quality_label: QualityLabel
    score: float | None
    confidence: float
    reason_codes: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    schema_version: Literal["0"] = ORACLE_VERDICT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.trainable:
            if self.verdict_source == "abstain":
                raise ValueError("trainable=True cannot have verdict_source='abstain'")
            if self.quality_label not in {"pass", "fail", "partial"}:
                raise ValueError(
                    "trainable=True requires concrete quality_label, "
                    f"got {self.quality_label!r}"
                )
            if self.score is None:
                raise ValueError("trainable=True requires score, got None")
            if not (0.0 <= self.score <= 1.0):
                raise ValueError(f"score must be in [0, 1], got {self.score}")
            if len(self.reason_codes) < 1:
                raise ValueError("trainable=True requires at least one reason_code")
            if len(self.evidence) < 1:
                raise ValueError("trainable=True requires at least one EvidenceRef")

        if not self.trainable and self.verdict_source != "abstain":
            raise ValueError(
                "trainable=False verdicts must use verdict_source='abstain'"
            )

        if self.verdict_source == "abstain":
            if self.trainable:
                raise ValueError("verdict_source='abstain' must have trainable=False")
            if self.quality_label != "unknown":
                raise ValueError(
                    "verdict_source='abstain' must have quality_label='unknown', "
                    f"got {self.quality_label!r}"
                )
            if self.score is not None:
                raise ValueError("verdict_source='abstain' must have score=None")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        """Canonical dict for JSONL emission + RunFrame summary. No raw output."""
        return {
            "schema_version": self.schema_version,
            "trainable": self.trainable,
            "verdict_source": self.verdict_source,
            "quality_label": self.quality_label,
            "score": self.score,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "evidence": [
                {
                    "run_id": e.run_id,
                    "node_run_id": e.node_run_id,
                    "event_seq": e.event_seq,
                    "output_sha256": e.output_sha256,
                    "tool_call_id": e.tool_call_id,
                    "verifier_id": e.verifier_id,
                }
                for e in self.evidence
            ],
        }
