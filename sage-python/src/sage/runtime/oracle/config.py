from __future__ import annotations

from dataclasses import dataclass

from sage.runtime.oracle.verdict import VerdictSource


@dataclass(frozen=True, slots=True)
class OracleConfig:
    min_confidence: float = 0.7
    enable_llm_judge: bool = False
    enable_lexical_fallback: bool = False
    timeout_per_oracle_sec: float = 5.0
    require_evidence_when_trainable: bool = True
    allowed_sources: tuple[VerdictSource, ...] = (
        "exact",
        "tool",
        "formal",
        "spec",
        "llm_judge",
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
        if self.timeout_per_oracle_sec <= 0.0:
            raise ValueError(
                "timeout_per_oracle_sec must be positive, "
                f"got {self.timeout_per_oracle_sec}"
            )
        if "abstain" in self.allowed_sources:
            raise ValueError("abstain is the terminal fallback, not an allowed source")
