"""Evidence records for every claim YGN-SAGE makes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum


class EvidenceLevel(IntEnum):
    HEURISTIC = 1
    CHECKED = 2
    MODEL_JUDGED = 3
    SOLVER_PROVED = 4
    EMPIRICALLY_VALIDATED = 5


@dataclass
class EvidenceRecord:
    level: EvidenceLevel
    proof_strength: float = 0.0
    external_validity: bool = False
    coverage: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "level": self.level.name.lower(),
            "proof_strength": self.proof_strength,
            "external_validity": self.external_validity,
            "coverage": self.coverage,
            "assumptions": self.assumptions,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp.isoformat(),
        }

    def readme_label(self) -> str:
        return self.level.name.lower()
