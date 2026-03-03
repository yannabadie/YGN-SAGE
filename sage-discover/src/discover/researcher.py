"""Research agent: the core discovery loop.

Implements the explore -> hypothesize -> evolve -> evaluate cycle
using YGN-SAGE's 5 cognitive pillars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    """A research hypothesis to be tested."""
    id: str
    statement: str
    confidence: float = 0.0  # 0-1 confidence score
    evidence: list[str] = field(default_factory=list)
    status: str = "proposed"  # proposed, testing, confirmed, rejected


@dataclass
class Discovery:
    """A confirmed research finding."""
    hypothesis: Hypothesis
    code: str  # The solution code
    score: float
    evaluation_details: dict[str, Any] = field(default_factory=dict)


class ResearchAgent:
    """Core research agent implementing the discovery cycle.

    Phases:
    1. Explore: gather information, search papers/code
    2. Hypothesize: form testable hypotheses
    3. Evolve: use evolution engine to develop solutions
    4. Evaluate: test solutions through evaluation cascade
    """

    def __init__(self, domain: str, goal: str):
        self.domain = domain
        self.goal = goal
        self._hypotheses: list[Hypothesis] = []
        self._discoveries: list[Discovery] = []
        self._iteration = 0

    def add_hypothesis(self, statement: str, evidence: list[str] | None = None) -> Hypothesis:
        """Add a new hypothesis to investigate."""
        h = Hypothesis(
            id=f"H{len(self._hypotheses) + 1}",
            statement=statement,
            evidence=evidence or [],
        )
        self._hypotheses.append(h)
        return h

    def confirm_hypothesis(self, hypothesis_id: str, code: str, score: float, details: dict | None = None) -> Discovery:
        """Mark a hypothesis as confirmed and record the discovery."""
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                h.status = "confirmed"
                h.confidence = score
                discovery = Discovery(
                    hypothesis=h,
                    code=code,
                    score=score,
                    evaluation_details=details or {},
                )
                self._discoveries.append(discovery)
                return discovery
        raise ValueError(f"Hypothesis {hypothesis_id} not found")

    def reject_hypothesis(self, hypothesis_id: str, reason: str = "") -> None:
        """Mark a hypothesis as rejected."""
        for h in self._hypotheses:
            if h.id == hypothesis_id:
                h.status = "rejected"
                h.evidence.append(f"Rejected: {reason}")
                return
        raise ValueError(f"Hypothesis {hypothesis_id} not found")

    @property
    def hypotheses(self) -> list[Hypothesis]:
        return list(self._hypotheses)

    @property
    def discoveries(self) -> list[Discovery]:
        return list(self._discoveries)

    @property
    def pending_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self._hypotheses if h.status == "proposed"]

    def stats(self) -> dict[str, Any]:
        """Get research statistics."""
        return {
            "domain": self.domain,
            "goal": self.goal,
            "total_hypotheses": len(self._hypotheses),
            "confirmed": sum(1 for h in self._hypotheses if h.status == "confirmed"),
            "rejected": sum(1 for h in self._hypotheses if h.status == "rejected"),
            "pending": sum(1 for h in self._hypotheses if h.status == "proposed"),
            "discoveries": len(self._discoveries),
            "best_score": max((d.score for d in self._discoveries), default=0.0),
        }
