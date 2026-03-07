"""Write gating for memory — abstention when confidence is low.

Better to forget than to store noise. Every memory write goes through
the WriteGate, which checks confidence score, content validity, and
deduplication before allowing the write.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class WriteDecision:
    """Result of a write gate evaluation."""

    allowed: bool
    confidence: float
    reason: str = ""


class WriteGate:
    """Guards memory writes with confidence thresholding and dedup.

    Parameters
    ----------
    threshold:
        Minimum confidence score to allow a write (0.0-1.0).
    """

    def __init__(self, threshold: float = 0.5, max_dedup_size: int = 10_000) -> None:
        self.threshold = threshold
        self.max_dedup_size = max_dedup_size
        self._write_count = 0
        self._abstention_count = 0
        self._seen_content: OrderedDict[str, None] = OrderedDict()

    def evaluate(self, content: str, confidence: float) -> WriteDecision:
        """Decide whether to allow a memory write.

        Checks in order:
        1. Empty content -> block
        2. Duplicate content -> block
        3. Confidence below threshold -> block (abstain)
        4. Otherwise -> allow
        """
        # Empty content
        if not content or not content.strip():
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: empty content",
            )

        # Duplicate (bounded LRU dedup)
        if content in self._seen_content:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: duplicate content",
            )

        # Confidence threshold
        if confidence < self.threshold:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason=f"Blocked: confidence {confidence:.2f} below threshold {self.threshold:.2f}",
            )

        # Allow
        self._write_count += 1
        self._seen_content[content] = None
        if len(self._seen_content) > self.max_dedup_size:
            self._seen_content.popitem(last=False)  # Evict oldest
        return WriteDecision(
            allowed=True, confidence=confidence,
            reason="Allowed",
        )

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def abstention_count(self) -> int:
        return self._abstention_count

    @property
    def abstention_rate(self) -> float:
        total = self._write_count + self._abstention_count
        if total == 0:
            return 0.0
        return self._abstention_count / total

    def stats(self) -> dict:
        return {
            "writes": self._write_count,
            "abstentions": self._abstention_count,
            "abstention_rate": round(self.abstention_rate, 4),
            "threshold": self.threshold,
            "unique_entries": len(self._seen_content),
        }
