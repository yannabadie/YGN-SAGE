"""CRAG-style relevance gate for memory injection.

Evaluates whether retrieved memory context is relevant to the current task
before injecting it into the LLM prompt. Uses keyword overlap scoring (fast, no LLM call).
Sprint 3 evidence: blind injection = -30pp degradation.
"""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)


class RelevanceGate:
    """Gate that scores and filters memory context by relevance to task.

    Args:
        threshold: Minimum relevance score (0-1) to allow injection.
                   Default 0.3 calibrated against Sprint 3 benchmarks.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self._stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "not", "no", "nor",
            "this", "that", "these", "those", "it", "its", "i", "you",
            "he", "she", "we", "they", "me", "him", "her", "us", "them",
        })

    def _tokenize(self, text: str) -> set[str]:
        """Extract meaningful tokens (lowercase, no stop words, len >= 3)."""
        words = re.findall(r'\b[a-z][a-z0-9_]+\b', text.lower())
        return {w for w in words if w not in self._stop_words and len(w) >= 3}

    def score(self, task: str, context: str) -> float:
        """Compute relevance score. Returns float in [0.0, 1.0]."""
        if not task or not context:
            return 0.0
        task_tokens = self._tokenize(task)
        ctx_tokens = self._tokenize(context)
        if not task_tokens or not ctx_tokens:
            return 0.0
        overlap = task_tokens & ctx_tokens
        return len(overlap) / len(task_tokens)

    def is_relevant(self, task: str, context: str) -> bool:
        """Check if context passes relevance threshold for injection."""
        if not context or not context.strip():
            return False
        s = self.score(task, context)
        if s < self.threshold:
            log.debug("RelevanceGate rejected (score=%.2f < threshold=%.2f)", s, self.threshold)
            return False
        return True


# Rust acceleration (Phase 2 rationalization)
try:
    from sage_core import RustRelevanceGate as _RustGate
    _HAS_RUST_GATE = True
except ImportError:
    _HAS_RUST_GATE = False


def create_relevance_gate(threshold: float = 0.3) -> RelevanceGate:
    """Factory: returns Rust gate when available, Python otherwise."""
    if _HAS_RUST_GATE:
        try:
            gate = _RustGate(threshold=threshold)
            log.info("RelevanceGate: using Rust acceleration")
            return gate  # type: ignore[return-value]  # duck-type compatible
        except Exception:
            pass
    return RelevanceGate(threshold=threshold)
