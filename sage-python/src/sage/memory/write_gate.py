"""Write gating for memory — composite salience scoring.

Research basis: arXiv 2603.15994 — composite salience gate achieves 100%
accuracy vs 13% with ungated writes. 5-signal scoring: confidence, novelty,
reliability, recency, task relevance.

Backward-compatible: WriteGate is the simple gate (legacy), CompositeWriteGate
is the SOTA 5-signal gate. Both expose evaluate() -> WriteDecision.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from sage.constants import (
    SALIENCE_WEIGHT_CONFIDENCE,
    SALIENCE_WEIGHT_NOVELTY,
    SALIENCE_WEIGHT_RELIABILITY,
    SALIENCE_WEIGHT_RECENCY,
    SALIENCE_WEIGHT_RELEVANCE,
    SALIENCE_NOVELTY_SIM_THRESHOLD,
    SALIENCE_DEFAULT_THRESHOLD,
)


@dataclass
class WriteDecision:
    """Result of a write gate evaluation."""

    allowed: bool
    confidence: float
    reason: str = ""
    salience_score: float = 0.0
    signal_breakdown: dict[str, float] = field(default_factory=dict)


# -- Reliability tiers ---------------------------------------------------------

# Source tier -> reputation score (0-1). Configurable, not heuristic.
DEFAULT_RELIABILITY_SCORES: dict[str, float] = {
    "codex": 0.95,
    "reasoner": 0.90,
    "fast": 0.75,
    "budget": 0.70,
    "budget-alt": 0.70,
    "unknown": 0.60,
}


class WriteGate:
    """Legacy write gate — simple confidence threshold + dedup.

    Retained for backward compatibility. Use CompositeWriteGate for SOTA.
    """

    def __init__(self, threshold: float = 0.5, max_dedup_size: int = 10_000) -> None:
        self.threshold = threshold
        self.max_dedup_size = max_dedup_size
        self._write_count = 0
        self._abstention_count = 0
        self._seen_content: OrderedDict[str, None] = OrderedDict()

    def evaluate(self, content: str, confidence: float, **kwargs: Any) -> WriteDecision:
        """Decide whether to allow a memory write."""
        if not content or not content.strip():
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: empty content",
            )

        if content in self._seen_content:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: duplicate content",
            )

        if confidence < self.threshold:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason=f"Blocked: confidence {confidence:.2f} below threshold {self.threshold:.2f}",
            )

        self._write_count += 1
        self._seen_content[content] = None
        if len(self._seen_content) > self.max_dedup_size:
            self._seen_content.popitem(last=False)
        return WriteDecision(allowed=True, confidence=confidence, reason="Allowed")

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


class CompositeWriteGate:
    """SOTA write gate with 5-signal composite salience scoring.

    Signals:
    1. Confidence — model confidence in the content
    2. Novelty — 1 - max_cosine_similarity to recent entries
    3. Reliability — source tier reputation score
    4. Recency — time decay since task start
    5. Task relevance — keyword overlap with current task

    All weights from arXiv 2603.15994, documented as "subject to ablation".
    """

    def __init__(
        self,
        threshold: float = SALIENCE_DEFAULT_THRESHOLD,
        max_seen: int = 200,
        recency_halflife_s: float = 300.0,
        reliability_scores: dict[str, float] | None = None,
        w_confidence: float | None = None,
        w_novelty: float | None = None,
        w_reliability: float | None = None,
        w_recency: float | None = None,
        w_relevance: float | None = None,
        novelty_sim_threshold: float | None = None,
    ) -> None:
        self.threshold = threshold
        self._recency_halflife = recency_halflife_s
        self._reliability_scores = reliability_scores or DEFAULT_RELIABILITY_SCORES
        self._max_seen = max_seen

        # Per-instance weight overrides (parity with Rust RustCompositeWriteGate).
        # None → use the module-level SALIENCE_WEIGHT_* constants (default behavior).
        self._w_confidence = w_confidence if w_confidence is not None else SALIENCE_WEIGHT_CONFIDENCE
        self._w_novelty = w_novelty if w_novelty is not None else SALIENCE_WEIGHT_NOVELTY
        self._w_reliability = w_reliability if w_reliability is not None else SALIENCE_WEIGHT_RELIABILITY
        self._w_recency = w_recency if w_recency is not None else SALIENCE_WEIGHT_RECENCY
        self._w_relevance = w_relevance if w_relevance is not None else SALIENCE_WEIGHT_RELEVANCE
        self._novelty_sim_threshold = (
            novelty_sim_threshold if novelty_sim_threshold is not None
            else SALIENCE_NOVELTY_SIM_THRESHOLD
        )

        # Ring buffer of recent content embeddings for novelty computation
        self._seen_embeddings: deque[list[float]] = deque(maxlen=max_seen)
        # Exact dedup (bounded)
        self._seen_content: OrderedDict[str, None] = OrderedDict()
        self._max_dedup = 10_000

        self._write_count = 0
        self._abstention_count = 0
        self._task_start_time: float = time.monotonic()

        # Lazy embedder (loaded on first novelty computation)
        self._embedder: Any = None

    def reset_task_timer(self) -> None:
        """Reset recency timer (call at task start)."""
        self._task_start_time = time.monotonic()

    def evaluate(
        self,
        content: str,
        confidence: float,
        *,
        task: str = "",
        source_tier: str = "unknown",
        embedding: list[float] | None = None,
    ) -> WriteDecision:
        """Evaluate write with composite salience scoring.

        Parameters
        ----------
        content : str
            The content to be written.
        confidence : float
            Model confidence score (0-1).
        task : str
            Current task description (for relevance signal).
        source_tier : str
            Model tier that produced the content (for reliability signal).
        embedding : list[float] | None
            Pre-computed embedding of content. If None, novelty defaults to 1.0.
        """
        # Hard checks (same as legacy gate)
        if not content or not content.strip():
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: empty content",
            )

        if content in self._seen_content:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False, confidence=confidence,
                reason="Blocked: exact duplicate",
            )

        # --- Compute 5 signals ---
        sig_confidence = min(max(confidence, 0.0), 1.0)

        # Novelty: 1 - max similarity to recent embeddings
        sig_novelty = self._compute_novelty(embedding)

        # Reliability: source tier reputation
        sig_reliability = self._reliability_scores.get(
            source_tier, self._reliability_scores.get("unknown", 0.6)
        )

        # Recency: exponential decay since task start
        elapsed = time.monotonic() - self._task_start_time
        sig_recency = math.exp(-0.693 * elapsed / max(self._recency_halflife, 1.0))

        # Task relevance: keyword overlap
        sig_relevance = self._compute_relevance(task, content) if task else 0.5

        # Composite score (per-instance weights, default to module constants)
        score = (
            self._w_confidence * sig_confidence
            + self._w_novelty * sig_novelty
            + self._w_reliability * sig_reliability
            + self._w_recency * sig_recency
            + self._w_relevance * sig_relevance
        )

        breakdown = {
            "confidence": round(sig_confidence, 3),
            "novelty": round(sig_novelty, 3),
            "reliability": round(sig_reliability, 3),
            "recency": round(sig_recency, 3),
            "relevance": round(sig_relevance, 3),
        }

        if score < self.threshold:
            self._abstention_count += 1
            return WriteDecision(
                allowed=False,
                confidence=confidence,
                salience_score=round(score, 4),
                signal_breakdown=breakdown,
                reason=f"Blocked: salience {score:.3f} < threshold {self.threshold:.3f}",
            )

        # Allow — update state
        self._write_count += 1
        self._seen_content[content] = None
        if len(self._seen_content) > self._max_dedup:
            self._seen_content.popitem(last=False)
        if embedding is not None:
            self._seen_embeddings.append(embedding)

        return WriteDecision(
            allowed=True,
            confidence=confidence,
            salience_score=round(score, 4),
            signal_breakdown=breakdown,
            reason="Allowed",
        )

    def _compute_novelty(self, embedding: list[float] | None) -> float:
        """Compute novelty as 1 - max_cosine_similarity to seen embeddings."""
        if embedding is None or not self._seen_embeddings:
            return 1.0  # No comparison possible → fully novel

        max_sim = 0.0
        norm_a = math.sqrt(sum(x * x for x in embedding)) or 1e-9
        for seen in self._seen_embeddings:
            dot = sum(a * b for a, b in zip(embedding, seen))
            norm_b = math.sqrt(sum(x * x for x in seen)) or 1e-9
            sim = dot / (norm_a * norm_b)
            if sim > max_sim:
                max_sim = sim

        if max_sim > self._novelty_sim_threshold:
            return 0.0  # Near-duplicate → zero novelty

        return 1.0 - max_sim

    @staticmethod
    def _compute_relevance(task: str, content: str) -> float:
        """Keyword overlap scoring (same logic as RelevanceGate)."""
        import re
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "but", "or", "not", "this", "that", "it", "its",
        })

        def tokenize(text: str) -> set[str]:
            words = re.findall(r'\b[a-z][a-z0-9_]+\b', text.lower())
            return {w for w in words if w not in stop_words and len(w) >= 3}

        task_tokens = tokenize(task)
        content_tokens = tokenize(content)
        if not task_tokens:
            return 0.5
        overlap = task_tokens & content_tokens
        return len(overlap) / len(task_tokens)

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
            "seen_embeddings": len(self._seen_embeddings),
            "unique_entries": len(self._seen_content),
        }


def log_write_gate_decision(decision: Any, source_tier: str = "unknown") -> None:
    """Emit a structured INFO log for a WriteDecision.

    Factored out so every runtime path that calls `gate.evaluate()` can share
    the exact same format -- `grep -c "memory.write_gate.fired"` on a smoke
    run is then a reliable signal of Memory-pillar activity. Callers MUST
    invoke this immediately after `gate.evaluate()` returns.

    Format: key=value pairs, one line. Parseable by a simple regex.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    allowed = bool(getattr(decision, "allowed", True))
    salience = float(getattr(decision, "salience_score", 0.0) or 0.0)
    breakdown = getattr(decision, "signal_breakdown", {}) or {}
    reason = getattr(decision, "reason", "") or ""
    _log.info(
        "memory.write_gate.fired decision=%s salience=%.3f tier=%s "
        "conf=%.2f nov=%.2f rel=%.2f rec=%.2f relv=%.2f reason=%r",
        "persist" if allowed else "abstain",
        salience,
        source_tier or "unknown",
        float(breakdown.get("confidence", 0.0)),
        float(breakdown.get("novelty", 0.0)),
        float(breakdown.get("reliability", 0.0)),
        float(breakdown.get("recency", 0.0)),
        float(breakdown.get("relevance", 0.0)),
        reason[:80],
    )


# -- Rust acceleration (same pattern as RustRelevanceGate) ---------------------

try:
    from sage_core import RustCompositeWriteGate as _RustGate
    _HAS_RUST_GATE = True
except ImportError:
    _HAS_RUST_GATE = False


def create_composite_write_gate(
    threshold: float = SALIENCE_DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> Any:
    """Factory: returns Rust gate when available, Python CompositeWriteGate otherwise.

    The Rust implementation is ~10x faster on novelty cosine computation
    and uses FNV hashing for O(1) exact dedup.
    """
    import logging
    _log = logging.getLogger(__name__)
    if _HAS_RUST_GATE:
        try:
            gate = _RustGate(threshold=threshold, **kwargs)
            _log.info("CompositeWriteGate: using Rust acceleration")
            return gate
        except Exception:
            pass
    return CompositeWriteGate(threshold=threshold, **kwargs)


# -- Source tier resolution (2026-04-19 gate-wiring audit) ---------------------
#
# The gate's `source_tier` feeds the reliability signal. Higher-tier models
# produce more reliable writes, so their entries cross the threshold more
# easily. Mapping: S3→"reasoner" (0.90), S2→"fast" (0.75), S1→"budget" (0.70).
# Unknown model → "unknown" (0.60) — degrades gracefully.
#
# Source of truth: cards.toml loaded via Rust ModelRegistry (Critical
# Directive #6: no training-leak hardcodes on provider model strings).
_TIER_REGISTRY: dict[str, str] = {}
_TIER_REGISTRY_LOADED = False


def _load_tier_registry() -> None:
    """Populate _TIER_REGISTRY from cards.toml via Rust ModelRegistry."""
    global _TIER_REGISTRY_LOADED
    if _TIER_REGISTRY_LOADED:
        return
    _TIER_REGISTRY_LOADED = True  # Even if load fails, don't retry every call
    try:
        from sage_core import ModelRegistry  # type: ignore[import-not-found]
        from pathlib import Path
        for p in [
            Path.cwd() / "sage-core" / "config" / "cards.toml",
            Path.cwd().parent / "sage-core" / "config" / "cards.toml",
            Path.cwd() / "config" / "cards.toml",
            Path(__file__).resolve().parents[4] / "sage-core" / "config" / "cards.toml",
        ]:
            if p.exists():
                reg = ModelRegistry.from_toml_file(str(p))
                for card in reg.all_models():
                    best = max(
                        ("s1", card.s1_affinity),
                        ("s2", card.s2_affinity),
                        ("s3", card.s3_affinity),
                        key=lambda x: x[1],
                    )[0]
                    tier = {"s3": "reasoner", "s2": "fast", "s1": "budget"}[best]
                    _TIER_REGISTRY[card.id] = tier
                break
    except (ImportError, IOError, OSError):
        pass  # Rust unavailable → all writes get "unknown" (0.60)


def infer_source_tier(model_id: str | None) -> str:
    """Map a model_id to the gate's reliability-tier string.

    Returns: "reasoner" | "fast" | "budget" | "unknown". Uses cards.toml
    as source of truth per Critical Directive #6 — does NOT pattern-match
    on model id substrings.
    """
    if not model_id:
        return "unknown"
    _load_tier_registry()
    return _TIER_REGISTRY.get(model_id, "unknown")
