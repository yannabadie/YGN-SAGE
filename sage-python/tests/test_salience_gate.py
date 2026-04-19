"""Tests for composite salience write gate.

Research basis: arXiv 2603.15994 — composite salience scoring achieves 100%
accuracy vs 13% with ungated writes.
"""
from __future__ import annotations

import time

import pytest

from sage.memory.write_gate import WriteGate, CompositeWriteGate, WriteDecision


# -- Legacy WriteGate backward compatibility -----------------------------------

class TestLegacyWriteGate:
    """Ensure the legacy gate still works unchanged."""

    def test_high_confidence_passes(self):
        gate = WriteGate(threshold=0.5)
        d = gate.evaluate("hello", 0.8)
        assert d.allowed
        assert gate.write_count == 1

    def test_low_confidence_blocked(self):
        gate = WriteGate(threshold=0.5)
        d = gate.evaluate("hello", 0.3)
        assert not d.allowed
        assert gate.abstention_count == 1

    def test_empty_content_blocked(self):
        gate = WriteGate()
        d = gate.evaluate("", 0.9)
        assert not d.allowed

    def test_duplicate_blocked(self):
        gate = WriteGate()
        gate.evaluate("same", 0.9)
        d = gate.evaluate("same", 0.9)
        assert not d.allowed
        assert "duplicate" in d.reason

    def test_stats(self):
        gate = WriteGate()
        gate.evaluate("ok", 0.9)
        gate.evaluate("bad", 0.1)
        s = gate.stats()
        assert s["writes"] == 1
        assert s["abstentions"] == 1


# -- CompositeWriteGate --------------------------------------------------------

class TestCompositeWriteGate:
    """Test the 5-signal composite salience gate."""

    def test_high_confidence_novel_content_passes(self):
        gate = CompositeWriteGate(threshold=0.3)
        d = gate.evaluate("novel content about algorithms", confidence=0.9, task="implement algorithms")
        assert d.allowed
        assert d.salience_score > 0.3
        assert "confidence" in d.signal_breakdown

    def test_empty_content_blocked(self):
        gate = CompositeWriteGate()
        d = gate.evaluate("", confidence=0.9)
        assert not d.allowed
        assert "empty" in d.reason

    def test_exact_duplicate_blocked(self):
        gate = CompositeWriteGate()
        gate.evaluate("same content", confidence=0.9)
        d = gate.evaluate("same content", confidence=0.9)
        assert not d.allowed
        assert "duplicate" in d.reason

    def test_novelty_signal_blocks_similar_embeddings(self):
        gate = CompositeWriteGate(threshold=0.5)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.99, 0.01, 0.0]  # Very similar

        gate.evaluate("first entry", confidence=0.9, embedding=emb1)
        d = gate.evaluate("second entry", confidence=0.9, embedding=emb2)
        # Novelty should be near 0 (similar embedding), reducing salience
        assert d.signal_breakdown["novelty"] < 0.1

    def test_novelty_high_for_orthogonal_embeddings(self):
        gate = CompositeWriteGate(threshold=0.1)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]  # Orthogonal

        gate.evaluate("first", confidence=0.9, embedding=emb1)
        d = gate.evaluate("second", confidence=0.9, embedding=emb2)
        assert d.signal_breakdown["novelty"] == 1.0  # Fully novel

    def test_novelty_is_1_without_embedding(self):
        gate = CompositeWriteGate(threshold=0.1)
        d = gate.evaluate("content", confidence=0.8)
        assert d.signal_breakdown["novelty"] == 1.0

    def test_reliability_by_source_tier(self):
        gate = CompositeWriteGate(threshold=0.01)
        d1 = gate.evaluate("content1", confidence=0.5, source_tier="codex")
        d2 = gate.evaluate("content2", confidence=0.5, source_tier="budget")
        assert d1.signal_breakdown["reliability"] > d2.signal_breakdown["reliability"]

    def test_reliability_unknown_tier_defaults(self):
        gate = CompositeWriteGate(threshold=0.01)
        d = gate.evaluate("content", confidence=0.5, source_tier="never_heard_of")
        assert d.signal_breakdown["reliability"] == 0.6  # default for unknown

    def test_recency_decays_over_time(self):
        gate = CompositeWriteGate(threshold=0.01, recency_halflife_s=0.1)
        d1 = gate.evaluate("early", confidence=0.5)
        time.sleep(0.15)  # > halflife
        d2 = gate.evaluate("late", confidence=0.5)
        assert d2.signal_breakdown["recency"] < d1.signal_breakdown["recency"]

    def test_relevance_signal(self):
        gate = CompositeWriteGate(threshold=0.01)
        d1 = gate.evaluate("implements quicksort algorithm", confidence=0.5, task="implement quicksort")
        d2 = gate.evaluate("weather is nice today", confidence=0.5, task="implement quicksort")
        assert d1.signal_breakdown["relevance"] > d2.signal_breakdown["relevance"]

    def test_relevance_default_when_no_task(self):
        gate = CompositeWriteGate(threshold=0.01)
        d = gate.evaluate("some content", confidence=0.5, task="")
        assert d.signal_breakdown["relevance"] == 0.5

    def test_low_salience_blocked(self):
        gate = CompositeWriteGate(threshold=0.9)  # Very high threshold
        d = gate.evaluate("content", confidence=0.1, source_tier="budget")
        assert not d.allowed
        assert "salience" in d.reason

    def test_stats(self):
        gate = CompositeWriteGate(threshold=0.01)
        gate.evaluate("a", confidence=0.9, embedding=[1.0, 0.0])
        gate.evaluate("b", confidence=0.9, embedding=[0.0, 1.0])
        s = gate.stats()
        assert s["writes"] == 2
        assert s["seen_embeddings"] == 2

    def test_reset_task_timer(self):
        gate = CompositeWriteGate(threshold=0.01, recency_halflife_s=1.0)
        time.sleep(0.1)
        gate.reset_task_timer()
        d = gate.evaluate("fresh", confidence=0.5)
        # Recency should be near 1.0 since timer was just reset
        assert d.signal_breakdown["recency"] > 0.9

    def test_dedup_eviction(self):
        gate = CompositeWriteGate(threshold=0.01)
        gate._max_dedup = 3
        for i in range(5):
            gate.evaluate(f"content_{i}", confidence=0.9)
        assert len(gate._seen_content) <= 3

    # G-series audit (2026-04-19): gate is now wired into phases/act.py.
    # These tests validate the per-instance weight overrides and the factory
    # config used by Pipeline (w_confidence=0, w_novelty=0.40, w_relevance=0.30).
    def test_per_instance_weight_overrides(self):
        """Pipeline uses w_confidence=0.0 since the loop has no per-turn
        confidence signal. Weights redistributed to novelty/relevance."""
        gate = CompositeWriteGate(
            threshold=0.35,
            w_confidence=0.0,
            w_novelty=0.40,
            w_reliability=0.20,
            w_recency=0.10,
            w_relevance=0.30,
        )
        assert gate._w_confidence == 0.0
        assert gate._w_novelty == 0.40
        assert gate._w_relevance == 0.30
        # Sanity: with w_confidence=0, a low-confidence write with good
        # relevance + novelty still passes.
        d = gate.evaluate(
            "fix for astropy units parsing bug",
            confidence=0.0,
            task="fix astropy units parsing",
        )
        assert d.allowed, f"expected allowed with 0 confidence + strong relevance: {d.reason}"

    def test_sentinel_second_occurrence_blocked_by_dedup(self):
        """The cross-node sentinel cascade (astropy-14995) emits the same
        sentinel string from planner, coder, synthesizer. With a shared
        pipeline-scoped gate, the 2nd and 3rd hits block on exact dedup."""
        sentinel = "[sage: agent exited after 20 steps with no content]"
        gate = CompositeWriteGate(
            threshold=0.35,
            w_confidence=0.0,
            w_novelty=0.40,
            w_reliability=0.20,
            w_recency=0.10,
            w_relevance=0.30,
        )
        d1 = gate.evaluate(sentinel, confidence=0.0, task="fix bug", source_tier="budget")
        d2 = gate.evaluate(sentinel, confidence=0.0, task="fix bug", source_tier="budget")
        d3 = gate.evaluate(sentinel, confidence=0.0, task="fix bug", source_tier="budget")
        # First one may pass or block (composite score depends on defaults);
        # what matters is d2 and d3 are blocked by exact dedup.
        assert not d2.allowed, "2nd sentinel should hit exact-dedup hash"
        assert not d3.allowed, "3rd sentinel should hit exact-dedup hash"
        assert "duplicate" in d2.reason.lower()


# -- Source tier inference (2026-04-19 gate wiring) ----------------------------

class TestInferSourceTier:
    """Validate infer_source_tier maps model ids to gate reliability tiers."""

    def test_unknown_model_returns_unknown(self):
        from sage.memory.write_gate import infer_source_tier
        assert infer_source_tier("no-such-model") == "unknown"

    def test_none_returns_unknown(self):
        from sage.memory.write_gate import infer_source_tier
        assert infer_source_tier(None) == "unknown"
        assert infer_source_tier("") == "unknown"

    def test_real_model_from_cards_toml(self):
        """If cards.toml is reachable, a known model should map to a known
        tier. If not reachable (Rust uncompiled), test degrades to 'unknown'
        without failing — gate just uses default reliability 0.60."""
        from sage.memory.write_gate import infer_source_tier
        # These ids come from sage-core/config/cards.toml. If Rust isn't
        # compiled, the registry is empty and we expect "unknown".
        tier = infer_source_tier("gemini-3.1-pro-preview")
        assert tier in {"reasoner", "fast", "budget", "unknown"}

    def test_signal_breakdown_has_all_5_keys(self):
        gate = CompositeWriteGate(threshold=0.01)
        d = gate.evaluate("test", confidence=0.7, task="task", source_tier="fast")
        assert set(d.signal_breakdown.keys()) == {"confidence", "novelty", "reliability", "recency", "relevance"}

    def test_abstention_rate(self):
        gate = CompositeWriteGate(threshold=0.8)  # High threshold
        gate.evaluate("good content about algorithms", confidence=0.95, task="algorithms")  # passes
        gate.evaluate("bad", confidence=0.0, source_tier="unknown")  # fails (low salience)
        assert gate.abstention_count >= 1
        assert gate.write_count >= 1
        assert gate.abstention_rate > 0
