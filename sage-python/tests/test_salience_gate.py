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
