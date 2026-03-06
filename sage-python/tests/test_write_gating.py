"""Tests for memory write gating — abstention when confidence is low."""
from __future__ import annotations

import pytest
from sage.memory.write_gate import WriteGate, WriteDecision


# ---------------------------------------------------------------------------
# Basic gating
# ---------------------------------------------------------------------------

def test_high_confidence_passes():
    gate = WriteGate(threshold=0.5)
    decision = gate.evaluate(content="Python is a language", confidence=0.9)
    assert decision.allowed is True
    assert decision.confidence == 0.9


def test_low_confidence_blocked():
    gate = WriteGate(threshold=0.5)
    decision = gate.evaluate(content="Maybe something", confidence=0.2)
    assert decision.allowed is False
    assert "below threshold" in decision.reason


def test_exact_threshold_passes():
    gate = WriteGate(threshold=0.5)
    decision = gate.evaluate(content="Exact", confidence=0.5)
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# Abstention tracking
# ---------------------------------------------------------------------------

def test_abstention_counter():
    gate = WriteGate(threshold=0.7)
    gate.evaluate(content="low", confidence=0.1)
    gate.evaluate(content="low", confidence=0.2)
    gate.evaluate(content="high", confidence=0.9)
    assert gate.abstention_count == 2
    assert gate.write_count == 1


def test_abstention_rate():
    gate = WriteGate(threshold=0.5)
    gate.evaluate(content="a", confidence=0.1)
    gate.evaluate(content="b", confidence=0.9)
    assert gate.abstention_rate == 0.5


def test_abstention_rate_no_evaluations():
    gate = WriteGate(threshold=0.5)
    assert gate.abstention_rate == 0.0


# ---------------------------------------------------------------------------
# Content-based filtering
# ---------------------------------------------------------------------------

def test_empty_content_blocked():
    gate = WriteGate(threshold=0.3)
    decision = gate.evaluate(content="", confidence=0.9)
    assert decision.allowed is False
    assert "empty" in decision.reason


def test_duplicate_content_blocked():
    gate = WriteGate(threshold=0.3)
    gate.evaluate(content="Same fact", confidence=0.8)
    decision = gate.evaluate(content="Same fact", confidence=0.8)
    assert decision.allowed is False
    assert "duplicate" in decision.reason


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats():
    gate = WriteGate(threshold=0.5)
    gate.evaluate(content="a", confidence=0.9)
    gate.evaluate(content="b", confidence=0.1)
    stats = gate.stats()
    assert stats["writes"] == 1
    assert stats["abstentions"] == 1
    assert stats["threshold"] == 0.5
