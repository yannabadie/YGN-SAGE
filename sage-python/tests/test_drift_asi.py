"""Tests for the extended drift monitor (12-dim ASI) and adaptive mutator.

Research basis:
- Agent Drift (2601.04170): 12-dimension Agent Stability Index
- Behavioral Consistency (2602.11619): action variance predicts failure
- ShinkaEvolve (2509.19349): bandit-based adaptive LLM ensemble
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from sage.monitoring.extended_drift import (
    BehaviorTracker,
    ExtendedDriftMonitor,
    _normalized_levenshtein,
)
from sage.evolution.llm_mutator import AdaptiveMutator


# ── Mock event for testing ────────────────────────────────────────────────────

@dataclass
class MockEvent:
    latency_ms: float = 100.0
    cost_usd: float = 0.01
    content: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# ── BehaviorTracker tests ────────────────────────────────────────────────────

class TestBehaviorTracker:

    def test_empty_tracker_returns_1(self):
        bt = BehaviorTracker()
        assert bt.consistency_score() == 1.0

    def test_single_sequence_returns_1(self):
        bt = BehaviorTracker()
        bt.record_actions(["search", "read", "write"])
        assert bt.consistency_score() == 1.0

    def test_identical_sequences_perfect_consistency(self):
        bt = BehaviorTracker()
        for _ in range(5):
            bt.record_actions(["search", "read", "write"])
        assert bt.consistency_score() == 1.0

    def test_completely_different_sequences_low_consistency(self):
        bt = BehaviorTracker()
        bt.record_actions(["a", "b", "c"])
        bt.record_actions(["x", "y", "z"])
        assert bt.consistency_score() < 0.5

    def test_gradually_changing_sequences(self):
        bt = BehaviorTracker()
        bt.record_actions(["a", "b", "c"])
        bt.record_actions(["a", "b", "d"])  # 1 change
        bt.record_actions(["a", "e", "d"])  # 1 change
        score = bt.consistency_score()
        assert 0.3 < score < 0.9  # Moderate consistency

    def test_window_limit(self):
        bt = BehaviorTracker(window=3)
        for i in range(10):
            bt.record_actions([str(i)])
        assert bt.sequence_count == 3  # Window capped

    def test_sequence_count(self):
        bt = BehaviorTracker()
        bt.record_actions(["a"])
        bt.record_actions(["b"])
        assert bt.sequence_count == 2


class TestNormalizedLevenshtein:

    def test_identical_lists(self):
        assert _normalized_levenshtein(["a", "b"], ["a", "b"]) == 0.0

    def test_completely_different(self):
        assert _normalized_levenshtein(["a", "b"], ["c", "d"]) == 1.0

    def test_empty_lists(self):
        assert _normalized_levenshtein([], []) == 0.0

    def test_one_empty(self):
        assert _normalized_levenshtein(["a", "b"], []) == 1.0

    def test_one_insertion(self):
        d = _normalized_levenshtein(["a", "b"], ["a", "b", "c"])
        assert 0.0 < d < 1.0


# ── ExtendedDriftMonitor tests ────────────────────────────────────────────────

class TestExtendedDriftMonitor:

    def test_no_events_returns_zero(self):
        monitor = ExtendedDriftMonitor()
        report = monitor.analyze([])
        assert report.drift_score == 0.0
        assert report.action == "CONTINUE"

    def test_healthy_events_continue(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(latency_ms=100, cost_usd=0.01, content="Implementing the algorithm"),
            MockEvent(latency_ms=110, cost_usd=0.01, content="Testing the algorithm"),
            MockEvent(latency_ms=105, cost_usd=0.01, content="Algorithm is working"),
        ]
        report = monitor.analyze(events, task="implement algorithm")
        assert report.action == "CONTINUE"

    def test_high_error_rate_triggers_action(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(meta={"error": "timeout"}),
            MockEvent(meta={"error": "timeout"}),
            MockEvent(meta={"error": "connection"}),
        ]
        report = monitor.analyze(events)
        assert report.drift_score > 0.3

    def test_report_has_extended_signals(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(content="hello world", meta={"tool": "search"}),
            MockEvent(content="goodbye world", meta={"tool": "read"}),
            MockEvent(content="hello again", meta={"tool": "search"}),
        ]
        report = monitor.analyze(events, task="search for files")
        assert "asi_semantic" in report.details
        assert "asi_behavioral" in report.details
        assert "asi_topic" in report.details

    def test_behavioral_drift_detected(self):
        monitor = ExtendedDriftMonitor()
        # Record very different action sequences
        monitor.behavior_tracker.record_actions(["search", "read", "write"])
        monitor.behavior_tracker.record_actions(["delete", "upload", "download"])

        events = [MockEvent(), MockEvent(), MockEvent()]
        report = monitor.analyze(events)
        assert report.extended_signals.get("behavioral", 0) > 0.3

    def test_topic_drift_when_off_topic(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(content="The weather is sunny and warm today"),
            MockEvent(content="I like pizza and hamburgers"),
        ]
        report = monitor.analyze(events, task="implement binary search tree")
        # Output is unrelated to task -> topic drift should be high
        assert report.extended_signals.get("topic", 0) > 0.5

    def test_tool_diversity_all_same_tool(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(meta={"tool": "search"}),
            MockEvent(meta={"tool": "search"}),
            MockEvent(meta={"tool": "search"}),
        ]
        report = monitor.analyze(events)
        # Using only one tool -> diversity drift = 0 (only 1 unique tool, returns 0)
        assert report.extended_signals.get("tool_diversity", 0) == 0.0

    def test_tool_diversity_mixed(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(meta={"tool": "search"}),
            MockEvent(meta={"tool": "read"}),
            MockEvent(meta={"tool": "write"}),
            MockEvent(meta={"tool": "delete"}),
        ]
        report = monitor.analyze(events)
        # Uniform distribution -> low drift
        assert report.extended_signals.get("tool_diversity", 1) < 0.3

    def test_coordination_drift_with_unfinished_agents(self):
        monitor = ExtendedDriftMonitor()
        events = [
            MockEvent(meta={"sub_agent_spawn": True}),
            MockEvent(meta={"sub_agent_spawn": True}),
            MockEvent(meta={"sub_agent_complete": True}),
            # 2 spawns, 1 complete -> ratio 0.5 -> drift 0.5
        ]
        report = monitor.analyze(events)
        assert report.extended_signals.get("coordination", 0) > 0.3

    def test_asi_score_is_bounded(self):
        monitor = ExtendedDriftMonitor()
        events = [MockEvent(), MockEvent(), MockEvent()]
        report = monitor.analyze(events)
        assert 0.0 <= report.drift_score <= 1.0
        assert 0.0 <= report.asi_score <= 1.0


# ── AdaptiveMutator tests ────────────────────────────────────────────────────

class TestAdaptiveMutator:

    def test_select_tier_returns_valid_tier(self):
        am = AdaptiveMutator()
        tier = am.select_tier()
        assert tier in am.tiers

    def test_record_updates_posterior(self):
        am = AdaptiveMutator()
        am.record("budget", improved=True)
        assert am._successes["budget"] == 2.0  # 1 prior + 1 success
        am.record("budget", improved=False)
        assert am._failures["budget"] == 2.0  # 1 prior + 1 failure

    def test_convergence_toward_best_tier(self):
        am = AdaptiveMutator(tiers=["good", "bad"])
        # Simulate: "good" always improves, "bad" never does
        for _ in range(50):
            am.record("good", improved=True)
            am.record("bad", improved=False)

        # After strong evidence, "good" should be selected much more often
        selections = {"good": 0, "bad": 0}
        for _ in range(100):
            tier = am.select_tier()
            selections[tier] += 1

        assert selections["good"] > 80  # Should be heavily favored

    def test_success_rate(self):
        am = AdaptiveMutator(tiers=["a"])
        am.record("a", improved=True)
        am.record("a", improved=True)
        am.record("a", improved=False)
        # Prior: alpha=1, beta=1. After: alpha=3, beta=2. Rate = 3/5 = 0.6
        assert am.success_rate("a") == pytest.approx(0.6, abs=0.01)

    def test_stats(self):
        am = AdaptiveMutator(tiers=["x", "y"])
        am.record("x", improved=True)
        s = am.stats()
        assert "x" in s
        assert "y" in s
        assert s["x"]["successes"] == 2.0
        assert s["x"]["success_rate"] > 0.5

    def test_custom_tiers(self):
        am = AdaptiveMutator(tiers=["alpha", "beta", "gamma"])
        assert len(am.tiers) == 3
        tier = am.select_tier()
        assert tier in ["alpha", "beta", "gamma"]

    def test_record_unknown_tier_adds_it(self):
        am = AdaptiveMutator(tiers=["a"])
        am.record("new_tier", improved=True)
        assert "new_tier" in am._successes
