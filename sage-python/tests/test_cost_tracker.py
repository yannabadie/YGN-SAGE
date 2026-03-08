"""Tests for CostTracker negative-cost clamping."""
from __future__ import annotations

from sage.contracts.cost_tracker import CostTracker


def test_cost_never_negative():
    """Incremental cost should never be negative."""
    tracker = CostTracker(budget_usd=10.0)
    tracker.record("node1", 0.005)
    tracker.record("node2", -0.001)  # negative should be clamped
    assert tracker.cost_for("node2") >= 0.0
    assert tracker.total_spent >= 0.0
    # Only node1 should contribute to total
    assert abs(tracker.total_spent - 0.005) < 1e-9
