"""Tests for non-circular routing quality benchmark (RTG-01).

Baseline (heuristic-only, March 2026):
  S1: 15/15 correct (100%)
  S2:  2/15 correct (13%)  — only "implement" keyword triggers S2
  S3:  0/15 correct (0%)   — no single task hits both keyword groups
  Overall: 17/45 (37.8%)   — reveals heuristic under-routing weakness

These thresholds are intentionally loose: the benchmark exists to measure
the gap between heuristic and LLM-based routing, not to gate CI.
"""
from sage.bench.routing_quality import run_routing_quality, GROUND_TRUTH


def test_ground_truth_has_all_levels():
    """Ground truth must have tasks at S1, S2, and S3 levels."""
    levels = {t[1] for t in GROUND_TRUTH}
    assert levels == {1, 2, 3}


def test_ground_truth_minimum_size():
    """Ground truth must have at least 45 tasks."""
    assert len(GROUND_TRUTH) >= 45


def test_benchmark_runs_without_error():
    """Benchmark completes and returns valid results."""
    result = run_routing_quality()
    assert result.total == len(GROUND_TRUTH)
    assert result.correct >= 0
    assert result.under_routed >= 0
    assert result.over_routed >= 0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.correct + result.under_routed <= result.total


def test_s1_accuracy():
    """Heuristic should route all S1 tasks correctly (trivial detection)."""
    result = run_routing_quality()
    s1_details = [d for d in result.details if d["expected_min"] == 1]
    s1_correct = sum(1 for d in s1_details if d["correct"])
    assert s1_correct == len(s1_details), (
        f"S1 accuracy {s1_correct}/{len(s1_details)}: "
        "heuristic should handle trivial tasks"
    )


def test_accuracy_above_baseline():
    """Router must meet heuristic baseline (~38%). Increase as routing improves."""
    result = run_routing_quality()
    assert result.accuracy >= 0.33, (
        f"Router accuracy {result.accuracy:.1%} below 33% baseline. "
        f"Under-routed: {result.under_routed}, Over-routed: {result.over_routed}"
    )


def test_over_routing_below_threshold():
    """Over-routing (wasting expensive models) must stay below 10%."""
    result = run_routing_quality()
    assert result.over_routing_rate < 0.10, (
        f"Over-routing rate {result.over_routing_rate:.1%} exceeds 10% threshold"
    )
