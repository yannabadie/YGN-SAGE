"""Tests for downstream routing quality evaluator."""
import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.bench.routing_downstream import DownstreamEvaluator, TierMetrics


def test_tier_metrics_precision():
    tm = TierMetrics(tier=1, total=10, successes=8)
    assert tm.precision == 0.8


def test_tier_metrics_empty():
    tm = TierMetrics(tier=1)
    assert tm.precision == 0.0
    assert tm.avg_latency_ms == 0.0


def test_evaluator_records():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, latency_ms=50, cost_usd=0.001, routing_ms=1.0)
    ev.record(tier=2, quality=0.7, latency_ms=200, cost_usd=0.01, routing_ms=5.0)
    ev.record(tier=1, quality=0.3, latency_ms=30, cost_usd=0.001, routing_ms=0.5)
    result = ev.result()
    assert result.total_tasks == 3
    assert result.escalation_rate == 0.0
    assert len(result.tier_metrics) == 2
    assert result.tier_metrics[1].total == 2
    assert result.tier_metrics[1].successes == 1  # only 0.9 >= 0.5


def test_escalation_rate():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, escalated=False)
    ev.record(tier=2, quality=0.5, escalated=True)
    ev.record(tier=2, quality=0.8, escalated=False)
    assert abs(ev.result().escalation_rate - 1 / 3) < 0.01


def test_routing_percentiles():
    ev = DownstreamEvaluator()
    for ms in [1.0, 2.0, 3.0, 4.0, 100.0]:
        ev.record(tier=1, quality=0.9, routing_ms=ms)
    assert ev.result().routing_p50_ms == 3.0
    assert ev.result().routing_p99_ms == 100.0


def test_to_dict():
    ev = DownstreamEvaluator()
    ev.record(tier=1, quality=0.9, cost_usd=0.001, routing_ms=1.0)
    d = ev.result().to_dict()
    assert "total_tasks" in d
    assert "escalation_rate" in d
    assert "tier_precision" in d
    assert "routing_p50_ms" in d


def test_tier_metrics_avg_cost():
    tm = TierMetrics(tier=2, total=4, successes=3, total_cost_usd=0.04)
    assert abs(tm.avg_cost_usd - 0.01) < 1e-9


def test_empty_result_defaults():
    ev = DownstreamEvaluator()
    result = ev.result()
    assert result.total_tasks == 0
    assert result.escalation_rate == 0.0
    assert result.avg_quality == 0.0
    assert result.routing_p50_ms == 0.0
    assert result.routing_p99_ms == 0.0
