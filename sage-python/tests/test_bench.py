"""Tests for the benchmark pipeline (sage.bench)."""

import sys
import types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import json

import pytest

from sage.bench.runner import BenchReport, BenchmarkRunner, TaskResult
from sage.bench.routing import RoutingAccuracyBench


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

def test_task_result_creation():
    tr = TaskResult(task_id="t1", passed=True, system_used=2, latency_ms=42.0)
    assert tr.task_id == "t1"
    assert tr.passed is True
    assert tr.system_used == 2
    assert tr.latency_ms == 42.0
    assert tr.cost_usd == 0.0
    assert tr.error == ""


def test_task_result_defaults():
    tr = TaskResult(task_id="t2", passed=False)
    assert tr.system_used == 0
    assert tr.latency_ms == 0.0
    assert tr.cost_usd == 0.0
    assert tr.sandbox_executions == 0
    assert tr.memory_events == 0
    assert tr.escalations == 0
    assert tr.z3_checks == 0
    assert tr.tokens_used == 0
    assert tr.error == ""


# ---------------------------------------------------------------------------
# BenchReport
# ---------------------------------------------------------------------------

def test_bench_report_creation():
    report = BenchReport(
        benchmark="test",
        total=10,
        passed=8,
        failed=2,
        errors=0,
        pass_rate=0.8,
        avg_latency_ms=5.0,
        avg_cost_usd=0.001,
        routing_breakdown={"S1": 4, "S2": 3, "S3": 3},
        results=[],
        model_config={"tier": "fast"},
    )
    assert report.benchmark == "test"
    assert report.total == 10
    assert report.pass_rate == 0.8
    assert report.routing_breakdown["S1"] == 4


def test_bench_report_from_results():
    results = [
        TaskResult(task_id="a", passed=True, system_used=1, latency_ms=10.0, cost_usd=0.001),
        TaskResult(task_id="b", passed=True, system_used=2, latency_ms=20.0, cost_usd=0.002),
        TaskResult(task_id="c", passed=False, system_used=3, latency_ms=30.0, cost_usd=0.003),
        TaskResult(task_id="d", passed=False, system_used=2, latency_ms=40.0, cost_usd=0.004, error="timeout"),
    ]
    report = BenchReport.from_results("test_bench", results, model_config={"tier": "fast"})
    assert report.total == 4
    assert report.passed == 2
    assert report.failed == 2
    assert report.errors == 1  # one result has a non-empty error
    assert report.pass_rate == pytest.approx(0.5)
    assert report.avg_latency_ms == pytest.approx(25.0)
    assert report.avg_cost_usd == pytest.approx(0.0025)
    assert report.routing_breakdown == {"S1": 1, "S2": 2, "S3": 1}
    assert report.benchmark == "test_bench"
    assert report.model_config == {"tier": "fast"}
    assert report.timestamp != ""


def test_bench_report_from_results_empty():
    """Division by zero protection on empty list."""
    report = BenchReport.from_results("empty", [])
    assert report.total == 0
    assert report.passed == 0
    assert report.pass_rate == 0.0
    assert report.avg_latency_ms == 0.0
    assert report.avg_cost_usd == 0.0
    assert report.routing_breakdown == {"S1": 0, "S2": 0, "S3": 0}


def test_bench_report_from_results_no_model_config():
    results = [TaskResult(task_id="x", passed=True, system_used=1, latency_ms=5.0)]
    report = BenchReport.from_results("minimal", results)
    assert report.model_config == {}


# ---------------------------------------------------------------------------
# RoutingAccuracyBench
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routing_accuracy_bench():
    from sage.strategy.metacognition import MetacognitiveController

    mc = MetacognitiveController()
    bench = RoutingAccuracyBench(metacognition=mc)
    report = await bench.run()

    assert report.total == 30
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.avg_latency_ms >= 0
    assert report.benchmark == "routing_accuracy"
    # Verify routing_breakdown has S1, S2, S3 keys
    assert all(k in report.routing_breakdown for k in ["S1", "S2", "S3"])
    # Every result should have a system_used in {1, 2, 3}
    for r in report.results:
        assert r.system_used in (1, 2, 3)


@pytest.mark.asyncio
async def test_routing_accuracy_bench_labels_count():
    """LABELED_TASKS should have exactly 30 entries (10 per system)."""
    from sage.bench.routing import LABELED_TASKS

    assert len(LABELED_TASKS) == 30
    s1_count = sum(1 for t in LABELED_TASKS if t["expected"] == 1)
    s2_count = sum(1 for t in LABELED_TASKS if t["expected"] == 2)
    s3_count = sum(1 for t in LABELED_TASKS if t["expected"] == 3)
    assert s1_count == 10
    assert s2_count == 10
    assert s3_count == 10


def test_bench_report_json_serializable():
    """BenchReport should be JSON-serializable via __dict__."""
    results = [TaskResult(task_id="z", passed=True, system_used=1, latency_ms=1.0)]
    report = BenchReport.from_results("json_test", results)
    # Should not raise
    import dataclasses
    data = dataclasses.asdict(report)
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed["benchmark"] == "json_test"
    assert parsed["total"] == 1
