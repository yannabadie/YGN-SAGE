"""Tests for BenchmarkManifest and TaskTrace."""
from __future__ import annotations

import json
import pytest
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace


def test_manifest_creation():
    m = BenchmarkManifest(benchmark="humaneval", model="gemini-3.1-flash-lite")
    assert m.benchmark == "humaneval"
    assert m.git_sha is not None


def test_task_trace_serialization():
    t = TaskTrace(
        task_id="HumanEval/0", passed=True,
        latency_ms=1200, cost_usd=0.001,
        model="gemini-3.1-flash-lite", routing="S2",
    )
    d = t.to_dict()
    assert d["task_id"] == "HumanEval/0"
    assert d["passed"] is True


def test_manifest_add_trace_and_export():
    m = BenchmarkManifest(benchmark="humaneval", model="test-model")
    m.add(TaskTrace(task_id="t1", passed=True, latency_ms=100, cost_usd=0.0))
    m.add(TaskTrace(task_id="t2", passed=False, latency_ms=200, cost_usd=0.0))
    export = m.to_jsonl()
    lines = export.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["task_id"] == "t1"


def test_manifest_summary():
    m = BenchmarkManifest(benchmark="test", model="m")
    m.add(TaskTrace(task_id="a", passed=True, latency_ms=100, cost_usd=0.01))
    m.add(TaskTrace(task_id="b", passed=False, latency_ms=300, cost_usd=0.02))
    s = m.summary()
    assert s["total"] == 2
    assert s["passed"] == 1
    assert s["pass_rate"] == 0.5


@pytest.mark.asyncio
async def test_routing_bench_produces_manifest():
    """Routing benchmark should populate a manifest with 30 traces."""
    from sage.strategy.metacognition import ComplexityRouter
    from sage.bench.routing import RoutingAccuracyBench, LABELED_TASKS

    mc = ComplexityRouter()
    bench = RoutingAccuracyBench(metacognition=mc)
    report = await bench.run()

    # Manifest should exist and have one trace per labeled task
    assert bench.manifest is not None
    assert len(bench.manifest.traces) == len(LABELED_TASKS)
    assert bench.manifest.benchmark == "routing_accuracy"
    assert bench.manifest.model == "heuristic"

    # Every trace should have a valid routing tier
    for trace in bench.manifest.traces:
        assert trace.routing in ("S1", "S2", "S3")
        assert trace.task_id.startswith("routing_")
        assert trace.latency_ms >= 0

    # JSONL export should produce one line per trace
    jsonl = bench.manifest.to_jsonl()
    lines = jsonl.strip().split("\n")
    assert len(lines) == len(LABELED_TASKS)
    for line in lines:
        parsed = json.loads(line)
        assert "task_id" in parsed
        assert "passed" in parsed

    # Summary should match report totals
    summary = bench.manifest.summary()
    assert summary["total"] == report.total
    assert summary["passed"] == report.passed
