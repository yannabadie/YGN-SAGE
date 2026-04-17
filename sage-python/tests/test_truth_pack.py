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
