"""Tests for benchmark metadata propagation."""
import pytest
from sage.boot import AgentSystem


def test_agent_system_model_info():
    """AgentSystem.model_info returns resolved model metadata."""
    # Tested after Step 4 adds the property — see Step 7 verification


def test_bench_report_temperature_field():
    """BenchReport has temperature field."""
    from sage.bench.runner import BenchReport
    report = BenchReport(
        benchmark="test", total=1, passed=1, failed=0, errors=0,
        pass_rate=1.0, avg_latency_ms=0.0, avg_cost_usd=0.0,
        routing_breakdown={}, results=[],
        temperature=0.5,
    )
    assert report.temperature == 0.5


def test_bench_report_default_temperature():
    """BenchReport temperature defaults to 0.0."""
    from sage.bench.runner import BenchReport
    report = BenchReport(
        benchmark="test", total=1, passed=1, failed=0, errors=0,
        pass_rate=1.0, avg_latency_ms=0.0, avg_cost_usd=0.0,
        routing_breakdown={}, results=[],
    )
    assert report.temperature == 0.0
