"""Tests for benchmark metadata propagation."""
import pytest
from sage.boot import AgentSystem


def test_agent_system_model_info():
    """AgentSystem.model_info returns resolved model metadata."""
    from unittest.mock import MagicMock

    system = AgentSystem.__new__(AgentSystem)

    # Minimal mock: agent_loop with a fake _llm, and a metacognition stub
    mock_llm = MagicMock()
    mock_llm.model_id = "gemini-2.5-flash"
    type(mock_llm).__name__ = "GoogleProvider"

    mock_loop = MagicMock()
    mock_loop._llm = mock_llm

    system.agent_loop = mock_loop  # type: ignore[attr-defined]

    mock_meta = MagicMock()
    mock_meta._current_tier = "budget"
    system.metacognition = mock_meta  # type: ignore[attr-defined]

    info = system.model_info

    assert isinstance(info, dict), "model_info must return a dict"
    assert "model" in info, "model_info must contain 'model' key"
    assert "provider" in info, "model_info must contain 'provider' key"
    assert "tier" in info, "model_info must contain 'tier' key"
    assert info["model"] == "gemini-2.5-flash"
    assert info["provider"] == "GoogleProvider"
    assert info["tier"] == "budget"


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
