"""Tests for quality-gated cascade in ModelAgent."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch


class FakeModel:
    """Minimal ModelProfile stub."""
    def __init__(self, id: str, provider: str = "google", cost_input: float = 0.001):
        self.id = id
        self.provider = provider
        self.cost_input = cost_input
        self.cost_output = cost_input


class FakeRegistry:
    """Minimal ModelRegistry stub."""
    def __init__(self, models: list[FakeModel]):
        self._models = models

    def list_available(self) -> list[FakeModel]:
        return self._models


@pytest.mark.asyncio
async def test_quality_cascade_escalates_on_low_quality():
    """Low quality response from cheap model → escalate to better model."""
    from sage.orchestrator import ModelAgent

    cheap = FakeModel("gemini-flash-lite", cost_input=0.0003)
    better = FakeModel("gemini-flash", cost_input=0.001)
    registry = FakeRegistry([cheap, better])

    agent = ModelAgent(
        name="test",
        model=cheap,
        registry=registry,
        quality_threshold=0.6,
    )

    with patch.object(agent, "_call_provider", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = ["ok", "Here is a detailed, correct implementation..."]
        result = await agent.run("implement sort")

    assert "detailed" in result
    assert mock_call.call_count == 2  # cheap tried first, then escalated


@pytest.mark.asyncio
async def test_quality_cascade_stops_when_quality_sufficient():
    """Good quality response from cheap model → no escalation (cost saved)."""
    from sage.orchestrator import ModelAgent

    cheap = FakeModel("gemini-flash-lite", cost_input=0.0003)
    better = FakeModel("gemini-flash", cost_input=0.001)
    registry = FakeRegistry([cheap, better])

    agent = ModelAgent(
        name="test",
        model=cheap,
        registry=registry,
        quality_threshold=0.5,
    )

    with patch.object(agent, "_call_provider", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (
            "Here is a complete function implementation:\n"
            "```python\ndef sort(arr):\n    return sorted(arr)\n```\n"
            "This handles edge cases correctly."
        )
        result = await agent.run("implement sort")

    assert mock_call.call_count == 1  # No escalation needed!
    assert "sort" in result


@pytest.mark.asyncio
async def test_quality_cascade_falls_back_on_exception():
    """Exception still triggers fallback (defense in depth)."""
    from sage.orchestrator import ModelAgent

    cheap = FakeModel("gemini-flash-lite", cost_input=0.0003)
    better = FakeModel("gemini-flash", cost_input=0.001)
    registry = FakeRegistry([cheap, better])

    agent = ModelAgent(
        name="test",
        model=cheap,
        registry=registry,
        quality_threshold=0.6,
    )

    with patch.object(agent, "_call_provider", new_callable=AsyncMock) as mock_call:
        mock_call.side_effect = [
            Exception("Rate limited"),
            "Fallback answer with enough detail to pass quality check.",
        ]
        result = await agent.run("task")

    assert mock_call.call_count == 2
    assert "Fallback" in result


@pytest.mark.asyncio
async def test_quality_cascade_disabled_when_no_threshold():
    """When quality_threshold is None, behave like original (exception-only)."""
    from sage.orchestrator import ModelAgent

    model = FakeModel("gemini-flash", cost_input=0.001)

    agent = ModelAgent(
        name="test",
        model=model,
        quality_threshold=None,
    )

    with patch.object(agent, "_call_provider", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = "x"  # Low quality but no cascade
        result = await agent.run("task")

    assert mock_call.call_count == 1
    assert result == "x"


@pytest.mark.asyncio
async def test_orchestrator_wires_quality_thresholds():
    """CognitiveOrchestrator sets different thresholds per cognitive system."""
    from sage.orchestrator import ModelAgent

    model = FakeModel("test-model")
    agent_s1 = ModelAgent(name="s1", model=model, quality_threshold=0.4)
    agent_s2 = ModelAgent(name="s2", model=model, quality_threshold=0.6)
    agent_s3 = ModelAgent(name="s3", model=model, quality_threshold=0.8)

    assert agent_s1._quality_threshold == 0.4
    assert agent_s2._quality_threshold == 0.6
    assert agent_s3._quality_threshold == 0.8
