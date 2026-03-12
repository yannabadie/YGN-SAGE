"""Tests for ComplexityRouter provider injection."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from sage.strategy.metacognition import ComplexityRouter
from sage.llm.base import LLMResponse


def test_complexity_router_accepts_llm_provider():
    """ComplexityRouter must accept an llm_provider parameter."""
    mock = MagicMock()
    router = ComplexityRouter(llm_provider=mock)
    assert router._llm_provider is mock


def test_complexity_router_default_provider_is_none():
    router = ComplexityRouter()
    assert router._llm_provider is None


@pytest.mark.asyncio
async def test_assess_uses_injected_provider():
    """When llm_provider is set, should use it instead of google.genai."""
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=LLMResponse(
        content='{"complexity": 0.3, "uncertainty": 0.1, "tool_required": false, "reasoning": "Simple math"}',
        model="test",
    ))

    router = ComplexityRouter(llm_provider=mock_provider)
    profile = await router.assess_complexity_async("What is 2+2?")

    assert 0.0 <= profile.complexity <= 1.0
    assert profile.reasoning == "Simple math"
    mock_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_assess_falls_back_to_heuristic_on_provider_error(monkeypatch):
    """If injected provider fails, fall back to heuristic."""
    # Remove GOOGLE_API_KEY so it doesn't try the legacy Gemini path
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=RuntimeError("API down"))

    router = ComplexityRouter(llm_provider=mock_provider)
    profile = await router.assess_complexity_async("What is 2+2?")

    assert profile.reasoning == "degraded_heuristic"
