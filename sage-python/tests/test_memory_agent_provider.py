"""Tests for MemoryAgent provider injection (no vendor lock-in)."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from sage.memory.memory_agent import MemoryAgent
from sage.llm.base import LLMResponse


def test_memory_agent_accepts_provider_kwarg():
    """Constructor must accept llm_provider keyword argument."""
    mock = MagicMock()
    agent = MemoryAgent(use_llm=True, llm_provider=mock)
    assert agent._llm_provider is mock


def test_memory_agent_default_provider_is_none():
    """Without llm_provider, _llm_provider should be None."""
    agent = MemoryAgent(use_llm=True)
    assert agent._llm_provider is None


@pytest.mark.asyncio
async def test_memory_agent_uses_injected_provider():
    """MemoryAgent should use the injected LLM provider, not hardcoded Google."""
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=LLMResponse(
        content='{"entities": ["Python"], "relationships": [["Python", "is", "language"]], "summary": "About Python"}',
        model="test-model",
    ))

    agent = MemoryAgent(use_llm=True, llm_provider=mock_provider)
    result = await agent.extract("Python is a programming language")

    mock_provider.generate.assert_awaited_once()
    assert "Python" in result.entities


@pytest.mark.asyncio
async def test_memory_agent_heuristic_without_provider():
    """Without provider and use_llm=False, heuristic extraction works."""
    agent = MemoryAgent(use_llm=False)
    result = await agent.extract("Python is great")
    assert isinstance(result.entities, list)
