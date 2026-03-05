"""Tests for the agent runtime."""
import pytest
from unittest.mock import AsyncMock
from sage.agent import Agent, AgentConfig
from sage.llm.base import LLMConfig, LLMResponse, Message, Role
from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.name = "mock"
    provider.generate = AsyncMock(
        return_value=LLMResponse(content="I'm done.", tool_calls=[])
    )
    return provider
@pytest.fixture
def basic_config():
    return AgentConfig(
        name="test-agent",
        llm=LLMConfig(provider="mock", model="mock-model"),
        system_prompt="You are a helpful assistant.",
        max_steps=5,
        validation_level=1
    )


def test_agent_creation(basic_config, mock_llm):
    agent = Agent(config=basic_config, llm_provider=mock_llm)
    assert agent.config.name == "test-agent"
    assert agent.step_count == 0


@pytest.mark.asyncio
async def test_agent_simple_response(basic_config, mock_llm):
    agent = Agent(config=basic_config, llm_provider=mock_llm)
    result = await agent.run("Hello!")
    assert result == "I'm done."
    assert agent.step_count == 1


@pytest.mark.asyncio
async def test_agent_with_tool_use(basic_config, mock_llm):
    from sage.llm.base import ToolCall as LLMToolCall

    # First call returns tool use, second returns final response
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[LLMToolCall(id="tc1", name="greet", arguments={"name": "World"})],
            ),
            LLMResponse(content="The greeting is: Hello, World!", tool_calls=[]),
        ]
    )

    @Tool.define(
        name="greet",
        description="Greet someone",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry = ToolRegistry()
    registry.register(greet)

    agent = Agent(config=basic_config, llm_provider=mock_llm, tool_registry=registry)
    result = await agent.run("Greet the world")

    assert "Hello, World!" in result
    assert agent.step_count == 2


@pytest.mark.asyncio
async def test_agent_max_steps(basic_config, mock_llm):
    from sage.llm.base import ToolCall as LLMToolCall

    basic_config.max_steps = 2

    # Always returns tool calls, never stops
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc", name="greet", arguments={"name": "X"})],
        )
    )

    @Tool.define(name="greet", description="G", parameters={"type": "object", "properties": {"name": {"type": "string"}}})
    async def greet(name: str) -> str:
        return f"Hi {name}"

    registry = ToolRegistry()
    registry.register(greet)

    agent = Agent(config=basic_config, llm_provider=mock_llm, tool_registry=registry)
    result = await agent.run("go")

    assert agent.step_count == 2  # Stopped at max_steps
