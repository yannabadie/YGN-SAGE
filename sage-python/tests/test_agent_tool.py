"""Tests for Agents-as-Tools wrapper."""
import pytest
from sage.tools.agent_tool import AgentTool


class FakeAgent:
    """Minimal agent with async run()."""

    def __init__(self, response: str = "done"):
        self._response = response

    async def run(self, task: str) -> str:
        return f"{self._response}: {task}"


class NotAnAgent:
    """Object without run() method."""
    pass


@pytest.mark.asyncio
async def test_agent_tool_executes():
    agent = FakeAgent("result")
    tool = AgentTool.from_agent(agent, name="fake", description="A fake agent")

    result = await tool.execute({"task": "hello"})
    assert not result.is_error
    assert result.output == "result: hello"


@pytest.mark.asyncio
async def test_agent_tool_spec():
    agent = FakeAgent()
    tool = AgentTool.from_agent(agent, name="researcher", description="Research tool")

    assert tool.spec.name == "researcher"
    assert tool.spec.description == "Research tool"
    assert "task" in tool.spec.parameters["properties"]
    assert tool.spec.parameters["required"] == ["task"]


def test_agent_tool_rejects_non_agent():
    with pytest.raises(TypeError, match="callable 'run' method"):
        AgentTool.from_agent(NotAnAgent(), name="bad", description="bad")


@pytest.mark.asyncio
async def test_agent_tool_error_handling():
    class FailingAgent:
        async def run(self, task: str) -> str:
            raise ValueError("boom")

    tool = AgentTool.from_agent(FailingAgent(), name="fail", description="fails")
    result = await tool.execute({"task": "test"})
    assert result.is_error
    assert "ValueError" in result.output
    assert "boom" in result.output
