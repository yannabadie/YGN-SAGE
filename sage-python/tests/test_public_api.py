"""Tests for the public SDK API surface."""
import pytest


def test_top_level_imports():
    """All public symbols importable from sage."""
    from sage import Agent, AgentConfig, LLMConfig, Tool, ToolRegistry, ToolResult
    from sage import create, __version__
    assert __version__ == "0.1.0"
    assert callable(create)


@pytest.mark.asyncio
async def test_create_returns_agent_system():
    """sage.create() returns a usable AgentSystem with mock LLM."""
    from sage import create
    system = await create(mock=True)
    assert hasattr(system, "run")
    result = await system.run("Hello")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_create_with_tools():
    """sage.create() accepts custom tools."""
    from sage import create, Tool

    @Tool.define(name="greet", description="Greet someone", parameters={"name": {"type": "string"}})
    async def greet(name: str = "world") -> str:
        return f"Hello, {name}!"

    system = await create(mock=True, tools=[greet])
    # Verify tool was registered
    tool_names = [t.spec.name for t in system.tool_registry._tools.values()]
    assert "greet" in tool_names
