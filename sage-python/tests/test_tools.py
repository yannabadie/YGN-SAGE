"""Tests for the tool system."""
import pytest
from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry


def test_tool_creation():
    @Tool.define(
        name="greet",
        description="Greet someone",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet.spec.name == "greet"
    assert greet.spec.description == "Greet someone"


@pytest.mark.asyncio
async def test_tool_execution():
    @Tool.define(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    async def add(a: float, b: float) -> str:
        return str(a + b)

    result = await add.execute({"a": 2, "b": 3})
    assert isinstance(result, ToolResult)
    assert result.output == "5"
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_error_handling():
    @Tool.define(name="fail", description="Always fails", parameters={"type": "object"})
    async def fail() -> str:
        raise ValueError("intentional error")

    result = await fail.execute({})
    assert result.is_error
    assert "intentional error" in result.output


def test_registry_register_and_list():
    registry = ToolRegistry()

    @Tool.define(name="tool_a", description="A", parameters={"type": "object"})
    async def tool_a() -> str:
        return "a"

    registry.register(tool_a)
    assert "tool_a" in registry.list_tools()


def test_registry_get_tool():
    registry = ToolRegistry()

    @Tool.define(name="my_tool", description="Mine", parameters={"type": "object"})
    async def my_tool() -> str:
        return "mine"

    registry.register(my_tool)
    tool = registry.get("my_tool")
    assert tool is not None
    assert tool.spec.name == "my_tool"


def test_registry_search():
    registry = ToolRegistry()

    @Tool.define(name="bash", description="Execute bash commands", parameters={"type": "object"})
    async def bash() -> str:
        return ""

    @Tool.define(name="python", description="Execute Python code", parameters={"type": "object"})
    async def python() -> str:
        return ""

    registry.register(bash)
    registry.register(python)

    results = registry.search("execute")
    assert len(results) == 2

    results = registry.search("bash")
    assert len(results) == 1
