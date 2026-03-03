"""Tests for meta-tools: dynamic tool creation and management."""
import pytest
from sage.tools.meta import MetaToolFactory
from sage.tools.registry import ToolRegistry
from sage.sandbox.manager import SandboxManager


@pytest.fixture
def factory():
    registry = ToolRegistry()
    sandbox = SandboxManager(use_docker=False)
    return MetaToolFactory(registry=registry, sandbox_manager=sandbox)


def test_create_tool_from_code(factory):
    code = '''
async def execute(x: int, y: int) -> str:
    return str(x + y)
'''
    tool = factory.create_tool_from_code(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
        code=code,
    )
    assert tool.spec.name == "add"


@pytest.mark.asyncio
async def test_dynamic_tool_execution(factory):
    code = '''
async def execute(name: str) -> str:
    return f"Hello, {name}!"
'''
    tool = factory.create_tool_from_code(
        name="greet",
        description="Greet someone",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
        code=code,
    )
    result = await tool.execute({"name": "SAGE"})
    assert result.output == "Hello, SAGE!"
    assert not result.is_error


def test_register_dynamic_tool(factory):
    code = 'async def execute() -> str:\n    return "ok"'
    name = factory.register_dynamic_tool(
        name="test_tool",
        description="Test",
        parameters={"type": "object"},
        code=code,
    )
    assert name == "test_tool"
    assert factory._registry.get("test_tool") is not None


def test_create_tool_missing_execute(factory):
    with pytest.raises(ValueError, match="must define an 'execute' function"):
        factory.create_tool_from_code(
            name="bad",
            description="Bad tool",
            parameters={"type": "object"},
            code="x = 42",
        )


def test_create_tool_syntax_error(factory):
    with pytest.raises(SyntaxError):
        factory.create_tool_from_code(
            name="bad",
            description="Bad tool",
            parameters={"type": "object"},
            code="def execute(:: broken",
        )


@pytest.mark.asyncio
async def test_meta_tool_create_tool(factory):
    create_tool = factory.build_create_tool_tool()
    result = await create_tool.execute({
        "name": "multiply",
        "description": "Multiply two numbers",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
        "code": 'async def execute(a, b) -> str:\n    return str(a * b)',
    })
    assert "successfully" in result.output
    assert factory._registry.get("multiply") is not None


@pytest.mark.asyncio
async def test_meta_tool_list_tools(factory):
    factory.register_all_meta_tools()
    list_tool = factory._registry.get("list_tools")
    result = await list_tool.execute({})
    assert "create_tool" in result.output
    assert "list_tools" in result.output
    assert "search_tools" in result.output


@pytest.mark.asyncio
async def test_meta_tool_search_tools(factory):
    factory.register_all_meta_tools()
    search_tool = factory._registry.get("search_tools")
    result = await search_tool.execute({"query": "create"})
    assert "create_tool" in result.output


def test_register_all_meta_tools(factory):
    names = factory.register_all_meta_tools()
    assert "create_tool" in names
    assert "list_tools" in names
    assert "search_tools" in names
