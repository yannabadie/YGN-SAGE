"""Test MCP server wrapper."""
import pytest

mcp = pytest.importorskip("mcp", reason="mcp package not installed")


def test_create_mcp_server_returns_server():
    from sage.protocols.mcp_server import create_mcp_server
    from sage.tools.registry import ToolRegistry

    registry = ToolRegistry()
    server = create_mcp_server(tool_registry=registry)
    assert server is not None


@pytest.mark.asyncio
async def test_mcp_server_has_run_task_tool():
    from sage.protocols.mcp_server import create_mcp_server
    from sage.tools.registry import ToolRegistry

    registry = ToolRegistry()
    server = create_mcp_server(tool_registry=registry)
    # list_tools() is async in mcp SDK
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "run_task" in tool_names


@pytest.mark.asyncio
async def test_mcp_server_without_deps():
    """Server works with no tool_registry, no agent_loop, no event_bus."""
    from sage.protocols.mcp_server import create_mcp_server

    server = create_mcp_server()
    tools = await server.list_tools()
    assert len(tools) >= 1  # At least run_task


def test_mcp_server_tool_count_via_internal():
    """Tool count accessible synchronously via _tool_manager._tools."""
    from sage.protocols.mcp_server import create_mcp_server

    server = create_mcp_server()
    assert len(server._tool_manager._tools) >= 1
