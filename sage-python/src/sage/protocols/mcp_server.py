"""MCP (Model Context Protocol) server wrapper for YGN-SAGE.

Exposes YGN-SAGE tools via MCP protocol, enabling interoperability
with Claude Desktop, Cursor, and any MCP client.

Usage:
    from sage.protocols.mcp_server import create_mcp_server
    server = create_mcp_server(tool_registry, agent_loop, event_bus)
    server.run(transport="streamable-http", host="0.0.0.0", port=8001)
"""
from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

_log = logging.getLogger(__name__)


def create_mcp_server(
    tool_registry: Any | None = None,
    agent_loop: Any | None = None,
    event_bus: Any | None = None,
) -> FastMCP:
    """Create an MCP server exposing YGN-SAGE capabilities.

    Parameters
    ----------
    tool_registry:
        ToolRegistry instance. All registered tools are exposed via MCP.
    agent_loop:
        AgentLoop instance. If provided, adds a ``run_task`` meta-tool.
    event_bus:
        EventBus instance. If provided, exposes recent events as MCP resource.
    """
    server = FastMCP(
        "YGN-SAGE",
        json_response=True,
    )

    # Meta-tool: run a task through the full SAGE pipeline
    @server.tool()
    async def run_task(task: str, system: int = 0) -> str:  # noqa: ARG001
        """Run a task through the YGN-SAGE cognitive pipeline.

        Args:
            task: The task to execute.
            system: Force cognitive system (0=auto, 1=S1, 2=S2, 3=S3).

        Returns:
            The agent's response.
        """
        if agent_loop is None:
            return "Error: AgentLoop not configured. Start SAGE with full boot sequence."
        try:
            result = await agent_loop.run(task)
            return result if isinstance(result, str) else str(result)
        except Exception as exc:
            return f"Error: {exc}"

    # Register all tools from ToolRegistry
    if tool_registry is not None:
        for name, tool in tool_registry._tools.items():
            _register_tool_as_mcp(server, name, tool)

    # Expose EventBus as resource (read-only)
    if event_bus is not None:
        @server.resource("sage://events/recent")
        async def recent_events() -> str:
            """Recent agent events (last 20)."""
            import json
            events = event_bus.query(last_n=20)
            return json.dumps([
                {"phase": e.phase, "content": e.content[:200]}
                for e in events
            ], indent=2)

    # Count registered tools via internal dict (list_tools() is async, not usable here)
    tool_count = len(server._tool_manager._tools)
    _log.info("MCP server created with %d tools", tool_count)
    return server


def _register_tool_as_mcp(server: FastMCP, name: str, tool: Any) -> None:
    """Register a SAGE tool as an MCP tool."""
    try:
        spec = tool.spec if hasattr(tool, "spec") else None
        description = spec.description if spec else f"SAGE tool: {name}"

        @server.tool(name=name, description=description)
        async def _wrapper(**kwargs: Any) -> str:
            try:
                import inspect
                handler = tool.handler if hasattr(tool, "handler") else tool
                if inspect.iscoroutinefunction(handler):
                    result = await handler(**kwargs)
                else:
                    result = handler(**kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                return str(result)
            except Exception as exc:
                return f"Error in {name}: {exc}"
    except Exception as exc:
        _log.debug("Failed to register tool %s as MCP: %s", name, exc)
