"""Tool registry for managing available tools."""
from __future__ import annotations

from sage.tools.base import Tool


class ToolRegistry:
    """Registry for managing tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def search(self, query: str) -> list[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        return [
            tool
            for tool in self._tools.values()
            if query_lower in tool.spec.name.lower()
            or query_lower in tool.spec.description.lower()
        ]

    def get_tool_defs(self, names: list[str] | None = None) -> list:
        """Get ToolDef list for LLM calls."""
        if names is None:
            return [t.spec for t in self._tools.values()]
        return [self._tools[n].spec for n in names if n in self._tools]
