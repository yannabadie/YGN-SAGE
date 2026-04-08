"""Tool registry for managing available tools."""
from __future__ import annotations

from sage.tools.base import Tool


class ToolRegistry:
    """Registry for managing tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._usage: dict[str, dict] = {}  # name -> {usage_count, success_count, source}

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
        """Get tool definitions in OpenAI function-calling format."""
        specs = (
            [t.spec for t in self._tools.values()]
            if names is None
            else [self._tools[n].spec for n in names if n in self._tools]
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in specs
        ]

    # ── Usage tracking (ToolForge axis) ────────────────────────────────────

    def record_usage(self, name: str, success: bool = True) -> None:
        """Record a tool invocation for usage tracking."""
        if name not in self._usage:
            self._usage[name] = {"usage_count": 0, "success_count": 0, "source": "builtin"}
        self._usage[name]["usage_count"] += 1
        if success:
            self._usage[name]["success_count"] += 1

    def get_usage(self, name: str) -> dict:
        """Get usage stats for a tool. Returns default dict if unknown."""
        return self._usage.get(name, {"usage_count": 0, "success_count": 0, "source": "builtin"})

    def mark_source(self, name: str, source: str) -> None:
        """Mark the origin of a tool (builtin, forged, user)."""
        if name not in self._usage:
            self._usage[name] = {"usage_count": 0, "success_count": 0, "source": source}
        else:
            self._usage[name]["source"] = source
