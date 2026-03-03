"""Meta-tools: tools that create and manage other tools dynamically.

This implements the self-programming aspect of YGN-SAGE, allowing agents
to synthesize new tools at runtime from natural language descriptions.
"""
from __future__ import annotations

import textwrap
from typing import Any

from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry
from sage.sandbox.manager import SandboxManager, Sandbox


class MetaToolFactory:
    """Factory for creating meta-tools that allow agents to build new tools."""

    def __init__(self, registry: ToolRegistry, sandbox_manager: SandboxManager):
        self._registry = registry
        self._sandbox = sandbox_manager

    def create_tool_from_code(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        code: str,
    ) -> Tool:
        """Create a tool from Python code string.

        The code must define an async function named `execute` that
        takes keyword arguments matching the parameters schema.
        """
        # Compile the code to verify syntax
        compiled = compile(code, f"<tool:{name}>", "exec")

        namespace: dict[str, Any] = {}
        exec(compiled, namespace)

        if "execute" not in namespace:
            raise ValueError(f"Tool code must define an 'execute' function. Got: {list(namespace.keys())}")

        handler = namespace["execute"]
        tool = Tool(
            spec=__import__("sage.llm.base", fromlist=["ToolDef"]).ToolDef(
                name=name,
                description=description,
                parameters=parameters,
            ),
            handler=handler,
        )
        return tool

    def register_dynamic_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        code: str,
    ) -> str:
        """Create and register a new tool. Returns the tool name."""
        tool = self.create_tool_from_code(name, description, parameters, code)
        self._registry.register(tool)
        return name

    def build_create_tool_tool(self) -> Tool:
        """Build the 'create_tool' meta-tool that agents can use to create new tools."""
        factory = self

        @Tool.define(
            name="create_tool",
            description=(
                "Create a new tool from Python code. The code must define an async function "
                "named 'execute' that takes keyword arguments. The tool will be registered "
                "and immediately available for use."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the new tool"},
                    "description": {"type": "string", "description": "What the tool does"},
                    "parameters": {
                        "type": "object",
                        "description": "JSON Schema for the tool's parameters",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code defining an async 'execute' function",
                    },
                },
                "required": ["name", "description", "parameters", "code"],
            },
        )
        async def create_tool(
            name: str, description: str, parameters: dict, code: str
        ) -> str:
            try:
                factory.register_dynamic_tool(name, description, parameters, code)
                return f"Tool '{name}' created and registered successfully."
            except Exception as e:
                return f"Error creating tool: {e}"

        return create_tool

    def build_list_tools_tool(self) -> Tool:
        """Build the 'list_tools' meta-tool."""
        registry = self._registry

        @Tool.define(
            name="list_tools",
            description="List all currently available tools with their descriptions.",
            parameters={"type": "object", "properties": {}},
        )
        async def list_tools() -> str:
            tools = registry.get_tool_defs()
            lines = [f"- {t.name}: {t.description}" for t in tools]
            return "\n".join(lines) if lines else "No tools registered."

        return list_tools

    def build_search_tools_tool(self) -> Tool:
        """Build the 'search_tools' meta-tool."""
        registry = self._registry

        @Tool.define(
            name="search_tools",
            description="Search for tools by name or description keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )
        async def search_tools(query: str) -> str:
            results = registry.search(query)
            if not results:
                return f"No tools found matching '{query}'."
            lines = [f"- {t.spec.name}: {t.spec.description}" for t in results]
            return "\n".join(lines)

        return search_tools

    def register_all_meta_tools(self) -> list[str]:
        """Register all meta-tools and return their names."""
        tools = [
            self.build_create_tool_tool(),
            self.build_list_tools_tool(),
            self.build_search_tools_tool(),
        ]
        for tool in tools:
            self._registry.register(tool)
        return [t.spec.name for t in tools]
