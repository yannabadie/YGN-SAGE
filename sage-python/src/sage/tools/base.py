"""Base tool types and decorator."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from sage.llm.base import ToolDef


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class Tool:
    """A tool that an agent can use."""

    def __init__(self, spec: ToolDef, handler: Callable[..., Awaitable[str]]):
        self.spec = spec
        self._handler = handler

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            output = await self._handler(**arguments)
            return ToolResult(output=output, is_error=False)
        except Exception as e:
            return ToolResult(
                output=f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                is_error=True,
            )

    @staticmethod
    def define(
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable[[Callable[..., Awaitable[str]]], Tool]:
        """Decorator to define a tool from an async function."""

        def decorator(func: Callable[..., Awaitable[str]]) -> Tool:
            spec = ToolDef(name=name, description=description, parameters=parameters)
            return Tool(spec=spec, handler=func)

        return decorator
