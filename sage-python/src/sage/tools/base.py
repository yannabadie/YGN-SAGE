"""Base tool types and decorator."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from sage.llm.base import ToolDef

log = logging.getLogger(__name__)


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
        """Execute the tool with given arguments.

        Exception handling (2026-04-23, ALIRE2 §6 "Traceback leakage"):
        the model-visible ToolResult.output contains ONLY the exception
        type + message. The full traceback goes to the operator-side
        log via log.exception — not into the agent's prompt, where it
        previously leaked internal file paths, module names, and other
        structural info about the host. Exception type names are kept
        because the model often steers on them (e.g. FileNotFoundError
        → "create it first" vs PermissionError → "request approval").
        """
        try:
            output = await self._handler(**arguments)
            return ToolResult(output=output, is_error=False)
        except Exception as e:
            log.exception("Tool %s raised %s", self.spec.name, type(e).__name__)
            return ToolResult(
                output=f"Error: {type(e).__name__}: {e}",
                is_error=True,
            )

    async def run(self, arguments: dict[str, Any]) -> str:
        """Execute and return raw output string."""
        result = await self.execute(arguments)
        return result.output

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
