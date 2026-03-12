"""Agents-as-Tools: wrap any agent as a callable tool for dynamic delegation.

Universal pattern from OpenAI Agents SDK, Google ADK, and CrewAI — exposes
any object with an ``async run(task: str) -> str`` method as a Tool that other
agents can invoke via the standard tool-call protocol.

Usage::

    from sage.tools.agent_tool import AgentTool

    researcher = SomeAgent(...)       # anything with async run(task)->str
    tool = AgentTool.from_agent(researcher, name="researcher",
                                description="Deep research on a topic")
    registry.register(tool)           # now available to any agent
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from sage.llm.base import ToolDef
from sage.tools.base import Tool, ToolResult


@runtime_checkable
class RunnableAgent(Protocol):
    """Minimal protocol for an agent that can be wrapped as a tool."""

    async def run(self, task: str) -> str: ...


class AgentTool(Tool):
    """A Tool backed by an agent — delegates execution to ``agent.run()``.

    The tool accepts a single parameter ``task`` (string) and returns the
    agent's text output. Errors are caught and returned as ToolResult with
    ``is_error=True``.
    """

    def __init__(self, agent: Any, name: str, description: str) -> None:
        spec = ToolDef(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to delegate to this agent.",
                    }
                },
                "required": ["task"],
            },
        )
        self._agent = agent
        # Initialize Tool base with spec and handler
        super().__init__(spec=spec, handler=self._handle)

    async def _handle(self, task: str) -> str:
        """Execute the wrapped agent and return its output."""
        return await self._agent.run(task)

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        name: str,
        description: str,
    ) -> AgentTool:
        """Create an AgentTool from any object with ``async run(task) -> str``.

        Parameters
        ----------
        agent:
            The agent to wrap. Must have an ``async run(task: str) -> str`` method.
        name:
            Tool name (used in LLM tool-call protocol).
        description:
            Human-readable description (shown to the LLM).
        """
        if not hasattr(agent, "run") or not callable(getattr(agent, "run")):
            raise TypeError(
                f"Agent must have a callable 'run' method, got {type(agent).__name__}"
            )
        return cls(agent=agent, name=name, description=description)
