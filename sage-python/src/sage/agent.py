"""Core agent runtime for YGN-SAGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    llm: LLMConfig
    system_prompt: str = "You are a helpful AI assistant."
    max_steps: int = 100
    tools: list[str] | None = None  # Tool names to use (None = all in registry)


class Agent:
    """Core agent that runs the LLM -> Tool -> LLM loop."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._messages: list[Message] = []
        self.step_count: int = 0
        self.result: str | None = None

    async def run(self, task: str) -> str:
        """Execute the agent on a task. Returns the final text response."""
        # Initialize conversation
        self._messages = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.USER, content=task),
        ]

        # Get tool definitions
        tool_defs = self._get_tool_defs()

        # Agent loop
        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Call LLM
            response = await self._llm.generate(
                messages=self._messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )

            # If no tool calls, we're done
            if not response.tool_calls:
                self.result = response.content
                self._messages.append(
                    Message(role=Role.ASSISTANT, content=response.content)
                )
                return response.content

            # Add assistant message with tool calls
            self._messages.append(
                Message(role=Role.ASSISTANT, content=response.content)
            )

            # Execute tool calls
            for tc in response.tool_calls:
                tool = self._tools.get(tc.name)
                if tool is None:
                    tool_output = f"Error: Unknown tool '{tc.name}'"
                    is_error = True
                else:
                    result = await tool.execute(tc.arguments)
                    tool_output = result.output
                    is_error = result.is_error

                self._messages.append(
                    Message(
                        role=Role.TOOL,
                        content=tool_output,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        # Max steps reached
        self.result = f"Agent reached max steps ({self.config.max_steps})"
        return self.result

    def _get_tool_defs(self) -> list[ToolDef]:
        """Get tool definitions for the LLM."""
        if self.config.tools is not None:
            return self._tools.get_tool_defs(self.config.tools)
        return self._tools.get_tool_defs()
