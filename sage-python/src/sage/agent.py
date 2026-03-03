"""Core agent runtime for YGN-SAGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
from sage.memory.working import WorkingMemory
from sage.memory.compressor import MemoryCompressor
from sage.sandbox.manager import SandboxManager


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    llm: LLMConfig
    system_prompt: str = "You are a helpful AI assistant."
    max_steps: int = 100
    tools: list[str] | None = None  # Tool names to use (None = all in registry)
    use_docker_sandbox: bool = False
    snapshot_to_restore: str | None = None


class Agent:
    """Core agent that runs the LLM -> Tool -> LLM loop."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        memory_compressor: MemoryCompressor | None = None,
        sandbox_manager: SandboxManager | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._messages: list[Message] = []
        self.step_count: int = 0
        self.result: str | None = None
        
        # SOTA Integration
        self.working_memory = WorkingMemory(agent_id=config.name)
        self.memory_compressor = memory_compressor
        self.sandbox_manager = sandbox_manager or SandboxManager(use_docker=config.use_docker_sandbox)
        self.sandbox = None

    async def initialize_sandbox(self) -> None:
        """Initialize the agent's sandbox (with optional warm-start)."""
        self.sandbox = await self.sandbox_manager.create(
            from_snapshot=self.config.snapshot_to_restore
        )

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.sandbox:
            await self.sandbox_manager.destroy(self.sandbox.id)

    async def run(self, task: str) -> str:
        """Execute the agent on a task. Returns the final text response."""
        await self.initialize_sandbox()
        
        try:
            # Initialize conversation
            self._messages = [
                Message(role=Role.SYSTEM, content=self.config.system_prompt),
                Message(role=Role.USER, content=task),
            ]
            self.working_memory.add_event("SYSTEM", self.config.system_prompt)
            self.working_memory.add_event("USER", task)

            # Get tool definitions
            tool_defs = self._get_tool_defs()

            # Agent loop
            while self.step_count < self.config.max_steps:
                self.step_count += 1

                # If memory is getting large, compress it
                if self.memory_compressor:
                    compressed = await self.memory_compressor.step(self.working_memory)
                    if compressed:
                        # Rebuild context from working memory to avoid context window limits
                        self._rebuild_messages_from_memory()

                # Call LLM
                response = await self._llm.generate(
                    messages=self._messages,
                    tools=tool_defs if tool_defs else None,
                    config=self.config.llm,
                )
                self.working_memory.add_event("ASSISTANT", response.content or "Tool Calls Generated")

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
                        # Pass sandbox to tool if the tool supports it
                        # Note: In a full implementation, tools would be aware of the sandbox
                        result = await tool.execute(tc.arguments)
                        tool_output = result.output
                        is_error = result.is_error

                    self.working_memory.add_event("TOOL", f"{tc.name} -> {tool_output}")
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
        finally:
            await self.cleanup()

    def _rebuild_messages_from_memory(self) -> None:
        """Rebuilds the self._messages list based on the compressed working memory."""
        new_messages = []
        for event in self.working_memory._events:
            role_map = {"SYSTEM": Role.SYSTEM, "USER": Role.USER, "ASSISTANT": Role.ASSISTANT, "TOOL": Role.USER, "summary": Role.SYSTEM}
            role = role_map.get(event["type"], Role.USER)
            new_messages.append(Message(role=role, content=f"[{event['type']}] {event['content']}"))
        
        # Ensure system prompt is first
        if not new_messages or new_messages[0].role != Role.SYSTEM:
             new_messages.insert(0, Message(role=Role.SYSTEM, content=self.config.system_prompt))
             
        self._messages = new_messages

    def _get_tool_defs(self) -> list[ToolDef]:
        """Get tool definitions for the LLM."""
        if self.config.tools is not None:
            return self._tools.get_tool_defs(self.config.tools)
        return self._tools.get_tool_defs()
