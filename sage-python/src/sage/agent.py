"""Core agent runtime for YGN-SAGE."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry
from sage.memory.working import WorkingMemory
from sage.memory.compressor import MemoryCompressor
from sage.sandbox.manager import SandboxManager
from sage.topology.kg_rlvr import ProcessRewardModel


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
    enforce_system3: bool = True


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
        
        # SOTA/ASI Integration
        self.working_memory = WorkingMemory(agent_id=config.name)
        self.memory_compressor = memory_compressor
        self.sandbox_manager = sandbox_manager or SandboxManager(use_docker=config.use_docker_sandbox)
        self.sandbox = None
        self.prm = ProcessRewardModel()
        
        # OpenSage: Unified sub-agent pool
        self.agent_pool: dict[str, Agent] = {}
        
        # Benchmarking stats
        self.total_inference_time: float = 0.0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

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
        self.start_time = time.perf_counter()
        await self.initialize_sandbox()
        
        system_prompt = self.config.system_prompt
        if self.config.enforce_system3:
            system_prompt += "\n\nCRITICAL: You MUST use <think>...</think> tags to reason step-by-step before answering. Your reasoning is evaluated by a Process Reward Model."
        
        try:
            # Initialize conversation
            self._messages = [
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=task),
            ]
            self.working_memory.add_event("SYSTEM", system_prompt)
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
                        self._rebuild_messages_from_memory()

                # Call LLM with precision timing for AIO calculation
                inf_start = time.perf_counter()
                response = await self._llm.generate(
                    messages=self._messages,
                    tools=tool_defs if tool_defs else None,
                    config=self.config.llm,
                )
                self.total_inference_time += (time.perf_counter() - inf_start)
                
                content = response.content or "Tool Calls Generated"
                
                # SOTA 2026: KG-RLVR System 3 Reasoning Check
                if self.config.enforce_system3 and response.content:
                    r_path, details = self.prm.calculate_r_path(response.content)
                    if r_path < 0.0:
                        # Agent failed to use System 3 reasoning. Intercept and force correction.
                        warning = "SYSTEM: You failed to use <think> tags. You must break down the problem structurally before answering."
                        self._messages.append(Message(role=Role.USER, content=warning))
                        self.working_memory.add_event("SYSTEM_WARNING", warning)
                        continue
                    else:
                        self.working_memory.add_event("REWARD", f"R_path={r_path:.2f} Verifiable={details['verifiable_ratio']:.0%}")
                
                self.working_memory.add_event("ASSISTANT", content)

                # If no tool calls, we're done
                if not response.tool_calls:
                    self.result = response.content
                    self._messages.append(
                        Message(role=Role.ASSISTANT, content=response.content)
                    )
                    break

                # Add assistant message with tool calls
                self._messages.append(
                    Message(role=Role.ASSISTANT, content=response.content)
                )

                # Execute tool calls
                for tc in response.tool_calls:
                    tool = self._tools.get(tc.name)
                    if tool is None:
                        tool_output = f"Error: Unknown tool '{tc.name}'"
                    else:
                        # OpenSage injection: pass context to agent-mgmt tools
                        kwargs = tc.arguments.copy()
                        props = {}
                        if hasattr(tool, 'tool_def') and tool.tool_def:
                            props = tool.tool_def.parameters.get("properties", {})
                            
                        if "agent_pool" in props:
                            kwargs["agent_pool"] = self.agent_pool
                        if "parent_agent" in props:
                            kwargs["parent_agent"] = self
                        if "registry" in props:
                            kwargs["registry"] = self._tools
                            
                        result = await tool.execute(kwargs)
                        tool_output = result.output

                    self.working_memory.add_event("TOOL", f"{tc.name} -> {tool_output}")
                    self._messages.append(
                        Message(
                            role=Role.TOOL,
                            content=tool_output,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

            self.end_time = time.perf_counter()
            return self.result or f"Agent finished at step {self.step_count}"
        finally:
            await self.cleanup()

    def get_aio_stats(self) -> dict[str, float]:
        """Calculate Agentic Infrastructure Overhead (AIO)."""
        total_wall_time = self.end_time - self.start_time
        if total_wall_time <= 0:
            return {"aio_ratio": 0.0, "total_time": 0.0, "inf_time": 0.0}
        
        # AIO Ratio = (Total - Inference) / Total
        aio_ratio = (total_wall_time - self.total_inference_time) / total_wall_time
        return {
            "aio_ratio": max(0.0, aio_ratio),
            "total_wall_time": total_wall_time,
            "llm_inference_time": self.total_inference_time,
            "infrastructure_overhead_time": total_wall_time - self.total_inference_time
        }

    def _rebuild_messages_from_memory(self) -> None:
        """Rebuilds the self._messages list based on the compressed working memory."""
        new_messages = []
        for event in self.working_memory._events:
            role_map = {"SYSTEM": Role.SYSTEM, "USER": Role.USER, "ASSISTANT": Role.ASSISTANT, "TOOL": Role.USER, "summary": Role.SYSTEM}
            role = role_map.get(event["type"], Role.USER)
            new_messages.append(Message(role=role, content=f"[{event['type']}] {event['content']}"))
        
        if not new_messages or new_messages[0].role != Role.SYSTEM:
             new_messages.insert(0, Message(role=Role.SYSTEM, content=self.config.system_prompt))
             
        self._messages = new_messages

    def _get_tool_defs(self) -> list[ToolDef]:
        """Get tool definitions for the LLM."""
        if self.config.tools is not None:
            return self._tools.get_tool_defs(self.config.tools)
        return self._tools.get_tool_defs()
