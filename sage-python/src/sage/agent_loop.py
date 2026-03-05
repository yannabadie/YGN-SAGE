"""Structured agent runtime: perceive -> think -> act -> learn."""
from __future__ import annotations

import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from sage.agent import AgentConfig, Agent
from sage.llm.base import LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.registry import ToolRegistry
from sage.memory.working import WorkingMemory
from sage.memory.compressor import MemoryCompressor
from sage.sandbox.manager import SandboxManager
from sage.topology.kg_rlvr import ProcessRewardModel

log = logging.getLogger(__name__)

STREAM_FILE = Path("docs/plans/agent_stream.jsonl")


class LoopPhase(str, Enum):
    PERCEIVE = "perceive"
    THINK = "think"
    ACT = "act"
    LEARN = "learn"


@dataclass
class LoopEvent:
    phase: LoopPhase
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    step: int = 0


class AgentLoop:
    """Structured agent loop with event emission for dashboard."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        memory_compressor: MemoryCompressor | None = None,
        on_event: Callable[[LoopEvent], None] | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._on_event = on_event or self._default_event_handler
        self.working_memory = WorkingMemory(agent_id=config.name)
        self.memory_compressor = memory_compressor
        self.prm = ProcessRewardModel()
        self.agent_pool: dict[str, Any] = {}

        # Stats
        self.step_count = 0
        self.total_inference_time = 0.0
        self.start_time = 0.0

    def _emit(self, phase: LoopPhase, **data: Any) -> None:
        evt = LoopEvent(phase=phase, data=data, step=self.step_count)
        self._on_event(evt)

    def _default_event_handler(self, event: LoopEvent) -> None:
        log.info(f"[{event.phase.value}] step={event.step} {event.data}")
        # Append to stream file for dashboard
        try:
            STREAM_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STREAM_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": event.phase.value.upper(),
                    "step": event.step,
                    "timestamp": event.timestamp,
                    "meta": event.data,
                }) + "\n")
        except Exception:
            pass

    async def run(self, task: str) -> str:
        """Execute the full perceive -> think -> act -> learn cycle."""
        self.start_time = time.perf_counter()

        # === PERCEIVE: Gather context ===
        self._emit(LoopPhase.PERCEIVE, task=task, agent=self.config.name)

        system_prompt = self.config.system_prompt
        if self.config.enforce_system3:
            system_prompt += (
                "\n\nCRITICAL: Use <think>...</think> tags to reason step-by-step. "
                "Your reasoning is evaluated by a Process Reward Model."
            )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ]
        self.working_memory.add_event("USER", task)
        tool_defs = self._tools.get_tool_defs(
            self.config.tools if self.config.tools else None
        )

        result_text = ""

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Memory compression if needed
            if self.memory_compressor:
                compressed = await self.memory_compressor.step(self.working_memory)
                if compressed:
                    messages = self._rebuild_messages(system_prompt)

            # === THINK: Call LLM ===
            self._emit(LoopPhase.THINK, model=self.config.llm.model, step=self.step_count)

            t0 = time.perf_counter()
            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )
            self.total_inference_time += (time.perf_counter() - t0)

            content = response.content or ""

            # System 3 validation
            if self.config.enforce_system3 and content:
                r_path, details = self.prm.calculate_r_path(content)
                self._emit(LoopPhase.THINK, r_path=r_path, details=details)
                if r_path < 0.0 and "error" in details:
                    messages.append(Message(
                        role=Role.USER,
                        content="SYSTEM: Use <think> tags for structured reasoning.",
                    ))
                    continue

            self.working_memory.add_event("ASSISTANT", content)

            # No tool calls -> final answer
            if not response.tool_calls:
                result_text = content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            # === ACT: Execute tools ===
            messages.append(Message(role=Role.ASSISTANT, content=content))

            for tc in response.tool_calls:
                self._emit(LoopPhase.ACT, tool=tc.name, args=tc.arguments)

                tool = self._tools.get(tc.name)
                if tool is None:
                    output = f"Error: Unknown tool '{tc.name}'"
                else:
                    kwargs = tc.arguments.copy()
                    result = await tool.execute(kwargs)
                    output = result.output

                self.working_memory.add_event("TOOL", f"{tc.name} -> {output}")
                messages.append(Message(
                    role=Role.TOOL, content=output,
                    tool_call_id=tc.id, name=tc.name,
                ))

            # === LEARN: Update memory and stats ===
            aio = self._compute_aio()
            self._emit(LoopPhase.LEARN, aio_ratio=aio, events=self.working_memory.event_count())

        self._emit(LoopPhase.LEARN, result="complete", steps=self.step_count)
        return result_text or f"Agent finished at step {self.step_count}"

    def _compute_aio(self) -> float:
        wall = time.perf_counter() - self.start_time
        if wall <= 0:
            return 0.0
        return max(0.0, (wall - self.total_inference_time) / wall)

    def _rebuild_messages(self, system_prompt: str) -> list[Message]:
        msgs = [Message(role=Role.SYSTEM, content=system_prompt)]
        for event in self.working_memory._events:
            role_map = {
                "SYSTEM": Role.SYSTEM, "USER": Role.USER,
                "ASSISTANT": Role.ASSISTANT, "TOOL": Role.USER,
                "summary": Role.SYSTEM,
            }
            role = role_map.get(event["type"], Role.USER)
            msgs.append(Message(role=role, content=event["content"]))
        return msgs
