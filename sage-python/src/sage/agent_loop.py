"""Structured agent runtime: perceive -> think -> act -> learn."""
from __future__ import annotations

import math
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

# Approximate cost per 1K tokens (USD) for dashboard estimation
_COST_PER_1K = {
    "gpt-5.3-codex": 0.03,
    "gpt-5.2": 0.06,
    "gemini-3.1-pro-preview": 0.007,
    "gemini-3-flash-preview": 0.0015,
    "gemini-3.1-flash-lite-preview": 0.0005,
    "gemini-2.5-flash-lite": 0.0003,
    "gemini-2.5-flash": 0.001,
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


def _text_entropy(text: str) -> float:
    """Shannon entropy of character distribution (normalised 0-1)."""
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    ent = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)
    max_ent = math.log2(max(len(freq), 2))
    return ent / max_ent if max_ent > 0 else 0.0


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
        # Injected by boot.py
        self.metacognition: Any = None
        self.topology_population: Any = None

        # Stats
        self.step_count = 0
        self.total_inference_time = 0.0
        self.total_cost_usd = 0.0
        self.start_time = 0.0
        self._prm_retries = 0
        self._max_prm_retries = 2

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
        self.total_cost_usd = 0.0
        self._prm_retries = 0
        self.step_count = 0

        # === PERCEIVE: Gather context ===
        perceive_meta: dict[str, Any] = {"task": task, "agent": self.config.name}

        # Metacognitive routing (if wired)
        if self.metacognition:
            profile = self.metacognition.assess_complexity(task)
            decision = self.metacognition.route(profile)
            perceive_meta["system"] = decision.system
            perceive_meta["routed_tier"] = decision.llm_tier
            perceive_meta["use_z3"] = decision.use_z3
            perceive_meta["validation_level"] = decision.validation_level
            perceive_meta["complexity"] = round(profile.complexity, 2)
            perceive_meta["uncertainty"] = round(profile.uncertainty, 2)
            if profile.reasoning:
                perceive_meta["routing_reason"] = profile.reasoning

        self._emit(LoopPhase.PERCEIVE, **perceive_meta)

        system_prompt = self.config.system_prompt
        if self.config.validation_level >= 3:
            system_prompt += (
                "\n\nCRITICAL: Use <think>...</think> tags to reason step-by-step. "
                "Your reasoning is evaluated by a Process Reward Model."
            )
        elif self.config.validation_level >= 2:
            system_prompt += (
                "\n\nUse step-by-step reasoning to solve this task. "
                "Show your work clearly."
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
            model_name = self.config.llm.model
            self._emit(LoopPhase.THINK, model=model_name, step=self.step_count)

            t0 = time.perf_counter()
            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )
            inference_ms = (time.perf_counter() - t0) * 1000
            self.total_inference_time += inference_ms / 1000

            content = response.content or ""

            # Cost estimation
            tokens = _estimate_tokens(content)
            cost_per_k = _COST_PER_1K.get(model_name, 0.001)
            step_cost = (tokens / 1000) * cost_per_k
            self.total_cost_usd += step_cost

            # Entropy for CGRS self-braking
            entropy = _text_entropy(content)
            brake = False
            if self.metacognition:
                self.metacognition.record_output_entropy(entropy)
                brake = self.metacognition.should_brake()

            self._emit(
                LoopPhase.THINK,
                latency_ms=round(inference_ms, 1),
                cost_usd=round(self.total_cost_usd, 4),
                entropy=round(entropy, 3),
                brake=brake,
            )

            # System 3 validation (Z3 PRM) -- max 2 retries then accept
            if self.config.validation_level >= 3 and content:
                r_path, details = self.prm.calculate_r_path(content)
                self._emit(LoopPhase.THINK, r_path=r_path, details=details)
                if r_path < 0.0 and "error" in details:
                    self._prm_retries += 1
                    if self._prm_retries <= self._max_prm_retries:
                        messages.append(Message(
                            role=Role.USER,
                            content="SYSTEM: Use <think> tags for structured reasoning.",
                        ))
                        continue
                    # Max retries reached -- accept response as-is
                    log.warning("PRM retry limit reached, accepting response without <think> tags.")
                else:
                    self._prm_retries = 0  # Reset on success

            # CGRS: stop if converged
            if brake:
                log.info("CGRS self-brake triggered — stopping reasoning loop.")
                result_text = content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

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
            wall = time.perf_counter() - self.start_time
            learn_meta: dict[str, Any] = {
                "aio_ratio": aio,
                "events": self.working_memory.event_count(),
                "wall_time_s": round(wall, 1),
                "cost_usd": round(self.total_cost_usd, 4),
            }

            # Evolution grid snapshot (if wired)
            if self.topology_population and self.topology_population.size() > 0:
                try:
                    cells = []
                    best_fitness = 0.0
                    for (x, y), (genome, score) in self.topology_population._grid.items():
                        cells.append({"x": x, "y": y, "fitness": round(score, 2)})
                        best_fitness = max(best_fitness, score)
                    learn_meta["evo_cells"] = cells
                    learn_meta["evo_best"] = round(best_fitness, 2)
                    learn_meta["evo_grid_size"] = len(cells)
                except Exception:
                    pass

            self._emit(LoopPhase.LEARN, **learn_meta)

        # Final completion event
        self._emit(LoopPhase.LEARN,
                   result="complete", steps=self.step_count,
                   cost_usd=round(self.total_cost_usd, 4))
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
