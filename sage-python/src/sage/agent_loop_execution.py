"""Tool execution helpers for AgentLoop.

Extracted from agent_loop.py _run_legacy() to reduce file size.
These are standalone functions that take the tool registry and sandbox
as parameters instead of relying on self.X attribute access.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import Message, Role

log = logging.getLogger(__name__)


async def execute_tool_call(
    tc: Any,
    tool_registry: Any,
    emit_fn: Any,
) -> str:
    """Execute a single tool call with argument validation.

    Parameters
    ----------
    tc : ToolCall
        The tool call object (has .name, .arguments, .id).
    tool_registry : ToolRegistry
        Registry to look up the tool by name.
    emit_fn : callable
        Callable to emit LoopPhase.ACT events (for TOOL_GAP detection).

    Returns
    -------
    str
        The tool output string, or an error message.
    """
    from sage.agent_loop import LoopPhase

    tool = tool_registry.get(tc.name)
    if tool is None:
        # Emit TOOL_GAP for ToolForge gap detection
        emit_fn(
            LoopPhase.ACT,
            tool_gap=True,
            tool_name=tc.name,
            tool_args=tc.arguments,
        )
        return f"Error: Unknown tool '{tc.name}'"
    kwargs = tc.arguments
    if not isinstance(kwargs, dict):
        log.warning("Tool '%s' received non-dict arguments: %s", tc.name, type(kwargs))
        return (
            f"Error: Tool '{tc.name}' received invalid arguments "
            f"(expected dict, got {type(kwargs).__name__})"
        )
    try:
        result = await tool.execute(kwargs.copy())
        return result.output
    except (RuntimeError, ValueError, TimeoutError) as e:
        log.error("Tool '%s' execution failed: %s", tc.name, e)
        return f"Error executing tool '{tc.name}': {type(e).__name__}: {e}"


async def execute_tool_calls_and_record(
    response: Any,
    content: str,
    messages: list[Message],
    tool_registry: Any,
    working_memory: Any,
    causal_memory: Any,
    cb_causal: Any,
    skip_memory: bool,
    step_count: int,
    emit_fn: Any,
    max_messages: int,
) -> None:
    """Execute all tool calls from a response, record in memory, trim messages.

    This handles:
    - Appending the assistant message
    - Executing each tool call and appending TOOL messages
    - Recording causal edges (tool -> output) in causal memory
    - Trimming messages to max_messages

    Mutates *messages* in place.
    """
    from sage.agent_loop import LoopPhase

    messages.append(Message(role=Role.ASSISTANT, content=content))

    for tc in response.tool_calls:
        emit_fn(LoopPhase.ACT, tool=tc.name, args=tc.arguments)
        output = await execute_tool_call(tc, tool_registry, emit_fn)
        working_memory.add_event("TOOL", f"{tc.name} -> {output}")
        messages.append(Message(
            role=Role.TOOL, content=output,
            tool_call_id=tc.id, name=tc.name,
        ))
        # Causal edge: tool invocation "triggered" its output entity
        if causal_memory and not cb_causal.should_skip() and not skip_memory:
            try:
                tool_entity = f"tool:{tc.name}"
                output_entity = f"result:{tc.name}:{step_count}"
                causal_memory.add_entity(tool_entity)
                causal_memory.add_entity(output_entity)
                causal_memory.add_causal_edge(tool_entity, output_entity, cause_type="triggered")
                cb_causal.record_success()
            except (RuntimeError, AttributeError) as exc:
                cb_causal.record_failure(exc)

    if len(messages) > max_messages:
        messages[:] = messages[:2] + messages[-(max_messages - 2):]


async def store_episodic_and_entities(
    task: str,
    content: str,
    step_count: int,
    episodic_memory: Any,
    memory_agent: Any,
    semantic_memory: Any,
    causal_memory: Any,
    cb_episodic: Any,
    cb_entity: Any,
    cb_causal: Any,
    skip_memory: bool,
) -> None:
    """Store significant responses in episodic memory and extract entities.

    This handles:
    - Episodic memory storage for long responses
    - Entity extraction via memory_agent
    - Causal edge creation between consecutive extracted entities

    Called after working_memory.add_event("ASSISTANT", content) in the legacy loop.
    """
    # Episodic memory
    if episodic_memory and len(content) > 100 and not cb_episodic.should_skip() and not skip_memory:
        try:
            await episodic_memory.store(
                key=f"step-{step_count}",
                content=content[:500],
                metadata={"task": task, "step": step_count},
            )
            cb_episodic.record_success()
        except (RuntimeError, AttributeError) as e:
            cb_episodic.record_failure(e)

    # Semantic memory: extract entities from response
    if memory_agent and semantic_memory and content and len(content) > 50 and not cb_entity.should_skip() and not skip_memory:
        try:
            extraction = await memory_agent.extract(content[:1000])
            if extraction.entities:
                semantic_memory.add_extraction(extraction)
                # Feed causal memory: consecutive entities form causal edges
                # AMA-Bench (2602.22769): memory without causality fails
                if causal_memory and len(extraction.entities) >= 2 and not cb_causal.should_skip():
                    try:
                        for i in range(len(extraction.entities) - 1):
                            src, tgt = extraction.entities[i], extraction.entities[i + 1]
                            causal_memory.add_entity(src)
                            causal_memory.add_entity(tgt)
                            causal_memory.add_causal_edge(src, tgt, cause_type="enabled")
                        cb_causal.record_success()
                    except (RuntimeError, AttributeError) as exc:
                        cb_causal.record_failure(exc)
            cb_entity.record_success()
        except (RuntimeError, AttributeError) as e:
            cb_entity.record_failure(e)
