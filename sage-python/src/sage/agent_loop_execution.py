"""Tool execution helpers for AgentLoop.

Standalone functions for tool execution, topology scheduling, and CEGAR
repair. These take dependencies as parameters instead of relying on
self.X attribute access.
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


def schedule_from_topology(current_topology: Any) -> list[dict]:
    """Use Rust TopologyExecutor to get node execution order.

    Returns list of node specs for topology-aware execution.
    Falls back to empty list if executor unavailable.
    """
    if not current_topology:
        return []
    try:
        from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
        executor = PyTopologyExecutor(current_topology)
        schedule: list[dict] = []
        while not executor.is_done():
            ready = executor.next_ready(current_topology)
            if not ready:
                break
            for idx in ready:
                node = current_topology.get_node(idx)
                schedule.append({
                    "index": idx,
                    "role": node.role,
                    "model_id": node.model_id,
                    "system": node.system,
                })
                executor.mark_completed(idx)
        return schedule
    except (ImportError, RuntimeError):
        return []


async def run_topology(
    task: str,
    current_topology: Any,
    llm_provider: Any,
    llm_config: Any,
    emit_fn: Any,
) -> str | None:
    """Execute multi-node topology via TopologyRunner.

    Returns the result string if topology has >1 node, or None to fall
    through to standard single-LLM execution.
    """
    from sage.agent_loop import LoopPhase

    if not current_topology:
        return None

    schedule = schedule_from_topology(current_topology)
    if len(schedule) <= 1:
        return None  # Single node or empty -- use standard path

    try:
        from sage.topology.runner import TopologyRunner
        from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
        executor = PyTopologyExecutor(current_topology)
        runner = TopologyRunner(
            graph=current_topology,
            executor=executor,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )
        result = await runner.run(task)
        emit_fn(
            LoopPhase.THINK,
            topology_execution="multi_agent",
            node_count=len(schedule),
        )
        return result
    except (ImportError, RuntimeError, TimeoutError) as e:
        log.warning("TopologyRunner failed (%s), falling back to single-LLM", e)
        return None


async def cegar_repair(
    content: str,
    prm_details: Any,
    invariant_feedback: list[str],
    system_prompt: str,
    llm_provider: Any,
    llm_config: Any,
    prm: Any,
) -> str | None:
    """Attempt CEGAR repair of failed S3 verification.

    Extracts failed clauses from PRM details and invariant feedback,
    builds a targeted repair prompt, and makes a single LLM call.
    Returns repaired content if PRM passes, or None if repair fails.
    """
    repair_prompt = (
        "SYSTEM: Your formal verification FAILED. "
        "Do NOT regenerate from scratch -- fix the specific failures below.\n\n"
        f"Verification error: {prm_details}\n"
    )
    if invariant_feedback:
        repair_prompt += (
            "\nFailed invariant clauses:\n"
            + "\n".join(f"- {f}" for f in invariant_feedback)
            + "\n"
        )
    repair_prompt += (
        "\nFix your reasoning by adding the missing formal assertions. "
        "Use <think> tags with Z3 assertions for each step."
    )

    messages = [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(role=Role.ASSISTANT, content=content),
        Message(role=Role.USER, content=repair_prompt),
    ]

    try:
        response = await llm_provider.generate(
            messages=messages,
            config=llm_config,
        )
        repaired = response.content or ""
        if not repaired:
            return None

        # Verify the repair
        r_path, details = prm.calculate_r_path(repaired)
        if r_path >= 0.0 or "error" not in details:
            log.info("CEGAR repair succeeded: r_path=%.2f", r_path)
            return repaired
        else:
            log.warning("CEGAR repair failed: %s", details)
            return None
    except (RuntimeError, TimeoutError) as e:
        log.warning("CEGAR repair LLM call failed: %s", e)
        return None
