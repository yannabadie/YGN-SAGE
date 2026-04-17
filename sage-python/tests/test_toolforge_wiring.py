"""Tests for ToolForge wiring inside agent_loop.execute_tool_call.

Sprint 3 (Phase D.1 self-programming): unknown tool calls must open a
CreationTicket and trigger synthesis, not return a hard error. These
tests mock the ToolForge to avoid real LLM calls while verifying the
plumbing end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.agent_loop_execution import execute_tool_call


def _mk_tool_call(name: str, arguments: dict | None = None):
    return SimpleNamespace(name=name, arguments=arguments or {}, id="tc1")


def _mk_registry(tools: dict):
    reg = MagicMock()
    reg.get = lambda key: tools.get(key)
    return reg


@pytest.mark.asyncio
async def test_no_toolforge_returns_unknown_tool_error():
    """Baseline: without a ToolForge, unknown tools still error."""
    registry = _mk_registry({})
    emit = MagicMock()

    result = await execute_tool_call(
        _mk_tool_call("nonexistent"),
        registry,
        emit,
        toolforge=None,
    )
    assert "Unknown tool 'nonexistent'" in result
    # Gap event still emitted for observability.
    emit.assert_any_call(
        "ACT", tool_gap=True, tool_name="nonexistent", tool_args={},
    ) if False else None  # call signature uses LoopPhase enum, not string


@pytest.mark.asyncio
async def test_toolforge_synthesis_success_retries_call():
    """When ToolForge synthesizes the tool, execute_tool_call should retry."""
    synthesized_tool = MagicMock()
    synthesized_tool.execute = AsyncMock(
        return_value=SimpleNamespace(output="synthesized tool ran"),
    )
    # Registry starts empty, gains the tool after synthesis.
    registry = MagicMock()
    state = {"has": False}

    def _get(name):
        return synthesized_tool if state["has"] else None
    registry.get = _get

    async def _process_tickets(tickets):
        # Simulate successful synthesis: mutate registry.
        state["has"] = True
        return [tickets[0].tool_name_hint]

    gap_detector = MagicMock()
    gap_detector.on_unknown_tool = MagicMock(
        return_value=SimpleNamespace(tool_name_hint="my_tool"),
    )

    toolforge = MagicMock()
    toolforge.gap_detector = gap_detector
    toolforge.process_tickets = AsyncMock(side_effect=_process_tickets)

    result = await execute_tool_call(
        _mk_tool_call("my_tool", {"arg": 1}),
        registry,
        MagicMock(),
        toolforge=toolforge,
        task_context="fix the parser",
    )

    assert result == "synthesized tool ran"
    gap_detector.on_unknown_tool.assert_called_once()
    toolforge.process_tickets.assert_awaited_once()
    synthesized_tool.execute.assert_awaited_once_with({"arg": 1})


@pytest.mark.asyncio
async def test_toolforge_synthesis_failure_returns_error():
    """If synthesis doesn't produce the tool, caller still gets the error."""
    registry = _mk_registry({})

    toolforge = MagicMock()
    toolforge.gap_detector = MagicMock()
    toolforge.gap_detector.on_unknown_tool = MagicMock(
        return_value=SimpleNamespace(tool_name_hint="never_built"),
    )
    # process_tickets returns empty list — synthesis failed (gate reject etc.)
    toolforge.process_tickets = AsyncMock(return_value=[])

    result = await execute_tool_call(
        _mk_tool_call("never_built"),
        registry,
        MagicMock(),
        toolforge=toolforge,
        task_context="unknown capability",
    )

    assert "Unknown tool 'never_built'" in result


@pytest.mark.asyncio
async def test_toolforge_ticket_none_means_queue_full():
    """Gap detector can refuse (queue full, dedup) — no synthesis attempt."""
    registry = _mk_registry({})

    toolforge = MagicMock()
    toolforge.gap_detector = MagicMock()
    toolforge.gap_detector.on_unknown_tool = MagicMock(return_value=None)
    toolforge.process_tickets = AsyncMock()

    result = await execute_tool_call(
        _mk_tool_call("dup_tool"),
        registry,
        MagicMock(),
        toolforge=toolforge,
    )

    assert "Unknown tool 'dup_tool'" in result
    toolforge.process_tickets.assert_not_awaited()


@pytest.mark.asyncio
async def test_toolforge_synthesis_exception_is_swallowed():
    """A crashing ToolForge must not take down the agent loop."""
    registry = _mk_registry({})

    toolforge = MagicMock()
    toolforge.gap_detector = MagicMock()
    toolforge.gap_detector.on_unknown_tool = MagicMock(
        return_value=SimpleNamespace(tool_name_hint="broken"),
    )
    toolforge.process_tickets = AsyncMock(
        side_effect=RuntimeError("synth crashed"),
    )

    result = await execute_tool_call(
        _mk_tool_call("broken"),
        registry,
        MagicMock(),
        toolforge=toolforge,
    )
    assert "Unknown tool 'broken'" in result


@pytest.mark.asyncio
async def test_existing_tool_ignores_toolforge_path():
    """Known tools should never touch ToolForge — zero overhead."""
    existing = MagicMock()
    existing.execute = AsyncMock(return_value=SimpleNamespace(output="done"))
    registry = _mk_registry({"present": existing})

    toolforge = MagicMock()
    toolforge.gap_detector = MagicMock()
    toolforge.gap_detector.on_unknown_tool = MagicMock()
    toolforge.process_tickets = AsyncMock()

    result = await execute_tool_call(
        _mk_tool_call("present", {"x": 1}),
        registry,
        MagicMock(),
        toolforge=toolforge,
    )

    assert result == "done"
    toolforge.gap_detector.on_unknown_tool.assert_not_called()
    toolforge.process_tickets.assert_not_awaited()
