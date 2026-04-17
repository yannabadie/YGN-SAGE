"""Tests for the sage_recurse recursive self-invocation tool.

Sprint 4: The Conductor-style recursion. Key properties:
- Happy path returns the sub-run output.
- Depth cap refuses deeper-than-N calls and tells the caller why.
- Depth counter resets after each outer call.
- Empty sub_task is rejected.
- Callers that pass a system_hint see it forwarded.
- A run_callable that doesn't accept system_hint still works (graceful
  fallback) so old benches keep running.
"""
from __future__ import annotations

import asyncio

import pytest

from sage.tools.sage_recurse import (
    MAX_RECURSION_DEPTH,
    build_sage_recurse_tool,
    current_depth,
    _RECURSION_DEPTH,
)


@pytest.mark.asyncio
async def test_happy_path_returns_run_output():
    async def _run(task, *, system_hint=None):
        return f"answered:{task}|hint={system_hint}"

    tool = build_sage_recurse_tool(_run)
    result = await tool.run({"sub_task": "add 2+3", "system_hint": 2})
    assert "answered:add 2+3" in result
    assert "hint=2" in result


@pytest.mark.asyncio
async def test_empty_sub_task_rejected():
    async def _run(task, *, system_hint=None):
        return "never"

    tool = build_sage_recurse_tool(_run)
    result = await tool.run({"sub_task": "   "})
    assert "non-empty sub_task" in result


@pytest.mark.asyncio
async def test_depth_cap_refuses_deep_calls():
    """When the outer context already has depth >= MAX, refuse."""
    async def _run(task, *, system_hint=None):
        return "should never run"

    tool = build_sage_recurse_tool(_run)
    token = _RECURSION_DEPTH.set(MAX_RECURSION_DEPTH)
    try:
        result = await tool.run({"sub_task": "x"})
        assert "max recursion depth" in result
    finally:
        _RECURSION_DEPTH.reset(token)


@pytest.mark.asyncio
async def test_depth_increments_and_resets_around_call():
    observed = {}

    async def _run(task, *, system_hint=None):
        observed["inside"] = current_depth()
        return "ok"

    tool = build_sage_recurse_tool(_run)
    assert current_depth() == 0
    await tool.run({"sub_task": "hi"})
    assert observed["inside"] == 1, "depth should increment during inner call"
    assert current_depth() == 0, "depth must reset after the call returns"


@pytest.mark.asyncio
async def test_nested_calls_respect_cap():
    """Recursively calling sage_recurse from within sage_recurse must cap."""
    runs = []
    tool_holder: dict = {}

    async def _run(task, *, system_hint=None):
        runs.append((task, current_depth()))
        if current_depth() < MAX_RECURSION_DEPTH:
            # Inner call via the same tool.
            return await tool_holder["tool"].run({"sub_task": task + "+"})
        return f"bottom@depth{current_depth()}"

    tool_holder["tool"] = build_sage_recurse_tool(_run)
    result = await tool_holder["tool"].run({"sub_task": "start"})

    # We should observe exactly MAX_RECURSION_DEPTH nested runs before cap.
    depths_seen = [d for _, d in runs]
    assert depths_seen == list(range(1, MAX_RECURSION_DEPTH + 1))
    # The deepest inner call still returns a value (cap is enforced BEFORE
    # the next recursion, not from inside the deepest _run).
    assert "bottom" in result or "max recursion depth" in result


@pytest.mark.asyncio
async def test_fallback_when_runnable_has_no_system_hint():
    """Old callables without a system_hint kwarg must still work."""
    async def _plain_run(task):  # no system_hint kwarg
        return f"plain:{task}"

    tool = build_sage_recurse_tool(_plain_run)
    # Caller provides a hint; the wrapper should catch the TypeError and
    # retry without it.
    result = await tool.run({"sub_task": "demo", "system_hint": 3})
    assert result == "plain:demo"


@pytest.mark.asyncio
async def test_invalid_system_hint_rejected():
    async def _run(task, *, system_hint=None):
        return "never"

    tool = build_sage_recurse_tool(_run)
    result = await tool.run({"sub_task": "x", "system_hint": 99})
    assert "system_hint must be 1, 2, or 3" in result


@pytest.mark.asyncio
async def test_tool_spec_has_expected_shape():
    async def _run(task, *, system_hint=None):
        return ""

    tool = build_sage_recurse_tool(_run)
    assert tool.spec.name == "sage_recurse"
    assert "recursion depth" in tool.spec.description.lower() or \
           "recursion" in tool.spec.description.lower()
    params = tool.spec.parameters
    assert "sub_task" in params["properties"]
    assert "sub_task" in params["required"]


@pytest.mark.asyncio
async def test_depth_is_asyncio_safe():
    """Two concurrent outer calls must not share depth counters."""
    async def _run(task, *, system_hint=None):
        await asyncio.sleep(0)
        return f"d={current_depth()}"

    tool = build_sage_recurse_tool(_run)
    results = await asyncio.gather(
        tool.run({"sub_task": "a"}),
        tool.run({"sub_task": "b"}),
    )
    # Each concurrent call sees depth=1 (its own frame), not 2.
    assert all(r == "d=1" for r in results)
    assert current_depth() == 0
