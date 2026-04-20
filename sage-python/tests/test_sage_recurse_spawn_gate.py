"""Integration tests for the sage_recurse spawn-budget gate (Task D of
the 2026-04-20 post-Rust-First phase-1 stab plan)."""
from __future__ import annotations

import asyncio
from typing import Any
import pytest

from sage.topology_controller import TopologyController
from sage.tools.sage_recurse import build_sage_recurse_tool, sage_recurse_origin_node


async def _fake_run(sub_task: str, **kwargs: Any) -> str:
    return f"done: {sub_task}"


def _make_controller() -> TopologyController:
    return TopologyController()


@pytest.mark.asyncio
async def test_sage_recurse_no_gate_without_controller():
    """Backwards-compat: tool without controller keeps old behavior."""
    tool = build_sage_recurse_tool(_fake_run)  # no controller arg
    result = await tool.run({"sub_task": "compute"})
    assert result == "done: compute"


@pytest.mark.asyncio
async def test_sage_recurse_gate_allows_first_spawn():
    """First spawn passes gate and increments Rust counter."""
    ctrl = _make_controller()
    tool = build_sage_recurse_tool(_fake_run, controller=ctrl)
    token = sage_recurse_origin_node.set(7)
    try:
        result = await tool.run({"sub_task": "compute"})
        assert result == "done: compute"
    finally:
        sage_recurse_origin_node.reset(token)
    assert ctrl._rust_ctrl.spawn_count == 1


@pytest.mark.asyncio
async def test_sage_recurse_gate_refuses_over_budget():
    """After MAX_SPAWNS spawns, tool refuses."""
    ctrl = _make_controller()
    ctrl._seed_for_tests(spawn=3)  # MAX_SPAWNS=3, at cap
    tool = build_sage_recurse_tool(_fake_run, controller=ctrl)
    token = sage_recurse_origin_node.set(0)
    try:
        result = await tool.run({"sub_task": "another"})
    finally:
        sage_recurse_origin_node.reset(token)
    assert "spawn budget exhausted" in result
    assert ctrl._rust_ctrl.spawn_count == 3  # unchanged


@pytest.mark.asyncio
async def test_sage_recurse_records_budget_before_dispatch():
    """Failed dispatch still counts toward budget (DoS guard)."""
    async def _failing_run(sub_task: str, **_kw: Any) -> str:
        raise RuntimeError("dispatch exploded")

    ctrl = _make_controller()
    tool = build_sage_recurse_tool(_failing_run, controller=ctrl)
    token = sage_recurse_origin_node.set(0)
    try:
        result = await tool.run({"sub_task": "will fail"})
    finally:
        sage_recurse_origin_node.reset(token)
    assert "dispatch failed" in result or "dispatch exploded" in result
    assert ctrl._rust_ctrl.spawn_count == 1  # debited despite failure


@pytest.mark.asyncio
async def test_sage_recurse_no_origin_node_skips_gate():
    """Without origin_node ContextVar, gate no-ops (standalone callers)."""
    ctrl = _make_controller()
    ctrl._seed_for_tests(spawn=3)  # at cap
    tool = build_sage_recurse_tool(_fake_run, controller=ctrl)
    # Do NOT set sage_recurse_origin_node.
    result = await tool.run({"sub_task": "standalone"})
    assert result == "done: standalone"
    assert ctrl._rust_ctrl.spawn_count == 3  # unchanged — gate skipped


@pytest.mark.asyncio
async def test_sage_recurse_fallback_dispatch_failure_returns_string():
    """TypeError-fallback path must also be guarded (I1 review fix).

    Scenario: callable rejects the system_hint kwarg (TypeError), fallback
    retry without kwargs raises a different exception. The fallback must
    be guarded — otherwise the exception escapes the tool, violating the
    "never raise" contract.
    """
    first_call = True

    async def _flaky_run(sub_task: str, **kwargs: Any) -> str:
        nonlocal first_call
        if first_call:
            first_call = False
            raise TypeError("unexpected keyword argument 'system_hint'")
        raise RuntimeError("fallback also exploded")

    ctrl = _make_controller()
    tool = build_sage_recurse_tool(_flaky_run, controller=ctrl)
    token = sage_recurse_origin_node.set(0)
    try:
        result = await tool.run({"sub_task": "trigger fallback", "system_hint": 1})
    finally:
        sage_recurse_origin_node.reset(token)
    assert "dispatch failed" in result
    assert "fallback also exploded" in result or "RuntimeError" in result
    assert ctrl._rust_ctrl.spawn_count == 1  # debited before dispatch


@pytest.mark.asyncio
async def test_sage_recurse_origin_node_isolated_across_tasks():
    """ContextVar must not leak across concurrent asyncio tasks."""
    seen: dict[int, int | None] = {}

    async def _capture(node_idx: int) -> None:
        token = sage_recurse_origin_node.set(node_idx)
        try:
            await asyncio.sleep(0.001)  # yield control
            seen[node_idx] = sage_recurse_origin_node.get()
        finally:
            sage_recurse_origin_node.reset(token)

    await asyncio.gather(_capture(1), _capture(2), _capture(3))
    assert seen == {1: 1, 2: 2, 3: 3}, f"leaked: {seen}"
