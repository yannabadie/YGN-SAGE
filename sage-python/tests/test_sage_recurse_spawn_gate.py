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
