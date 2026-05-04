"""Regression test for A0a (2026-04-23, ALIRE2 §4 "shared mutable state").

The single-agent bypass path in ``Pipeline._stage_execute`` mutates
~10 fields on the shared ``self._agent_loop`` object. Prior to commit
<A0a>, the ``finally`` block restored only 3 of those 10, leaving
``write_gate``, ``gate_*``, ``_on_drift``, ``config.validation_level``,
``config.max_steps``, ``config.stall_after_tool_steps``, and
``_current_topology`` dirty for the next caller — state bleed under
nested / concurrent runs. This test locks the restoration contract.

Full concurrency safety (stop mutating a shared object entirely) is a
separate refactor (B9); this is the targeted fix.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# Sentinels chosen so any accidental truthiness / equality coercion
# will make the assertions fail rather than silently pass.
_SENTINEL_TOPOLOGY = object()
_SENTINEL_WRITE_GATE = object()
_SENTINEL_TASK = object()
_SENTINEL_TIER = "prior-tier-sentinel"
_SENTINEL_DRIFT = lambda *a, **kw: None  # noqa: E731
_SENTINEL_SKIP_ROUTING = False  # prior state


def _make_bypass_state_snapshot(loop) -> dict[str, Any]:
    """Mirror the snapshot shape ``pipeline.py`` takes at bypass entry."""
    return {
        "_skip_routing": getattr(loop, "_skip_routing", False),
        "_current_topology": loop._current_topology,
        "write_gate": getattr(loop, "write_gate", None),
        "gate_current_task": getattr(loop, "gate_current_task", None),
        "gate_source_tier": getattr(loop, "gate_source_tier", None),
        "_on_drift": getattr(loop, "_on_drift", None),
        "validation_level": loop.config.validation_level,
        "max_steps": loop.config.max_steps,
        "stall_after_tool_steps": loop.config.stall_after_tool_steps,
        "_llm": loop._llm,
        "llm_config": loop.config.llm,
    }


def _make_loop_with_prior_state() -> MagicMock:
    """Construct a mock AgentLoop with distinctive prior state values.

    Every field the bypass path mutates starts at a sentinel we can
    recognise post-run. If any field ends with a value different from
    its sentinel, the bypass-path restoration is leaking.
    """
    config = SimpleNamespace(
        validation_level=99,  # prior tier — pipeline sets 1/2/3 by ctx.system
        max_steps=123,  # prior value — pipeline sets 5/10/20 by ctx.system
        stall_after_tool_steps=77,
        llm=SimpleNamespace(model="prior-model"),
    )
    sandbox_manager = MagicMock()
    loop = MagicMock(
        config=config,
        sandbox_manager=sandbox_manager,
        _skip_routing=_SENTINEL_SKIP_ROUTING,
        _current_topology=_SENTINEL_TOPOLOGY,
        write_gate=_SENTINEL_WRITE_GATE,
        gate_current_task=_SENTINEL_TASK,
        gate_source_tier=_SENTINEL_TIER,
        _on_drift=_SENTINEL_DRIFT,
        _llm=SimpleNamespace(name="prior-provider"),
        total_cost_usd=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
    )
    loop.run = AsyncMock(return_value="bypass-output")
    return loop


@pytest.mark.asyncio
async def test_bypass_path_restores_all_mutated_fields():
    """After the bypass path's ``finally``, the agent_loop must look
    identical to its pre-bypass snapshot.

    We exercise the bypass path via a minimally-stubbed Pipeline that
    bypasses topology selection + verification + most boot wiring.
    The only thing under test is the mutation/restoration window
    inside ``_stage_execute``.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    # Stub a Pipeline whose only real code paths are __init__ and the
    # single-agent bypass block. Everything else short-circuits.
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = _make_loop_with_prior_state()
    # P6-B (cycle-11) instance state — `Pipeline.__new__` bypasses
    # `__init__`, so the lazy lock attrs never get set. Inject them so
    # `_get_agent_loop_bypass_lock` can find the slots and lazily build.
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.bandit = None  # skip bandit arm-selection in _stage_execute
    pipeline.write_gate = _SENTINEL_WRITE_GATE  # matches existing field
    pipeline.provider_pool = None
    pipeline.llm_provider = None
    pipeline._last_routing_decision = None
    pipeline._emit = MagicMock()
    pipeline.event_bus = None

    before = _make_bypass_state_snapshot(pipeline._agent_loop)

    ctx = SimpleNamespace(
        task="any task",
        topology=None,  # triggers single-agent bypass
        system=2,  # picks validation_level=2, max_steps=10, stall=9
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=True,  # skip the EXECUTE_UNVERIFIED warning
        bandit_decision_id=None,
    )
    await pipeline._stage_execute(ctx)

    after = _make_bypass_state_snapshot(pipeline._agent_loop)

    # EVERY field must equal its pre-bypass value.
    for k, pre_val in before.items():
        post_val = after[k]
        assert post_val == pre_val or post_val is pre_val, (
            f"Field {k!r} leaked: before={pre_val!r}, after={post_val!r}"
        )


@pytest.mark.asyncio
async def test_bypass_path_actually_mutates_during_run():
    """The restoration contract is only meaningful if the mutations
    happen in the first place — otherwise A0a's expanded ``finally``
    is guarding against nothing.

    We capture the agent_loop state from *inside* ``agent_loop.run`` via
    a side_effect, then assert the mutated fields have the bypass-path
    values (max_steps ∈ {5,10,20}, validation_level ∈ {1,2,3},
    _skip_routing=True, _current_topology=None, etc.) — NOT the prior
    sentinel state. This catches a hypothetical regression where the
    mutation block is deleted along with the restoration block.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = _make_loop_with_prior_state()
    # P6-B (cycle-11) instance state — `Pipeline.__new__` bypasses
    # `__init__`, so the lazy lock attrs never get set. Inject them so
    # `_get_agent_loop_bypass_lock` can find the slots and lazily build.
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.bandit = None
    pipeline.write_gate = "pipeline-gate-sentinel"
    pipeline.provider_pool = None
    pipeline.llm_provider = None
    pipeline._last_routing_decision = None
    pipeline._emit = MagicMock()
    pipeline.event_bus = None

    captured: dict[str, Any] = {}

    async def _capture_during_run(*_args, **_kwargs):
        # Snapshot the mutated state at the exact moment agent_loop.run
        # is invoked — after the mutation block, before the finally.
        loop = pipeline._agent_loop
        captured["max_steps"] = loop.config.max_steps
        captured["validation_level"] = loop.config.validation_level
        captured["stall_after_tool_steps"] = loop.config.stall_after_tool_steps
        captured["_skip_routing"] = loop._skip_routing
        captured["_current_topology"] = loop._current_topology
        captured["write_gate"] = loop.write_gate
        captured["gate_current_task"] = loop.gate_current_task
        return "bypass-output"

    pipeline._agent_loop.run = AsyncMock(side_effect=_capture_during_run)

    ctx = SimpleNamespace(
        task="bypass-mutation-check",
        topology=None,
        system=2,  # expect max_steps=10, validation_level=2, stall=9
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=True,
        bandit_decision_id=None,
    )
    await pipeline._stage_execute(ctx)

    # The bypass block SHOULD have mutated these — if it didn't, we
    # restored "nothing" and A0a is a no-op.
    assert captured["max_steps"] == 10, (
        f"expected max_steps=10 for system=2; got {captured['max_steps']}. "
        "Either the {1:5,2:10,3:20} scaling line was deleted or "
        "the bypass branch didn't run."
    )
    assert captured["validation_level"] == 2, (
        f"expected validation_level=2 for system=2 (with sandbox_manager); "
        f"got {captured['validation_level']}"
    )
    assert captured["stall_after_tool_steps"] == 9, (
        f"expected stall_after_tool_steps=9 (max_steps - 1); "
        f"got {captured['stall_after_tool_steps']}"
    )
    assert captured["_skip_routing"] is True, (
        "expected _skip_routing=True during bypass run"
    )
    assert captured["_current_topology"] is None, (
        "expected _current_topology=None during bypass run"
    )
    assert captured["write_gate"] == "pipeline-gate-sentinel", (
        "expected pipeline.write_gate to be installed on agent_loop"
    )
    assert captured["gate_current_task"] == "bypass-mutation-check", (
        "expected gate_current_task to be set to ctx.task"
    )


@pytest.mark.asyncio
async def test_bypass_path_restores_even_on_exception():
    """If ``agent_loop.run`` raises, the finally must still restore.

    This is the exception-safety half of the restoration contract —
    the prior 3-field restoration was already exception-safe; this
    test locks the same contract for the expanded 10-field
    restoration.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline as Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = _make_loop_with_prior_state()
    # P6-B (cycle-11) instance state — `Pipeline.__new__` bypasses
    # `__init__`, so the lazy lock attrs never get set. Inject them so
    # `_get_agent_loop_bypass_lock` can find the slots and lazily build.
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.bandit = None  # skip bandit arm-selection in _stage_execute
    pipeline._agent_loop.run = AsyncMock(side_effect=RuntimeError("boom"))
    pipeline.write_gate = _SENTINEL_WRITE_GATE
    pipeline.provider_pool = None
    pipeline.llm_provider = None
    pipeline._last_routing_decision = None
    pipeline._emit = MagicMock()
    pipeline.event_bus = None

    before = _make_bypass_state_snapshot(pipeline._agent_loop)

    ctx = SimpleNamespace(
        task="any task",
        topology=None,
        system=3,  # picks validation_level=3, max_steps=20, stall=19
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=True,
        bandit_decision_id=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await pipeline._stage_execute(ctx)

    after = _make_bypass_state_snapshot(pipeline._agent_loop)
    for k, pre_val in before.items():
        post_val = after[k]
        assert post_val == pre_val or post_val is pre_val, (
            f"Field {k!r} leaked after exception: "
            f"before={pre_val!r}, after={post_val!r}"
        )
