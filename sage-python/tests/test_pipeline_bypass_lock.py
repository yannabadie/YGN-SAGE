"""Regression tests for P6-B (cycle-11, cgpro round-4 review 2026-05-04).

Locks the contract that the AgentLoop bypass mutation block is
serialized via a lazy per-event-loop ``asyncio.Lock`` AND guarded by a
``ContextVar`` against re-entry from inside the same task (the
``sage_recurse`` deadlock case).

Why P6-B is defensive only:
    The boot singleton ``AgentLoop`` is shared across all pipeline runs
    that reach the single-agent bypass path. The block mutates 12
    fields on the singleton, runs ``await agent_loop.run(task)``, then
    restores from a per-call snapshot. Two concurrent runs on the same
    event loop would interleave snapshot/mutate/restore and clobber
    each other's state. P6-B serializes the block on a lock and aborts
    re-entry; P6-A (per-run AgentLoop factory, the structural fix) is
    deferred behind ADR-015 characterization tests.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.pipeline import (
    _BYPASS_AGENT_LOOP_ACTIVE,
    CognitiveOrchestrationPipeline as Pipeline,
)


def _make_loop_with_prior_state() -> MagicMock:
    """Mock AgentLoop with all 12 fields the bypass block snapshots."""
    config = SimpleNamespace(
        validation_level=99,
        max_steps=123,
        stall_after_tool_steps=77,
        llm=SimpleNamespace(model="prior-model"),
    )
    loop = MagicMock(
        config=config,
        sandbox_manager=MagicMock(),
        _skip_routing=False,
        _current_topology=object(),
        write_gate=object(),
        gate_current_task=object(),
        gate_source_tier="prior-tier",
        _on_drift=lambda *a, **kw: None,
        _run_frame_builder=None,
        _runtime_node_run_id=None,
        _llm=SimpleNamespace(name="prior-provider"),
        total_cost_usd=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
    )
    loop.run = AsyncMock(return_value="bypass-output")
    return loop


def _make_pipeline_for_bypass(agent_loop) -> Pipeline:
    """Stub a Pipeline that only exercises ``_stage_execute`` bypass branch."""
    pipeline = Pipeline.__new__(Pipeline)
    pipeline._agent_loop = agent_loop
    pipeline._agent_loop_bypass_lock = None
    pipeline._agent_loop_bypass_lock_loop = None
    pipeline.bandit = None
    pipeline.write_gate = object()
    pipeline.provider_pool = None
    pipeline.llm_provider = None
    pipeline._last_routing_decision = None
    pipeline._emit = MagicMock()
    pipeline.event_bus = None
    return pipeline


def _make_ctx(task: str = "task", topology: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        topology=topology,
        system=2,
        result=None,
        cost=0.0,
        tool_call_count=0,
        tool_turn_count=0,
        executed_commands=[],
        verification_passed=True,
        bandit_decision_id=None,
    )


@pytest.mark.asyncio
async def test_concurrent_bypass_serialized_max_one_active():
    """Two coroutines hitting the bypass block on the same event loop
    must never both be inside the mutation window simultaneously.

    Without the P6-B lock, both would snapshot/mutate/restore in
    interleaved order and clobber each other's restoration.
    """
    active_count = 0
    max_active = 0
    enter_count = 0

    async def _slow_run(*_args, **_kwargs):
        nonlocal active_count, max_active, enter_count
        active_count += 1
        enter_count += 1
        max_active = max(max_active, active_count)
        # Yield to the event loop so a concurrent coroutine that
        # bypassed the lock would get a chance to enter and bump
        # active_count to 2.
        await asyncio.sleep(0.01)
        active_count -= 1
        return "ok"

    loop_mock = _make_loop_with_prior_state()
    loop_mock.run = AsyncMock(side_effect=_slow_run)
    pipeline = _make_pipeline_for_bypass(loop_mock)

    await asyncio.gather(
        pipeline._stage_execute(_make_ctx("task-A")),
        pipeline._stage_execute(_make_ctx("task-B")),
        pipeline._stage_execute(_make_ctx("task-C")),
    )

    assert enter_count == 3, "all three runs must reach agent_loop.run"
    assert max_active == 1, (
        f"bypass block was entered concurrently: max_active={max_active} "
        "— P6-B lock is not actually serializing"
    )


@pytest.mark.asyncio
async def test_bypass_lock_released_after_exception_and_state_restored():
    """If ``agent_loop.run`` raises, the lock must be released AND the
    12-field restoration must still run. A subsequent bypass call must
    succeed (does not deadlock waiting for a never-released lock).
    """
    loop_mock = _make_loop_with_prior_state()
    prior_max_steps = loop_mock.config.max_steps
    prior_validation = loop_mock.config.validation_level
    prior_skip_routing = loop_mock._skip_routing

    pipeline = _make_pipeline_for_bypass(loop_mock)

    # First call: agent_loop.run raises
    loop_mock.run = AsyncMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await pipeline._stage_execute(_make_ctx("task-fail"))

    # Restoration must have happened despite the exception
    assert loop_mock.config.max_steps == prior_max_steps, (
        "max_steps not restored after exception"
    )
    assert loop_mock.config.validation_level == prior_validation, (
        "validation_level not restored after exception"
    )
    assert loop_mock._skip_routing is prior_skip_routing, (
        "_skip_routing not restored after exception"
    )

    # ContextVar must be reset (False) outside the failed call
    assert _BYPASS_AGENT_LOOP_ACTIVE.get() is False, (
        "ContextVar leaked True past the failed run"
    )

    # Lock must be released — second bypass on the same pipeline must
    # complete (not hang). pytest-asyncio's wait_for guards against
    # the deadlock case taking the whole suite hostage.
    loop_mock.run = AsyncMock(return_value="recovered")
    result_ctx = await asyncio.wait_for(
        pipeline._stage_execute(_make_ctx("task-recover")),
        timeout=5.0,
    )
    assert result_ctx.result == "recovered"


@pytest.mark.asyncio
async def test_bypass_lock_released_after_cancellation_and_state_restored():
    """Cancelling the awaiting task must release the lock and trigger
    the same 12-field restoration. Without finally semantics on the
    ContextVar reset + the inner finally on the snapshot, a cancel
    mid-run could leave the singleton dirty AND the lock held forever.
    """
    loop_mock = _make_loop_with_prior_state()
    prior_max_steps = loop_mock.config.max_steps
    prior_validation = loop_mock.config.validation_level

    pipeline = _make_pipeline_for_bypass(loop_mock)

    # Determinism: signal entry into agent_loop.run via an Event so we
    # cancel only after the bypass block has snapshotted + mutated the
    # singleton. Cancelling earlier would race past the lock and prove
    # nothing about the cancel-mid-bypass path.
    started = asyncio.Event()

    async def _long_sleep(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(60.0)
        return "never"

    loop_mock.run = AsyncMock(side_effect=_long_sleep)

    task = asyncio.create_task(pipeline._stage_execute(_make_ctx("task-cancel")))
    await asyncio.wait_for(started.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Restoration ran despite cancellation
    assert loop_mock.config.max_steps == prior_max_steps, (
        "max_steps not restored after cancellation"
    )
    assert loop_mock.config.validation_level == prior_validation, (
        "validation_level not restored after cancellation"
    )

    # ContextVar reset
    assert _BYPASS_AGENT_LOOP_ACTIVE.get() is False, (
        "ContextVar leaked True past the cancelled run"
    )

    # Lock released — a new bypass call completes
    loop_mock.run = AsyncMock(return_value="after-cancel")
    result_ctx = await asyncio.wait_for(
        pipeline._stage_execute(_make_ctx("task-after-cancel")),
        timeout=5.0,
    )
    assert result_ctx.result == "after-cancel"


@pytest.mark.asyncio
async def test_recursive_bypass_fails_fast_no_deadlock():
    """If the bypass block is re-entered from inside the same task —
    e.g. the ``sage_recurse`` tool calling ``system.run`` from within
    ``agent_loop.run`` — the ContextVar guard must raise immediately
    instead of awaiting a lock the same task already holds (deadlock).
    """
    loop_mock = _make_loop_with_prior_state()
    pipeline = _make_pipeline_for_bypass(loop_mock)

    inner_error: list[Exception] = []

    async def _reentrant_run(*_args, **_kwargs):
        # Simulate sage_recurse re-entering pipeline.run from inside
        # the AgentLoop step. The reentry guard MUST fire before the
        # inner call awaits the lock (which the outer task holds).
        try:
            await pipeline._stage_execute(_make_ctx("task-inner"))
        except RuntimeError as exc:
            inner_error.append(exc)
            return "outer-after-guarded-inner"
        return "outer-but-inner-deadlocked"

    loop_mock.run = AsyncMock(side_effect=_reentrant_run)

    # If the guard doesn't fire, this call deadlocks. wait_for caps
    # the failure at 5 s instead of hanging the suite.
    result_ctx = await asyncio.wait_for(
        pipeline._stage_execute(_make_ctx("task-outer")),
        timeout=5.0,
    )

    assert len(inner_error) == 1, (
        "expected the recursive bypass to be rejected by the ContextVar "
        f"guard; got {len(inner_error)} errors"
    )
    assert "Recursive AgentLoop bypass" in str(inner_error[0]), (
        f"unexpected error message: {inner_error[0]!r}"
    )
    assert result_ctx.result == "outer-after-guarded-inner"


@pytest.mark.asyncio
async def test_non_bypass_path_does_not_acquire_lock():
    """The lock is only relevant to the AgentLoop bypass branch. The
    topology / TopologyRunner branch must NOT acquire it (would
    serialize unrelated topology runs through a single mutex).

    Smoke: a pipeline with ``ctx.topology`` set to a non-None mock
    short-circuits before the ``if self._agent_loop:`` block and
    therefore never builds the lock.
    """
    loop_mock = _make_loop_with_prior_state()
    pipeline = _make_pipeline_for_bypass(loop_mock)

    # Force the non-bypass path: provide a topology stub. The pipeline
    # stub has no real TopologyRunner; we only need to confirm we
    # didn't hit the bypass branch (which would mutate the singleton).
    topology = MagicMock()
    topology.node_count = MagicMock(return_value=3)

    # Patch the helper that determines bypass eligibility so the test
    # is robust to whichever branch the real `_is_single_agent_execution`
    # would take. Returning False bypasses the bypass block entirely.
    pipeline._is_single_agent_execution = MagicMock(return_value=False)

    # Stub topology runner attributes so _stage_execute can dispatch
    # without actually executing a graph. We don't care what happens
    # downstream — we only assert the lock state.
    pipeline.topology_runner = MagicMock()
    pipeline.topology_runner.run = AsyncMock(return_value=("topology-result", 0.0))

    ctx = _make_ctx("task-topology", topology=topology)
    try:
        await pipeline._stage_execute(ctx)
    except Exception:
        # The stubbed pipeline is incomplete for the topology branch;
        # we tolerate any downstream attribute error. The contract we
        # care about is that the lock was never built.
        pass

    assert pipeline._agent_loop_bypass_lock is None, (
        "non-bypass path must not lazily build the bypass lock"
    )
    assert _BYPASS_AGENT_LOOP_ACTIVE.get() is False, (
        "ContextVar must remain False on non-bypass paths"
    )
