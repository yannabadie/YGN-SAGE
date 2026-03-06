"""Tests for counterexample-guided repair with hard fences."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint
from sage.contracts.dag import TaskDAG
from sage.contracts.verification import VFResult
from sage.contracts.repair import RepairLoop, RepairAction, RepairResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_dag():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Generate",
        output_schema=IOSchema(fields={"text": "string"}),
    ))
    return dag


# ---------------------------------------------------------------------------
# RepairAction
# ---------------------------------------------------------------------------

def test_repair_action_retry():
    action = RepairAction.from_failure(
        node_id="a",
        vf_result=VFResult(passed=False, message="Output missing 'text'"),
        attempt=1,
        max_retries=3,
    )
    assert action.action_type == "retry"
    assert action.node_id == "a"
    assert "text" in action.constraint


def test_repair_action_escalate():
    """After max retries, action should be 'escalate'."""
    action = RepairAction.from_failure(
        node_id="a",
        vf_result=VFResult(passed=False, message="Budget exceeded"),
        attempt=3,
        max_retries=3,
    )
    assert action.action_type == "escalate"


def test_repair_action_abort():
    """After escalation limit, action should be 'abort'."""
    action = RepairAction.from_failure(
        node_id="a",
        vf_result=VFResult(passed=False, message="Fatal"),
        attempt=6,
        max_retries=3,
    )
    assert action.action_type == "abort"


# ---------------------------------------------------------------------------
# RepairLoop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_repair_loop_succeeds_first_try():
    """Runner succeeds immediately — no repair needed."""
    call_count = 0

    async def runner(node_id, desc, data):
        nonlocal call_count
        call_count += 1
        return {"text": "hello"}

    dag = _make_simple_dag()
    loop = RepairLoop(dag, runner=runner, max_retries=3)
    result = await loop.execute({"input": "test"})
    assert result.success is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_repair_loop_retries_on_failure():
    """Runner fails twice then succeeds — repair loop retries."""
    call_count = 0

    async def flaky_runner(node_id, desc, data):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {}  # Missing 'text' — will fail post-check
        return {"text": "finally"}

    dag = _make_simple_dag()
    loop = RepairLoop(dag, runner=flaky_runner, max_retries=3)
    result = await loop.execute({})
    assert result.success is True
    assert call_count == 3


@pytest.mark.asyncio
async def test_repair_loop_aborts_after_max():
    """Runner always fails — repair loop aborts."""
    async def always_fail(node_id, desc, data):
        return {}  # Never produces 'text'

    dag = _make_simple_dag()
    loop = RepairLoop(dag, runner=always_fail, max_retries=2)
    result = await loop.execute({})
    assert result.success is False
    assert len(result.repair_actions) > 0
    assert result.repair_actions[-1].action_type in ("escalate", "abort")


@pytest.mark.asyncio
async def test_repair_result_tracks_attempts():
    call_count = 0

    async def fail_once(node_id, desc, data):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {}
        return {"text": "ok"}

    dag = _make_simple_dag()
    loop = RepairLoop(dag, runner=fail_once, max_retries=3)
    result = await loop.execute({})
    assert result.total_attempts >= 2


@pytest.mark.asyncio
async def test_repair_loop_runner_exception():
    """Runner raises exception — should be caught and retried."""
    call_count = 0

    async def exploding_runner(node_id, desc, data):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RuntimeError("Boom")
        return {"text": "recovered"}

    dag = _make_simple_dag()
    loop = RepairLoop(dag, runner=exploding_runner, max_retries=3)
    result = await loop.execute({})
    assert result.success is True
