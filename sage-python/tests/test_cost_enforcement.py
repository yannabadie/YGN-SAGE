"""Tests for mid-loop cost budget enforcement."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint
from sage.contracts.dag import TaskDAG
from sage.contracts.executor import DAGExecutor
from sage.contracts.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

def test_tracker_within_budget():
    tracker = CostTracker(budget_usd=1.0)
    tracker.record(node_id="a", cost_usd=0.3)
    tracker.record(node_id="b", cost_usd=0.4)
    assert tracker.remaining == pytest.approx(0.3)
    assert tracker.is_over_budget is False


def test_tracker_over_budget():
    tracker = CostTracker(budget_usd=0.50)
    tracker.record(node_id="a", cost_usd=0.3)
    tracker.record(node_id="b", cost_usd=0.3)
    assert tracker.is_over_budget is True
    assert tracker.total_spent == pytest.approx(0.6)


def test_tracker_no_budget():
    """No budget set — never over budget."""
    tracker = CostTracker(budget_usd=0.0)
    tracker.record(node_id="a", cost_usd=100.0)
    assert tracker.is_over_budget is False


def test_tracker_stats():
    tracker = CostTracker(budget_usd=1.0)
    tracker.record(node_id="a", cost_usd=0.1)
    tracker.record(node_id="b", cost_usd=0.2)
    stats = tracker.stats()
    assert stats["total_spent"] == pytest.approx(0.3)
    assert stats["budget"] == 1.0
    assert stats["remaining"] == pytest.approx(0.7)
    assert stats["per_node"]["a"] == pytest.approx(0.1)


def test_tracker_node_costs():
    tracker = CostTracker(budget_usd=1.0)
    tracker.record(node_id="x", cost_usd=0.5)
    assert tracker.cost_for("x") == pytest.approx(0.5)
    assert tracker.cost_for("unknown") == 0.0


# ---------------------------------------------------------------------------
# DAGExecutor with cost tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_tracks_cost():
    """DAGExecutor should track cumulative cost via _cost_usd in output."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="Step A"))
    dag.add_node(TaskNode(node_id="b", description="Step B"))
    dag.add_edge("a", "b")

    async def runner(nid, desc, data):
        return {"result": "ok", "_cost_usd": 0.05}

    executor = DAGExecutor(dag, runner=runner, total_budget_usd=1.0)
    result = await executor.execute({})
    assert result.success is True


@pytest.mark.asyncio
async def test_executor_halts_on_budget_overrun():
    """Executor should halt if cumulative cost exceeds total_budget_usd."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="Expensive A"))
    dag.add_node(TaskNode(node_id="b", description="Expensive B"))
    dag.add_node(TaskNode(node_id="c", description="Expensive C"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")

    async def expensive(nid, desc, data):
        return {"result": "ok", "_cost_usd": 0.40}

    executor = DAGExecutor(dag, runner=expensive, total_budget_usd=0.50)
    result = await executor.execute({})
    # Should fail after node B (cumulative 0.80 > 0.50)
    assert result.success is False
