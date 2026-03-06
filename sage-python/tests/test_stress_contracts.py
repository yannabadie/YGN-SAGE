"""Stress tests for Contract IR — large DAGs, deep chains, wide fan-out.

Verifies that TaskDAG, Z3 verification, PolicyVerifier, DAGExecutor,
and RepairLoop handle non-trivial graphs correctly.
"""
from __future__ import annotations

import pytest
from sage.contracts.task_node import (
    TaskNode, IOSchema, BudgetConstraint, SecurityLabel,
)
from sage.contracts.dag import TaskDAG, CycleError
from sage.contracts.executor import DAGExecutor
from sage.contracts.repair import RepairLoop
from sage.contracts.policy import PolicyVerifier
from sage.contracts.cost_tracker import CostTracker

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

from sage.contracts.z3_verify import (
    verify_capability_coverage,
    verify_budget_feasibility,
    verify_type_compatibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_chain(n: int, *, budget_per_node: float = 0.0) -> TaskDAG:
    """Build a linear chain A0 -> A1 -> ... -> A(n-1)."""
    dag = TaskDAG()
    for i in range(n):
        dag.add_node(TaskNode(
            node_id=f"n{i}",
            description=f"Step {i}",
            budget=BudgetConstraint(max_cost_usd=budget_per_node),
        ))
    for i in range(n - 1):
        dag.add_edge(f"n{i}", f"n{i + 1}")
    return dag


def _wide_fan_out(hub: str, spokes: int) -> TaskDAG:
    """Build a hub -> spoke_0 ... spoke_(n-1) fan-out."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id=hub, description="Hub"))
    for i in range(spokes):
        dag.add_node(TaskNode(node_id=f"spoke_{i}", description=f"Spoke {i}"))
        dag.add_edge(hub, f"spoke_{i}")
    return dag


def _diamond(n_layers: int) -> TaskDAG:
    """Build a diamond: start -> n parallel paths -> end."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="start", description="Start"))
    dag.add_node(TaskNode(node_id="end", description="End"))
    for i in range(n_layers):
        nid = f"mid_{i}"
        dag.add_node(TaskNode(node_id=nid, description=f"Middle {i}"))
        dag.add_edge("start", nid)
        dag.add_edge(nid, "end")
    return dag


# ===========================================================================
# ST-1: Deep linear chain (50 nodes)
# ===========================================================================

def test_st1_deep_chain_topo_sort():
    dag = _linear_chain(50)
    order = dag.topological_sort()
    assert len(order) == 50
    assert order[0] == "n0"
    assert order[-1] == "n49"


@pytest.mark.asyncio
async def test_st1_deep_chain_execution():
    dag = _linear_chain(50)

    async def runner(nid, desc, data):
        step = int(nid[1:])
        return {"step": step, "_cost_usd": 0.001}

    executor = DAGExecutor(dag, runner=runner, total_budget_usd=1.0)
    result = await executor.execute({})
    assert result.success is True
    assert len(result.node_results) == 50
    assert executor.cost_tracker.total_spent == pytest.approx(0.05)


# ===========================================================================
# ST-2: Wide fan-out (100 spokes)
# ===========================================================================

def test_st2_wide_fan_out_topo_sort():
    dag = _wide_fan_out("hub", 100)
    order = dag.topological_sort()
    assert order[0] == "hub"
    assert len(order) == 101


def test_st2_wide_fan_out_policy():
    dag = _wide_fan_out("hub", 100)
    pv = PolicyVerifier(dag, max_fan_out=50)
    violations = pv.check_fan_limits()
    assert len(violations) == 1
    assert violations[0].rule == "fan_out"


# ===========================================================================
# ST-3: Diamond topology (20 parallel paths)
# ===========================================================================

def test_st3_diamond_topo_sort():
    dag = _diamond(20)
    order = dag.topological_sort()
    assert order[0] == "start"
    assert order[-1] == "end"
    assert len(order) == 22


@pytest.mark.asyncio
async def test_st3_diamond_execution():
    dag = _diamond(20)

    async def runner(nid, desc, data):
        return {"from": nid}

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({"seed": 42})
    assert result.success is True
    assert len(result.node_results) == 22


# ===========================================================================
# ST-4: Z3 capability coverage on large DAG
# ===========================================================================

def test_st4_z3_capability_50_nodes():
    dag = TaskDAG()
    caps = ["structured_output", "streaming", "tool_role"]
    for i in range(50):
        dag.add_node(TaskNode(
            node_id=f"n{i}",
            description=f"Node {i}",
            capabilities_required=[caps[i % len(caps)]],
        ))
    for i in range(49):
        dag.add_edge(f"n{i}", f"n{i + 1}")

    # All capabilities available
    verdict = verify_capability_coverage(dag, set(caps))
    assert verdict.satisfied is True

    # Missing one capability
    verdict2 = verify_capability_coverage(dag, {"structured_output", "streaming"})
    assert verdict2.satisfied is False


# ===========================================================================
# ST-5: Z3 budget feasibility on large DAG
# ===========================================================================

def test_st5_z3_budget_50_nodes():
    dag = _linear_chain(50, budget_per_node=0.10)

    # Budget is 5.0, sum of per-node is 5.0 — should be exactly feasible
    verdict = verify_budget_feasibility(dag, total_budget_usd=5.0)
    assert verdict.satisfied is True

    # Budget too low
    verdict2 = verify_budget_feasibility(dag, total_budget_usd=4.0)
    assert verdict2.satisfied is False


# ===========================================================================
# ST-6: Type compatibility chain with schemas
# ===========================================================================

def test_st6_type_compatibility_chain():
    dag = TaskDAG()
    for i in range(10):
        dag.add_node(TaskNode(
            node_id=f"n{i}",
            description=f"Node {i}",
            input_schema=IOSchema(fields={"data": "string"}) if i > 0 else IOSchema(),
            output_schema=IOSchema(fields={"data": "string"}),
        ))
    for i in range(9):
        dag.add_edge(f"n{i}", f"n{i + 1}")

    verdict = verify_type_compatibility(dag)
    assert verdict.satisfied is True


def test_st6_type_incompatibility():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a",
        description="Produces score",
        output_schema=IOSchema(fields={"score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="b",
        description="Needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_edge("a", "b")

    verdict = verify_type_compatibility(dag)
    assert verdict.satisfied is False


# ===========================================================================
# ST-7: Cost tracker at scale
# ===========================================================================

def test_st7_cost_tracker_100_nodes():
    tracker = CostTracker(budget_usd=10.0)
    for i in range(100):
        tracker.record(f"n{i}", 0.09)
    assert tracker.total_spent == pytest.approx(9.0)
    assert tracker.is_over_budget is False
    assert tracker.remaining == pytest.approx(1.0)

    # One more pushes over
    tracker.record("n100", 1.5)
    assert tracker.is_over_budget is True


# ===========================================================================
# ST-8: Mixed security labels — large lattice
# ===========================================================================

def test_st8_info_flow_large_dag():
    dag = TaskDAG()
    # 10 TOP nodes feeding into 10 LOW nodes — all violations
    for i in range(10):
        dag.add_node(TaskNode(
            node_id=f"top_{i}",
            description=f"Top secret {i}",
            security_label=SecurityLabel.TOP,
        ))
        dag.add_node(TaskNode(
            node_id=f"low_{i}",
            description=f"Low {i}",
            security_label=SecurityLabel.LOW,
        ))
        dag.add_edge(f"top_{i}", f"low_{i}")

    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert len(violations) == 10


# ===========================================================================
# ST-9: Repair loop on deep chain — fails once per node then succeeds
# ===========================================================================

@pytest.mark.asyncio
async def test_st9_repair_deep_chain():
    dag = TaskDAG()
    for i in range(5):
        dag.add_node(TaskNode(
            node_id=f"n{i}",
            description=f"Step {i}",
            output_schema=IOSchema(fields={"val": "string"}),
        ))
    for i in range(4):
        dag.add_edge(f"n{i}", f"n{i + 1}")

    attempts: dict[str, int] = {}

    async def flaky(nid, desc, data):
        attempts[nid] = attempts.get(nid, 0) + 1
        if attempts[nid] < 2:
            return {"wrong": "field"}
        return {"val": f"ok-{nid}"}

    loop = RepairLoop(dag, runner=flaky, max_retries=3)
    result = await loop.execute({})
    assert result.success is True


# ===========================================================================
# ST-10: Executor budget halt on large chain
# ===========================================================================

@pytest.mark.asyncio
async def test_st10_budget_halt_large_chain():
    dag = _linear_chain(20)

    async def expensive(nid, desc, data):
        return {"result": "ok", "_cost_usd": 0.10}

    executor = DAGExecutor(dag, runner=expensive, total_budget_usd=0.50)
    result = await executor.execute({})
    # After 5 nodes: 0.50, after 6th: 0.60 > 0.50 → halt
    assert result.success is False
    executed = len(result.node_results)
    assert executed == 6  # 6th node triggers the halt
    assert executor.cost_tracker.total_spent == pytest.approx(0.60)
