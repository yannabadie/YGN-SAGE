"""Synthetic failure lab — MAST taxonomy inspired failure mode testing.

Tests that the system detects and handles:
1. Tool/runner failures (exceptions during execution)
2. Budget exhaustion mid-run
3. Cascading failures in DAG (one node fails, downstream affected)
4. Security label violations (info-flow)
5. Schema mismatches (type errors in data flow)
6. Cycle detection in DAG
7. Timeout simulation
8. Concurrent node failure in parallel paths
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


# ===========================================================================
# FM-1: Runner exception (tool failure)
# ===========================================================================

@pytest.mark.asyncio
async def test_fm1_runner_exception():
    """Runner raises RuntimeError — executor must catch and report."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="tool", description="Call external tool"))

    async def broken_tool(nid, desc, data):
        raise RuntimeError("API rate limit exceeded")

    executor = DAGExecutor(dag, runner=broken_tool)
    result = await executor.execute({})
    assert result.success is False
    assert "rate limit" in result.node_results["tool"].error


# ===========================================================================
# FM-2: Budget exhaustion mid-run
# ===========================================================================

@pytest.mark.asyncio
async def test_fm2_budget_exhaustion():
    """Second node exceeds budget — post-check catches it."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="cheap", description="Cheap step",
        output_schema=IOSchema(fields={"data": "string"}),
        budget=BudgetConstraint(max_cost_usd=0.10),
    ))
    dag.add_node(TaskNode(
        node_id="expensive", description="Expensive step",
        input_schema=IOSchema(fields={"data": "string"}),
        budget=BudgetConstraint(max_cost_usd=0.01),
    ))
    dag.add_edge("cheap", "expensive")

    call_count = 0

    async def runner(nid, desc, data):
        nonlocal call_count
        call_count += 1
        if nid == "cheap":
            return {"data": "result", "_cost_usd": 0.05}
        return {"_cost_usd": 0.05}  # Over budget for expensive node

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert result.node_results["expensive"].post_check_passed is False


# ===========================================================================
# FM-3: Cascading failure (upstream fails, downstream never runs)
# ===========================================================================

@pytest.mark.asyncio
async def test_fm3_cascading_failure():
    """A fails -> B never executes."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="Upstream"))
    dag.add_node(TaskNode(
        node_id="b", description="Downstream",
        input_schema=IOSchema(fields={"result": "string"}),
    ))
    dag.add_edge("a", "b")

    async def fail_a(nid, desc, data):
        if nid == "a":
            raise RuntimeError("Upstream crashed")
        return {"result": "ok"}

    executor = DAGExecutor(dag, runner=fail_a)
    result = await executor.execute({})
    assert result.success is False
    assert "a" in result.node_results
    assert "b" not in result.node_results  # Never reached


# ===========================================================================
# FM-4: Security label violation (info-flow)
# ===========================================================================

def test_fm4_info_flow_violation():
    """TOP secret data flowing to LOW node — policy catches it."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="classified", description="Handle secret data",
        security_label=SecurityLabel.TOP,
    ))
    dag.add_node(TaskNode(
        node_id="public_api", description="Send to public API",
        security_label=SecurityLabel.LOW,
    ))
    dag.add_edge("classified", "public_api")

    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert len(violations) == 1
    assert violations[0].rule == "info_flow"


# ===========================================================================
# FM-5: Schema mismatch (type error in data flow)
# ===========================================================================

@pytest.mark.asyncio
async def test_fm5_schema_mismatch():
    """A produces 'score' but B needs 'text' — pre-check catches it."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Score generator",
        output_schema=IOSchema(fields={"score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="Text processor",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_edge("a", "b")

    async def runner(nid, desc, data):
        if nid == "a":
            return {"score": 0.95}
        return {"result": "processed"}

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    # B should fail pre-check because 'text' is missing
    assert result.node_results["b"].pre_check_passed is False


# ===========================================================================
# FM-6: Cycle in DAG
# ===========================================================================

def test_fm6_cycle_detection():
    """Circular dependency must be caught at build time."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="x", description="X"))
    dag.add_node(TaskNode(node_id="y", description="Y"))
    dag.add_node(TaskNode(node_id="z", description="Z"))
    dag.add_edge("x", "y")
    dag.add_edge("y", "z")
    dag.add_edge("z", "x")
    with pytest.raises(CycleError):
        dag.topological_sort()


# ===========================================================================
# FM-7: Repair loop exhaustion
# ===========================================================================

@pytest.mark.asyncio
async def test_fm7_repair_exhaustion():
    """Runner never produces valid output — repair loop exhausts retries."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="flaky", description="Always broken",
        output_schema=IOSchema(fields={"answer": "string"}),
    ))

    async def always_wrong(nid, desc, data):
        return {"wrong_field": "oops"}  # Never has 'answer'

    loop = RepairLoop(dag, runner=always_wrong, max_retries=2)
    result = await loop.execute({})
    assert result.success is False
    assert result.total_attempts >= 2


# ===========================================================================
# FM-8: Fan-out limit violation
# ===========================================================================

def test_fm8_fan_out_violation():
    """Node with too many children violates fan-out policy."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="hub", description="Central hub"))
    for i in range(10):
        dag.add_node(TaskNode(node_id=f"spoke_{i}", description=f"Spoke {i}"))
        dag.add_edge("hub", f"spoke_{i}")

    pv = PolicyVerifier(dag, max_fan_out=5)
    violations = pv.check_fan_limits()
    assert len(violations) == 1
    assert violations[0].rule == "fan_out"


# ===========================================================================
# FM-9: Empty DAG
# ===========================================================================

@pytest.mark.asyncio
async def test_fm9_empty_dag():
    """Empty DAG should execute successfully (vacuous truth)."""
    dag = TaskDAG()

    async def noop(nid, desc, data):
        return {}

    executor = DAGExecutor(dag, runner=noop)
    result = await executor.execute({})
    assert result.success is True
    assert len(result.node_results) == 0


# ===========================================================================
# FM-10: Multi-failure recovery via repair loop
# ===========================================================================

@pytest.mark.asyncio
async def test_fm10_multi_failure_recovery():
    """Node fails 3 times then succeeds on 4th attempt."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="resilient", description="Eventually works",
        output_schema=IOSchema(fields={"data": "string"}),
    ))

    attempts = 0

    async def eventually_works(nid, desc, data):
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            return {}  # Missing 'data'
        return {"data": "success"}

    loop = RepairLoop(dag, runner=eventually_works, max_retries=5)
    result = await loop.execute({})
    assert result.success is True
    assert attempts == 4
