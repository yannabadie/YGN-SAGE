"""Ablation study — measure the impact of disabling individual components.

Each test compares execution with a component ENABLED vs DISABLED,
verifying that the component provides measurable value (catches errors,
reduces cost, or improves correctness).

Components tested:
1. Pre-check verification — catches schema mismatches
2. Post-check verification — catches budget violations
3. Policy verifier — catches info-flow violations
4. Cost tracker — halts before overspend
5. Repair loop — recovers from transient failures
6. Write gate — prevents low-confidence noise
7. Z3 plan-time verification — catches infeasible plans before execution
"""
from __future__ import annotations

import pytest
from sage.contracts.task_node import (
    TaskNode, IOSchema, BudgetConstraint, SecurityLabel,
)
from sage.contracts.dag import TaskDAG
from sage.contracts.executor import DAGExecutor
from sage.contracts.repair import RepairLoop
from sage.contracts.policy import PolicyVerifier
from sage.contracts.cost_tracker import CostTracker
from sage.memory.write_gate import WriteGate

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

from sage.contracts.z3_verify import (
    verify_capability_coverage,
    verify_budget_feasibility,
    verify_type_compatibility,
)


# ===========================================================================
# AB-1: Pre-check catches schema mismatch (enabled) vs silent corruption (disabled)
# ===========================================================================

@pytest.mark.asyncio
async def test_ab1_precheck_catches_schema_mismatch():
    """WITH pre-check: executor stops on missing input fields.
    WITHOUT pre-check: corrupt data flows silently through the DAG."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="producer",
        description="Produces score",
        output_schema=IOSchema(fields={"score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="consumer",
        description="Needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_edge("producer", "consumer")

    async def runner(nid, desc, data):
        if nid == "producer":
            return {"score": 0.95}
        return {"result": "processed"}

    # WITH pre-check (default) — catches the mismatch
    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert result.node_results["consumer"].pre_check_passed is False


@pytest.mark.asyncio
async def test_ab1_without_precheck_data_flows_silently():
    """Simulate what happens if pre-check were removed: wrong data passes through."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))  # No schema constraint
    dag.add_edge("a", "b")

    async def runner(nid, desc, data):
        return {"garbage": "data"}

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    # Without schemas, anything passes — silent corruption
    assert result.success is True  # This is the PROBLEM ablation reveals


# ===========================================================================
# AB-2: Post-check catches budget violations
# ===========================================================================

@pytest.mark.asyncio
async def test_ab2_postcheck_catches_per_node_budget():
    """Post-check detects when a single node exceeds its budget."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="expensive",
        description="Expensive step",
        budget=BudgetConstraint(max_cost_usd=0.01),
    ))

    async def runner(nid, desc, data):
        return {"result": "ok", "_cost_usd": 0.50}

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert result.node_results["expensive"].post_check_passed is False


# ===========================================================================
# AB-3: Policy verifier catches info-flow violations
# ===========================================================================

@pytest.mark.asyncio
async def test_ab3_policy_blocks_info_flow_violation():
    """Policy verifier prevents TOP→LOW data leak before any code runs."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="secret", description="Classified",
        security_label=SecurityLabel.TOP,
    ))
    dag.add_node(TaskNode(
        node_id="public", description="Public API",
        security_label=SecurityLabel.LOW,
    ))
    dag.add_edge("secret", "public")

    call_count = 0

    async def runner(nid, desc, data):
        nonlocal call_count
        call_count += 1
        return {}

    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert len(result.policy_violations) > 0
    # Key ablation insight: runner was NEVER called
    assert call_count == 0


# ===========================================================================
# AB-4: Cost tracker halts before overspend
# ===========================================================================

@pytest.mark.asyncio
async def test_ab4_cost_tracker_prevents_overspend():
    """With cost tracking: execution halts at budget.
    Without: all nodes run regardless of cost."""
    dag = TaskDAG()
    for i in range(10):
        dag.add_node(TaskNode(node_id=f"n{i}", description=f"Step {i}"))
    for i in range(9):
        dag.add_edge(f"n{i}", f"n{i + 1}")

    async def runner(nid, desc, data):
        return {"result": "ok", "_cost_usd": 0.20}

    # WITH cost tracking (budget=0.50) — halts after 3 nodes (0.60 > 0.50)
    executor = DAGExecutor(dag, runner=runner, total_budget_usd=0.50)
    result = await executor.execute({})
    assert result.success is False
    nodes_executed = len(result.node_results)
    assert nodes_executed < 10  # Stopped early
    assert executor.cost_tracker.total_spent <= 0.70  # Didn't run all 10

    # WITHOUT cost tracking (budget=0) — all 10 run
    executor2 = DAGExecutor(dag, runner=runner, total_budget_usd=0.0)
    result2 = await executor2.execute({})
    assert result2.success is True
    assert len(result2.node_results) == 10


# ===========================================================================
# AB-5: Repair loop recovers from transient failures
# ===========================================================================

@pytest.mark.asyncio
async def test_ab5_repair_recovers_vs_executor_fails():
    """RepairLoop recovers from transient failure.
    Plain DAGExecutor fails immediately."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="flaky",
        description="Flaky step",
        output_schema=IOSchema(fields={"answer": "string"}),
    ))

    attempts = {"count": 0}

    async def flaky_runner(nid, desc, data):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return {"wrong": "field"}  # Missing 'answer'
        return {"answer": "42"}

    # Plain executor — fails on first bad output
    attempts["count"] = 0
    executor = DAGExecutor(dag, runner=flaky_runner)
    result = await executor.execute({})
    assert result.success is False

    # Repair loop — retries and succeeds
    attempts["count"] = 0
    loop = RepairLoop(dag, runner=flaky_runner, max_retries=5)
    result2 = await loop.execute({})
    assert result2.success is True


# ===========================================================================
# AB-6: Write gate prevents noisy writes
# ===========================================================================

def test_ab6_write_gate_filters_noise():
    """With write gate: low-confidence data is blocked.
    Without: everything gets stored."""
    gate = WriteGate(threshold=0.5)

    # High confidence — allowed
    d1 = gate.evaluate("Important finding about Z3", confidence=0.9)
    assert d1.allowed is True

    # Low confidence — blocked
    d2 = gate.evaluate("Maybe something happened", confidence=0.2)
    assert d2.allowed is False

    # Empty — blocked
    d3 = gate.evaluate("", confidence=0.9)
    assert d3.allowed is False

    assert gate.abstention_count >= 1  # At least one blocked


# ===========================================================================
# AB-7: Z3 plan-time verification prevents infeasible plans
# ===========================================================================

def test_ab7_z3_catches_infeasible_budget():
    """Z3 catches budget infeasibility BEFORE execution starts."""
    dag = TaskDAG()
    for i in range(5):
        dag.add_node(TaskNode(
            node_id=f"n{i}",
            description=f"Step {i}",
            budget=BudgetConstraint(max_cost_usd=3.0),
        ))

    # Total node budgets: 5 * 3.0 = 15.0, but total budget is 10.0
    verdict = verify_budget_feasibility(dag, total_budget_usd=10.0)
    assert verdict.satisfied is False
    # Key insight: caught at PLAN TIME, no API calls made


def test_ab7_z3_catches_missing_capabilities():
    """Z3 catches capability gaps before execution."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="needs_search",
        description="Needs file search",
        capabilities_required=["file_search"],
    ))

    # No file_search available
    verdict = verify_capability_coverage(dag, {"streaming", "structured_output"})
    assert verdict.satisfied is False


# ===========================================================================
# AB-8: Combined ablation — full pipeline vs bare executor
# ===========================================================================

@pytest.mark.asyncio
async def test_ab8_full_pipeline_value():
    """Demonstrate that the full pipeline (policy + VF + cost tracking)
    catches issues that a bare runner would miss."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="step1",
        description="Step 1",
        output_schema=IOSchema(fields={"data": "string"}),
        budget=BudgetConstraint(max_cost_usd=0.05),
    ))
    dag.add_node(TaskNode(
        node_id="step2",
        description="Step 2",
        input_schema=IOSchema(fields={"data": "string"}),
    ))
    dag.add_edge("step1", "step2")

    async def runner(nid, desc, data):
        if nid == "step1":
            return {"data": "result", "_cost_usd": 0.10}  # Over per-node budget
        return {"output": "done"}

    # Full pipeline catches the per-node budget violation
    executor = DAGExecutor(dag, runner=runner)
    result = await executor.execute({})
    assert result.success is False
    assert result.node_results["step1"].post_check_passed is False

    # Verify step2 never ran (cascade prevention)
    assert "step2" not in result.node_results
