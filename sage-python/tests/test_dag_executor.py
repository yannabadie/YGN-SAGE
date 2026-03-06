"""Tests for DAGExecutor — contract-driven DAG execution with VFs."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import (
    TaskNode,
    IOSchema,
    BudgetConstraint,
    SecurityLabel,
)
from sage.contracts.dag import TaskDAG
from sage.contracts.verification import VFResult, VerificationFn
from sage.contracts.policy import PolicyVerifier
from sage.contracts.executor import DAGExecutor, DAGExecutionResult, NodeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _mock_runner(node_id: str, description: str, input_data: dict) -> dict:
    """Mock task runner that returns the description as 'summary'."""
    return {"summary": f"Result of {description}", **input_data}


async def _failing_runner(node_id: str, description: str, input_data: dict) -> dict:
    raise RuntimeError("Simulated failure")


# ---------------------------------------------------------------------------
# Basic DAG execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_single_node():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a",
        description="Summarize",
        input_schema=IOSchema(fields={"text": "string"}),
        output_schema=IOSchema(fields={"summary": "string"}),
    ))
    executor = DAGExecutor(dag, runner=_mock_runner)
    result = await executor.execute({"text": "hello"})
    assert result.success is True
    assert len(result.node_results) == 1
    assert result.node_results["a"].output["summary"].startswith("Result of")


@pytest.mark.asyncio
async def test_execute_linear_dag():
    """A -> B: B receives A's output."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Step A",
        output_schema=IOSchema(fields={"summary": "string"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="Step B",
        input_schema=IOSchema(fields={"summary": "string"}),
        output_schema=IOSchema(fields={"summary": "string"}),
    ))
    dag.add_edge("a", "b")
    executor = DAGExecutor(dag, runner=_mock_runner)
    result = await executor.execute({})
    assert result.success is True
    assert "a" in result.node_results
    assert "b" in result.node_results


# ---------------------------------------------------------------------------
# Pre-check VF blocks execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_check_blocks_on_missing_input():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    executor = DAGExecutor(dag, runner=_mock_runner)
    result = await executor.execute({})  # Missing 'text'
    assert result.success is False
    assert result.node_results["a"].pre_check_passed is False


# ---------------------------------------------------------------------------
# Post-check VF catches budget overrun
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_check_catches_budget():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Expensive",
        budget=BudgetConstraint(max_cost_usd=0.01),
    ))

    async def expensive_runner(node_id, desc, data):
        return {"_cost_usd": 0.05}  # Over budget

    executor = DAGExecutor(dag, runner=expensive_runner)
    result = await executor.execute({})
    assert result.success is False
    assert result.node_results["a"].post_check_passed is False


# ---------------------------------------------------------------------------
# Policy verification runs before execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policy_violation_blocks_execution():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="Secret",
        security_label=SecurityLabel.TOP,
    ))
    dag.add_node(TaskNode(
        node_id="b", description="Public",
        security_label=SecurityLabel.LOW,
    ))
    dag.add_edge("a", "b")
    executor = DAGExecutor(dag, runner=_mock_runner)
    result = await executor.execute({})
    assert result.success is False
    assert len(result.policy_violations) > 0


# ---------------------------------------------------------------------------
# Runner failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runner_failure_reported():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="Will fail"))
    executor = DAGExecutor(dag, runner=_failing_runner)
    result = await executor.execute({})
    assert result.success is False
    assert "Simulated failure" in result.node_results["a"].error


# ---------------------------------------------------------------------------
# DAGExecutionResult
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execution_result_has_all_fields():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="Simple"))
    executor = DAGExecutor(dag, runner=_mock_runner)
    result = await executor.execute({})
    assert isinstance(result, DAGExecutionResult)
    assert isinstance(result.node_results, dict)
    assert isinstance(result.policy_violations, list)
