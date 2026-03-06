"""Tests for Z3 contract verification on TaskDAG."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint
from sage.contracts.dag import TaskDAG

z3 = pytest.importorskip("z3", reason="z3-solver not installed")

from sage.contracts.z3_verify import (
    verify_capability_coverage,
    verify_budget_feasibility,
    verify_type_compatibility,
    ContractVerdict,
)


# ---------------------------------------------------------------------------
# ContractVerdict
# ---------------------------------------------------------------------------

def test_verdict_pass():
    v = ContractVerdict(satisfied=True, property_name="test")
    assert v.satisfied is True
    assert v.counterexample is None


def test_verdict_fail_with_counterexample():
    v = ContractVerdict(
        satisfied=False,
        property_name="test",
        counterexample="node X needs 'grounding' but no provider offers it",
    )
    assert v.satisfied is False
    assert "grounding" in v.counterexample


# ---------------------------------------------------------------------------
# Capability coverage
# ---------------------------------------------------------------------------

def test_capability_coverage_satisfied():
    """All required capabilities are available from providers."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        capabilities_required=["tool_role", "streaming"],
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        capabilities_required=["structured_output"],
    ))
    available = {"tool_role", "streaming", "structured_output", "grounding"}
    verdict = verify_capability_coverage(dag, available)
    assert verdict.satisfied is True


def test_capability_coverage_missing():
    """A node requires a capability not available."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        capabilities_required=["file_search", "grounding"],
    ))
    available = {"file_search"}  # grounding missing
    verdict = verify_capability_coverage(dag, available)
    assert verdict.satisfied is False
    assert "grounding" in verdict.counterexample


def test_capability_coverage_empty_requirements():
    """No capabilities required — trivially satisfied."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    verdict = verify_capability_coverage(dag, set())
    assert verdict.satisfied is True


# ---------------------------------------------------------------------------
# Budget feasibility
# ---------------------------------------------------------------------------

def test_budget_feasibility_within_limit():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        budget=BudgetConstraint(max_cost_usd=0.05),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        budget=BudgetConstraint(max_cost_usd=0.03),
    ))
    verdict = verify_budget_feasibility(dag, total_budget_usd=0.10)
    assert verdict.satisfied is True


def test_budget_feasibility_exceeded():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        budget=BudgetConstraint(max_cost_usd=0.06),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        budget=BudgetConstraint(max_cost_usd=0.06),
    ))
    verdict = verify_budget_feasibility(dag, total_budget_usd=0.10)
    assert verdict.satisfied is False


def test_budget_feasibility_zero_budget_nodes():
    """Nodes without budget constraints (0.0) should not count."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        budget=BudgetConstraint(max_cost_usd=0.05),
    ))
    verdict = verify_budget_feasibility(dag, total_budget_usd=0.05)
    assert verdict.satisfied is True


# ---------------------------------------------------------------------------
# Type compatibility (I/O across edges)
# ---------------------------------------------------------------------------

def test_type_compatibility_valid():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        output_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_edge("a", "b")
    verdict = verify_type_compatibility(dag)
    assert verdict.satisfied is True


def test_type_compatibility_missing_field():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        output_schema=IOSchema(fields={"score": "float"}),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        input_schema=IOSchema(fields={"text": "string"}),
    ))
    dag.add_edge("a", "b")
    verdict = verify_type_compatibility(dag)
    assert verdict.satisfied is False
    assert "text" in verdict.counterexample


def test_type_compatibility_no_edges():
    """No edges — trivially compatible."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    verdict = verify_type_compatibility(dag)
    assert verdict.satisfied is True
