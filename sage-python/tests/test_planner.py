"""Tests for TaskPlanner — Plan-and-Act inspired task decomposition."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint, SecurityLabel
from sage.contracts.dag import TaskDAG
from sage.contracts.planner import TaskPlanner, PlanResult


# ---------------------------------------------------------------------------
# PlanResult
# ---------------------------------------------------------------------------

def test_plan_result_single_task():
    """Simple task produces single-node DAG."""
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Summarize text", "input": {"text": "string"}, "output": {"summary": "string"}},
    ])
    assert isinstance(result, PlanResult)
    assert result.dag is not None
    assert len(result.dag.node_ids) == 1
    assert result.dag.get_node("a").description == "Summarize text"


def test_plan_result_linear_chain():
    """Sequential tasks produce linear DAG with edges."""
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Extract", "output": {"data": "string"}},
        {"id": "b", "description": "Transform", "input": {"data": "string"}, "output": {"result": "string"}, "depends_on": ["a"]},
    ])
    assert len(result.dag.node_ids) == 2
    assert "b" in result.dag.successors("a")


def test_plan_result_diamond():
    """Diamond dependency: A -> B, A -> C, B -> D, C -> D."""
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Start", "output": {"x": "int"}},
        {"id": "b", "description": "Path B", "input": {"x": "int"}, "output": {"y": "int"}, "depends_on": ["a"]},
        {"id": "c", "description": "Path C", "input": {"x": "int"}, "output": {"z": "int"}, "depends_on": ["a"]},
        {"id": "d", "description": "Merge", "input": {"y": "int", "z": "int"}, "depends_on": ["b", "c"]},
    ])
    order = result.dag.topological_sort()
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


# ---------------------------------------------------------------------------
# Node properties from spec
# ---------------------------------------------------------------------------

def test_plan_sets_capabilities():
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Code gen", "capabilities": ["code_generation", "tool_role"]},
    ])
    node = result.dag.get_node("a")
    assert "code_generation" in node.capabilities_required


def test_plan_sets_budget():
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Expensive", "budget_usd": 0.10, "budget_tokens": 8192},
    ])
    node = result.dag.get_node("a")
    assert node.budget.max_cost_usd == 0.10
    assert node.budget.max_tokens == 8192


def test_plan_sets_security_label():
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "PII handler", "security": "HIGH"},
    ])
    node = result.dag.get_node("a")
    assert node.security_label == SecurityLabel.HIGH


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_plan_validates_no_cycles():
    """Planner should detect cycles at plan time."""
    planner = TaskPlanner()
    with pytest.raises(ValueError, match="cycle"):
        planner.plan_static([
            {"id": "a", "description": "A", "depends_on": ["b"]},
            {"id": "b", "description": "B", "depends_on": ["a"]},
        ])


def test_plan_validates_io_compatibility():
    """Planner should warn on IO incompatibility."""
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "A", "output": {"score": "float"}},
        {"id": "b", "description": "B", "input": {"text": "string"}, "depends_on": ["a"]},
    ])
    assert len(result.warnings) > 0
    assert any("text" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

def test_plan_empty_raises():
    planner = TaskPlanner()
    with pytest.raises(ValueError, match="empty"):
        planner.plan_static([])


def test_plan_result_metadata():
    planner = TaskPlanner()
    result = planner.plan_static([
        {"id": "a", "description": "Simple"},
    ])
    assert result.node_count == 1
    assert result.edge_count == 0
