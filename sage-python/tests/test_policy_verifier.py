"""Tests for PolicyVerifier — info-flow, budget, and structural checks."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import (
    TaskNode,
    IOSchema,
    BudgetConstraint,
    SecurityLabel,
)
from sage.contracts.dag import TaskDAG
from sage.contracts.policy import PolicyVerifier, PolicyViolation


# ---------------------------------------------------------------------------
# Info-flow: no HIGH → LOW without declassify
# ---------------------------------------------------------------------------

def test_info_flow_same_level_ok():
    """HIGH → HIGH is fine."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A", security_label=SecurityLabel.HIGH))
    dag.add_node(TaskNode(node_id="b", description="B", security_label=SecurityLabel.HIGH))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert violations == []


def test_info_flow_low_to_high_ok():
    """LOW → HIGH is fine (data can flow up)."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A", security_label=SecurityLabel.LOW))
    dag.add_node(TaskNode(node_id="b", description="B", security_label=SecurityLabel.HIGH))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert violations == []


def test_info_flow_high_to_low_violation():
    """HIGH → LOW is a violation."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A", security_label=SecurityLabel.HIGH))
    dag.add_node(TaskNode(node_id="b", description="B", security_label=SecurityLabel.LOW))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert len(violations) == 1
    assert violations[0].rule == "info_flow"
    assert "a" in violations[0].message and "b" in violations[0].message


def test_info_flow_top_to_medium_violation():
    """TOP → MEDIUM is a violation."""
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A", security_label=SecurityLabel.TOP))
    dag.add_node(TaskNode(node_id="b", description="B", security_label=SecurityLabel.MEDIUM))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag)
    violations = pv.check_info_flow()
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Budget distribution
# ---------------------------------------------------------------------------

def test_budget_within_total():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        budget=BudgetConstraint(max_cost_usd=0.03),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        budget=BudgetConstraint(max_cost_usd=0.04),
    ))
    pv = PolicyVerifier(dag, total_budget_usd=0.10)
    violations = pv.check_budget()
    assert violations == []


def test_budget_exceeds_total():
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        budget=BudgetConstraint(max_cost_usd=0.06),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        budget=BudgetConstraint(max_cost_usd=0.06),
    ))
    pv = PolicyVerifier(dag, total_budget_usd=0.10)
    violations = pv.check_budget()
    assert len(violations) == 1
    assert violations[0].rule == "budget"


def test_budget_no_limit_passes():
    """No total_budget_usd set → no budget check."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        budget=BudgetConstraint(max_cost_usd=999.0),
    ))
    pv = PolicyVerifier(dag)
    violations = pv.check_budget()
    assert violations == []


# ---------------------------------------------------------------------------
# Fan-in / fan-out limits
# ---------------------------------------------------------------------------

def test_fan_out_within_limit():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="root", description="Root"))
    for i in range(3):
        dag.add_node(TaskNode(node_id=f"c{i}", description=f"Child {i}"))
        dag.add_edge("root", f"c{i}")
    pv = PolicyVerifier(dag, max_fan_out=5)
    violations = pv.check_fan_limits()
    assert violations == []


def test_fan_out_exceeded():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="root", description="Root"))
    for i in range(6):
        dag.add_node(TaskNode(node_id=f"c{i}", description=f"Child {i}"))
        dag.add_edge("root", f"c{i}")
    pv = PolicyVerifier(dag, max_fan_out=3)
    violations = pv.check_fan_limits()
    assert len(violations) == 1
    assert violations[0].rule == "fan_out"


def test_fan_in_exceeded():
    dag = TaskDAG()
    for i in range(6):
        dag.add_node(TaskNode(node_id=f"p{i}", description=f"Parent {i}"))
        dag.add_edge(f"p{i}", "sink") if "sink" in [n for n in dag.node_ids] else None
    # Build properly
    dag2 = TaskDAG()
    dag2.add_node(TaskNode(node_id="sink", description="Sink"))
    for i in range(6):
        dag2.add_node(TaskNode(node_id=f"p{i}", description=f"Parent {i}"))
        dag2.add_edge(f"p{i}", "sink")
    pv = PolicyVerifier(dag2, max_fan_in=3)
    violations = pv.check_fan_limits()
    assert len(violations) == 1
    assert violations[0].rule == "fan_in"


# ---------------------------------------------------------------------------
# Full verify (all checks combined)
# ---------------------------------------------------------------------------

def test_verify_all_passes():
    dag = TaskDAG()
    dag.add_node(TaskNode(node_id="a", description="A"))
    dag.add_node(TaskNode(node_id="b", description="B"))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag)
    violations = pv.verify_all()
    assert violations == []


def test_verify_all_collects_multiple():
    """Multiple violations from different rules."""
    dag = TaskDAG()
    dag.add_node(TaskNode(
        node_id="a", description="A",
        security_label=SecurityLabel.TOP,
        budget=BudgetConstraint(max_cost_usd=1.0),
    ))
    dag.add_node(TaskNode(
        node_id="b", description="B",
        security_label=SecurityLabel.LOW,
        budget=BudgetConstraint(max_cost_usd=1.0),
    ))
    dag.add_edge("a", "b")
    pv = PolicyVerifier(dag, total_budget_usd=0.50)
    violations = pv.verify_all()
    rules = {v.rule for v in violations}
    assert "info_flow" in rules
    assert "budget" in rules
