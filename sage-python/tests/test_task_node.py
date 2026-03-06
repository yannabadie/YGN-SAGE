"""Tests for TaskNode IR dataclass."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import (
    TaskNode,
    IOSchema,
    FailurePolicy,
    BudgetConstraint,
    SecurityLabel,
)


def test_task_node_creation():
    node = TaskNode(
        node_id="summarize",
        description="Summarize input text",
        input_schema=IOSchema(fields={"text": "string"}),
        output_schema=IOSchema(fields={"summary": "string"}),
    )
    assert node.node_id == "summarize"
    assert node.input_schema.fields["text"] == "string"


def test_task_node_with_capabilities():
    node = TaskNode(
        node_id="search",
        description="Search with file search",
        capabilities_required=["file_search", "grounding"],
    )
    assert "file_search" in node.capabilities_required


def test_task_node_budget_constraint():
    budget = BudgetConstraint(max_tokens=4096, max_cost_usd=0.10, max_wall_time_s=30.0)
    node = TaskNode(
        node_id="expensive",
        description="Expensive task",
        budget=budget,
    )
    assert node.budget.max_cost_usd == 0.10
    assert node.budget.max_tokens == 4096


def test_task_node_failure_policy():
    policy = FailurePolicy(max_retries=3, replan_on_failure=True)
    node = TaskNode(
        node_id="fragile",
        description="May fail",
        failure_policy=policy,
    )
    assert node.failure_policy.max_retries == 3
    assert node.failure_policy.replan_on_failure is True


def test_task_node_security_labels():
    node = TaskNode(
        node_id="sensitive",
        description="Handles PII",
        security_label=SecurityLabel.HIGH,
    )
    assert node.security_label == SecurityLabel.HIGH


def test_task_node_side_effects():
    node = TaskNode(
        node_id="writer",
        description="Writes files",
        read_set=["config.json"],
        write_set=["output.txt"],
        idempotent=False,
    )
    assert "output.txt" in node.write_set
    assert node.idempotent is False


def test_task_node_defaults():
    node = TaskNode(node_id="simple", description="Simple task")
    assert node.capabilities_required == []
    assert node.idempotent is True
    assert node.security_label == SecurityLabel.LOW
    assert node.failure_policy.max_retries == 1
    assert node.read_set == []
    assert node.write_set == []


def test_io_schema_validate_accepts_matching():
    schema = IOSchema(fields={"name": "string", "count": "int"})
    assert schema.validate({"name": "test", "count": 42}) is True


def test_io_schema_validate_rejects_missing():
    schema = IOSchema(fields={"name": "string", "count": "int"})
    assert schema.validate({"name": "test"}) is False
