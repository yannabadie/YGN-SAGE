"""Tests for Verification Functions (VFs) on TaskNode."""
from __future__ import annotations

import pytest
from sage.contracts.task_node import TaskNode, IOSchema, BudgetConstraint
from sage.contracts.verification import (
    VFResult,
    VerificationFn,
    pre_check,
    post_check,
    run_verification,
)


# ---------------------------------------------------------------------------
# VFResult
# ---------------------------------------------------------------------------

def test_vf_result_pass():
    r = VFResult(passed=True, message="ok")
    assert r.passed is True
    assert r.message == "ok"


def test_vf_result_fail():
    r = VFResult(passed=False, message="missing field", evidence={"field": "x"})
    assert r.passed is False
    assert r.evidence == {"field": "x"}


# ---------------------------------------------------------------------------
# pre_check: runs BEFORE task execution
# ---------------------------------------------------------------------------

def test_pre_check_validates_input_schema():
    """pre_check should reject data that doesn't match input_schema."""
    node = TaskNode(
        node_id="t1",
        description="needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    )
    result = pre_check(node, data={})
    assert result.passed is False
    assert "text" in result.message


def test_pre_check_passes_valid_input():
    node = TaskNode(
        node_id="t1",
        description="needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    )
    result = pre_check(node, data={"text": "hello"})
    assert result.passed is True


def test_pre_check_with_custom_vf():
    """Custom verification function added to node should run in pre_check."""
    def no_empty_text(node: TaskNode, data: dict) -> VFResult:
        if not data.get("text"):
            return VFResult(passed=False, message="text must not be empty")
        return VFResult(passed=True, message="ok")

    vf = VerificationFn(name="no_empty_text", phase="pre", fn=no_empty_text)
    node = TaskNode(
        node_id="t1",
        description="needs text",
        input_schema=IOSchema(fields={"text": "string"}),
    )

    # Valid schema but empty text — custom VF should catch it
    result = pre_check(node, data={"text": ""}, extra_vfs=[vf])
    assert result.passed is False
    assert "empty" in result.message


# ---------------------------------------------------------------------------
# post_check: runs AFTER task execution
# ---------------------------------------------------------------------------

def test_post_check_validates_output_schema():
    node = TaskNode(
        node_id="t1",
        description="produces summary",
        output_schema=IOSchema(fields={"summary": "string"}),
    )
    result = post_check(node, data={})
    assert result.passed is False
    assert "summary" in result.message


def test_post_check_passes_valid_output():
    node = TaskNode(
        node_id="t1",
        description="produces summary",
        output_schema=IOSchema(fields={"summary": "string"}),
    )
    result = post_check(node, data={"summary": "done"})
    assert result.passed is True


def test_post_check_budget_exceeded():
    """post_check should fail if cost exceeds budget."""
    node = TaskNode(
        node_id="t1",
        description="expensive",
        budget=BudgetConstraint(max_cost_usd=0.05),
    )
    result = post_check(node, data={}, actual_cost_usd=0.10)
    assert result.passed is False
    assert "cost" in result.message.lower() or "budget" in result.message.lower()


# ---------------------------------------------------------------------------
# run_verification: orchestrates multiple VFs
# ---------------------------------------------------------------------------

def test_run_verification_all_pass():
    node = TaskNode(
        node_id="t1",
        description="simple",
        input_schema=IOSchema(fields={"x": "int"}),
    )
    results = run_verification(node, phase="pre", data={"x": 42})
    assert all(r.passed for r in results)


def test_run_verification_collects_failures():
    def always_fail(node: TaskNode, data: dict) -> VFResult:
        return VFResult(passed=False, message="forced fail")

    vf = VerificationFn(name="always_fail", phase="pre", fn=always_fail)
    node = TaskNode(node_id="t1", description="simple")
    results = run_verification(node, phase="pre", data={}, extra_vfs=[vf])
    assert any(not r.passed for r in results)


def test_run_verification_skips_wrong_phase():
    """A VF registered for 'post' should not run during 'pre' phase."""
    def post_only(node: TaskNode, data: dict) -> VFResult:
        return VFResult(passed=False, message="should not run in pre")

    vf = VerificationFn(name="post_only", phase="post", fn=post_only)
    node = TaskNode(node_id="t1", description="simple")
    results = run_verification(node, phase="pre", data={}, extra_vfs=[vf])
    # The post-only VF should not have been called, so no failures from it
    assert all(r.passed for r in results)
