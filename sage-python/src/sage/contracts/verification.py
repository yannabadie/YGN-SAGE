"""Verification Functions (VFs) for TaskNode contracts.

Each TaskNode can have pre-checks (before execution) and post-checks
(after execution). VFs return VFResult with pass/fail + evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from sage.contracts.task_node import TaskNode


@dataclass
class VFResult:
    """Result of a single verification function."""

    passed: bool
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationFn:
    """A named verification function bound to a phase."""

    name: str
    phase: str  # "pre" or "post"
    fn: Callable[[TaskNode, dict], VFResult] = field(repr=False)


def pre_check(
    node: TaskNode,
    data: dict[str, Any],
    *,
    extra_vfs: list[VerificationFn] | None = None,
) -> VFResult:
    """Run pre-execution checks: input schema + custom VFs."""
    # 1. Built-in: validate input schema
    if node.input_schema.fields and not node.input_schema.validate(data):
        missing = [k for k in node.input_schema.fields if k not in data]
        return VFResult(
            passed=False,
            message=f"Input schema validation failed: missing {missing}",
            evidence={"missing_fields": missing},
        )

    # 2. Run extra VFs for "pre" phase
    if extra_vfs:
        for vf in extra_vfs:
            if vf.phase != "pre":
                continue
            result = vf.fn(node, data)
            if not result.passed:
                return result

    return VFResult(passed=True, message="pre_check passed")


def post_check(
    node: TaskNode,
    data: dict[str, Any],
    *,
    actual_cost_usd: float = 0.0,
    extra_vfs: list[VerificationFn] | None = None,
) -> VFResult:
    """Run post-execution checks: output schema + budget + custom VFs."""
    # 1. Built-in: validate output schema
    if node.output_schema.fields and not node.output_schema.validate(data):
        missing = [k for k in node.output_schema.fields if k not in data]
        return VFResult(
            passed=False,
            message=f"Output schema validation failed: missing {missing}",
            evidence={"missing_fields": missing},
        )

    # 2. Built-in: budget check
    if node.budget.max_cost_usd > 0 and actual_cost_usd > node.budget.max_cost_usd:
        return VFResult(
            passed=False,
            message=f"Budget exceeded: ${actual_cost_usd:.4f} > ${node.budget.max_cost_usd:.4f}",
            evidence={"actual": actual_cost_usd, "limit": node.budget.max_cost_usd},
        )

    # 3. Run extra VFs for "post" phase
    if extra_vfs:
        for vf in extra_vfs:
            if vf.phase != "post":
                continue
            result = vf.fn(node, data)
            if not result.passed:
                return result

    return VFResult(passed=True, message="post_check passed")


def run_verification(
    node: TaskNode,
    phase: str,
    data: dict[str, Any],
    *,
    extra_vfs: list[VerificationFn] | None = None,
    actual_cost_usd: float = 0.0,
) -> list[VFResult]:
    """Run all VFs for a given phase, collecting all results."""
    results: list[VFResult] = []

    if phase == "pre":
        results.append(pre_check(node, data, extra_vfs=extra_vfs))
    elif phase == "post":
        results.append(
            post_check(node, data, actual_cost_usd=actual_cost_usd, extra_vfs=extra_vfs)
        )

    return results
