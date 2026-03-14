# RESERVED FOR PHASE C: DAGExecutor — topo execution with VF pre/post checks + policy gate.
# To be integrated into Pipeline Stage 4 (_check_adaptation) for verified retry.
# See: docs/superpowers/specs/2026-03-14-cognitive-orchestration-pipeline-design.md
"""DAGExecutor — contract-driven execution of a TaskDAG.

Executes nodes in topological order, running pre/post VFs at each step.
Policy violations (info-flow, budget) are checked before execution begins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from sage.contracts.dag import TaskDAG
from sage.contracts.verification import pre_check, post_check
from sage.contracts.policy import PolicyVerifier, PolicyViolation
from sage.contracts.cost_tracker import CostTracker

log = logging.getLogger(__name__)

# Type alias for the async runner function
Runner = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class NodeResult:
    """Result of executing a single TaskNode."""

    node_id: str
    output: dict[str, Any] = field(default_factory=dict)
    pre_check_passed: bool = True
    post_check_passed: bool = True
    error: str = ""


@dataclass
class DAGExecutionResult:
    """Result of executing an entire TaskDAG."""

    success: bool = True
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    policy_violations: list[PolicyViolation] = field(default_factory=list)


class DAGExecutor:
    """Execute a TaskDAG with contract verification at each node.

    Parameters
    ----------
    dag:
        The task DAG to execute.
    runner:
        Async callable ``(node_id, description, input_data) -> output_dict``.
        The executor calls this for each node in topological order.
    total_budget_usd:
        Optional total budget for policy verification.
    max_fan_in:
        Optional max fan-in for policy verification.
    max_fan_out:
        Optional max fan-out for policy verification.
    """

    def __init__(
        self,
        dag: TaskDAG,
        runner: Runner,
        *,
        total_budget_usd: float = 0.0,
        max_fan_in: int = 0,
        max_fan_out: int = 0,
    ) -> None:
        self.dag = dag
        self.runner = runner
        self.total_budget_usd = total_budget_usd
        self.max_fan_in = max_fan_in
        self.max_fan_out = max_fan_out
        self.cost_tracker = CostTracker(budget_usd=total_budget_usd)

    async def execute(self, initial_data: dict[str, Any]) -> DAGExecutionResult:
        """Execute the DAG in topological order with VF checks."""
        result = DAGExecutionResult()

        # 1. Policy check before any execution
        pv = PolicyVerifier(
            self.dag,
            total_budget_usd=self.total_budget_usd,
            max_fan_in=self.max_fan_in,
            max_fan_out=self.max_fan_out,
        )
        violations = pv.verify_all()
        if violations:
            result.success = False
            result.policy_violations = violations
            return result

        # 2. Topological execution
        try:
            order = self.dag.topological_sort()
        except Exception as e:
            result.success = False
            result.node_results["_dag"] = NodeResult(
                node_id="_dag", error=str(e),
            )
            return result

        # Track outputs per node for data flow
        node_outputs: dict[str, dict[str, Any]] = {}

        for node_id in order:
            node = self.dag.get_node(node_id)

            # Build input: merge initial_data with predecessor outputs
            input_data = dict(initial_data)
            for pred_id in self.dag.predecessors(node_id):
                if pred_id in node_outputs:
                    input_data.update(node_outputs[pred_id])

            # Pre-check
            pre_result = pre_check(node, input_data)
            if not pre_result.passed:
                nr = NodeResult(
                    node_id=node_id,
                    pre_check_passed=False,
                    error=pre_result.message,
                )
                result.node_results[node_id] = nr
                result.success = False
                return result

            # Execute
            try:
                output = await self.runner(node_id, node.description, input_data)
            except Exception as e:
                nr = NodeResult(node_id=node_id, error=str(e))
                result.node_results[node_id] = nr
                result.success = False
                return result

            # Normalize output to dict
            if not isinstance(output, dict):
                nr = NodeResult(
                    node_id=node_id,
                    error=f"Runner returned {type(output).__name__}, expected dict",
                )
                result.node_results[node_id] = nr
                result.success = False
                return result

            # Post-check
            actual_cost = output.pop("_cost_usd", 0.0)
            post_result = post_check(node, output, actual_cost_usd=actual_cost)
            if not post_result.passed:
                nr = NodeResult(
                    node_id=node_id,
                    output=output,
                    post_check_passed=False,
                    error=post_result.message,
                )
                result.node_results[node_id] = nr
                result.success = False
                return result

            # Cumulative cost tracking
            if actual_cost > 0:
                self.cost_tracker.record(node_id, actual_cost)
            if self.cost_tracker.is_over_budget:
                nr = NodeResult(
                    node_id=node_id,
                    output=output,
                    error=f"Budget exceeded: spent {self.cost_tracker.total_spent:.2f} > {self.total_budget_usd:.2f}",
                )
                result.node_results[node_id] = nr
                result.success = False
                return result

            # Success — store output
            node_outputs[node_id] = output
            result.node_results[node_id] = NodeResult(
                node_id=node_id, output=output,
            )

        return result
