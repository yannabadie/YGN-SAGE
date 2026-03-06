"""Counterexample-guided repair with hard fences.

When a VF fails, the RepairLoop:
1. Generates a RepairAction (retry with constraint, escalate, or abort)
2. Feeds the counterexample back as a constraint to the runner
3. Hard fences: after max_retries, escalate; after 2x max_retries, abort

Inspired by counterexample-guided abstraction refinement (CEGAR).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sage.contracts.dag import TaskDAG
from sage.contracts.verification import pre_check, post_check, VFResult

log = logging.getLogger(__name__)

Runner = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class RepairAction:
    """A repair action derived from a VF failure."""

    action_type: str  # "retry", "escalate", "abort"
    node_id: str
    constraint: str  # The counterexample/constraint to feed back
    attempt: int = 0

    @staticmethod
    def from_failure(
        node_id: str,
        vf_result: VFResult,
        attempt: int,
        max_retries: int,
    ) -> RepairAction:
        """Create a repair action from a VF failure."""
        if attempt >= max_retries * 2:
            return RepairAction(
                action_type="abort",
                node_id=node_id,
                constraint=vf_result.message,
                attempt=attempt,
            )
        if attempt >= max_retries:
            return RepairAction(
                action_type="escalate",
                node_id=node_id,
                constraint=vf_result.message,
                attempt=attempt,
            )
        return RepairAction(
            action_type="retry",
            node_id=node_id,
            constraint=vf_result.message,
            attempt=attempt,
        )


@dataclass
class RepairResult:
    """Result of a repair loop execution."""

    success: bool = False
    node_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    repair_actions: list[RepairAction] = field(default_factory=list)
    total_attempts: int = 0


class RepairLoop:
    """Execute a TaskDAG with automatic repair on VF failures.

    Parameters
    ----------
    dag:
        The task DAG to execute.
    runner:
        Async callable (node_id, description, input_data) -> output_dict.
    max_retries:
        Max retries per node before escalation.
    """

    def __init__(
        self,
        dag: TaskDAG,
        runner: Runner,
        max_retries: int = 3,
    ) -> None:
        self.dag = dag
        self.runner = runner
        self.max_retries = max_retries

    async def execute(self, initial_data: dict[str, Any]) -> RepairResult:
        """Execute the DAG with repair loop."""
        result = RepairResult()

        try:
            order = self.dag.topological_sort()
        except Exception as e:
            result.success = False
            result.repair_actions.append(RepairAction(
                action_type="abort", node_id="_dag", constraint=str(e),
            ))
            return result

        node_outputs: dict[str, dict[str, Any]] = {}

        for node_id in order:
            node = self.dag.get_node(node_id)

            # Build input from predecessors
            input_data = dict(initial_data)
            for pred_id in self.dag.predecessors(node_id):
                if pred_id in node_outputs:
                    input_data.update(node_outputs[pred_id])

            # Pre-check
            pre_result = pre_check(node, input_data)
            if not pre_result.passed:
                result.success = False
                result.repair_actions.append(RepairAction(
                    action_type="abort", node_id=node_id,
                    constraint=pre_result.message,
                ))
                return result

            # Execute with retry loop
            success = False
            for attempt in range(1, self.max_retries * 2 + 1):
                result.total_attempts += 1

                try:
                    output = await self.runner(node_id, node.description, input_data)
                except Exception as e:
                    action = RepairAction.from_failure(
                        node_id=node_id,
                        vf_result=VFResult(passed=False, message=str(e)),
                        attempt=attempt,
                        max_retries=self.max_retries,
                    )
                    result.repair_actions.append(action)
                    if action.action_type == "abort":
                        result.success = False
                        return result
                    continue

                # Post-check
                actual_cost = output.pop("_cost_usd", 0.0) if isinstance(output, dict) else 0.0
                post_result = post_check(node, output, actual_cost_usd=actual_cost)

                if post_result.passed:
                    node_outputs[node_id] = output
                    result.node_results[node_id] = output
                    success = True
                    break

                # Failed — create repair action
                action = RepairAction.from_failure(
                    node_id=node_id,
                    vf_result=post_result,
                    attempt=attempt,
                    max_retries=self.max_retries,
                )
                result.repair_actions.append(action)

                if action.action_type == "abort":
                    result.success = False
                    return result

                # Inject constraint into input for retry
                input_data["_repair_constraint"] = action.constraint
                log.debug(
                    "Repair attempt %d for %s: %s",
                    attempt, node_id, action.constraint,
                )

            if not success:
                result.success = False
                return result

        result.success = True
        return result
