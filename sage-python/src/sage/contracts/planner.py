"""TaskPlanner — Plan-and-Act inspired task decomposition.

Takes a task specification (list of step dicts) and produces a verified
TaskDAG with TaskNodes. Separates planning from execution: the planner
outputs a DAG, the DAGExecutor runs it.

Supports:
- Static planning: from explicit step specs (plan_static)
- Validation: cycle detection, IO compatibility warnings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sage.contracts.task_node import (
    TaskNode,
    IOSchema,
    BudgetConstraint,
    SecurityLabel,
)
from sage.contracts.dag import TaskDAG, CycleError


_SECURITY_MAP = {
    "LOW": SecurityLabel.LOW,
    "MEDIUM": SecurityLabel.MEDIUM,
    "HIGH": SecurityLabel.HIGH,
    "TOP": SecurityLabel.TOP,
}


@dataclass
class PlanResult:
    """Result of planning: a verified DAG + metadata."""

    dag: TaskDAG
    warnings: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.dag.node_ids)

    @property
    def edge_count(self) -> int:
        count = 0
        for nid in self.dag.node_ids:
            count += len(self.dag.successors(nid))
        return count


class TaskPlanner:
    """Builds a verified TaskDAG from step specifications.

    Each step spec is a dict with keys:
        - id (str): unique node identifier
        - description (str): what this step does
        - input (dict[str, str], optional): input schema fields
        - output (dict[str, str], optional): output schema fields
        - depends_on (list[str], optional): predecessor node IDs
        - capabilities (list[str], optional): required capabilities
        - budget_usd (float, optional): max cost
        - budget_tokens (int, optional): max tokens
        - security (str, optional): LOW/MEDIUM/HIGH/TOP
    """

    def plan_static(self, steps: list[dict[str, Any]]) -> PlanResult:
        """Build a TaskDAG from explicit step specifications."""
        if not steps:
            raise ValueError("Cannot plan from empty step list")

        dag = TaskDAG()

        # 1. Create nodes
        for step in steps:
            node = TaskNode(
                node_id=step["id"],
                description=step.get("description", ""),
                input_schema=IOSchema(fields=step.get("input", {})),
                output_schema=IOSchema(fields=step.get("output", {})),
                capabilities_required=step.get("capabilities", []),
                budget=BudgetConstraint(
                    max_cost_usd=step.get("budget_usd", 0.0),
                    max_tokens=step.get("budget_tokens", 0),
                ),
                security_label=_SECURITY_MAP.get(
                    step.get("security", "LOW"), SecurityLabel.LOW
                ),
            )
            dag.add_node(node)

        # 2. Create edges from depends_on
        for step in steps:
            for dep in step.get("depends_on", []):
                dag.add_edge(dep, step["id"])

        # 3. Validate: cycle detection (raises CycleError)
        try:
            dag.topological_sort()
        except CycleError:
            raise ValueError("Plan contains a cycle — cannot execute")

        # 4. Validate: IO compatibility warnings
        warnings = dag.validate_io_compatibility()

        return PlanResult(dag=dag, warnings=warnings)
