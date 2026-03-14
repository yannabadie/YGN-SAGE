"""TaskPlanner — Plan-and-Act inspired task decomposition.

Takes a task specification (list of step dicts) and produces a verified
TaskDAG with TaskNodes. Separates planning from execution: the planner
outputs a DAG, the DAGExecutor runs it.

Supports:
- Static planning: from explicit step specs (plan_static)
- Auto planning: LLM-driven decomposition (plan_auto)
- Validation: cycle detection, IO compatibility warnings
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sage.contracts.task_node import (
    TaskNode,
    IOSchema,
    BudgetConstraint,
    SecurityLabel,
)
from sage.contracts.dag import TaskDAG, CycleError

MAX_DECOMPOSITION_STEPS = 6


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

    async def plan_auto(self, task: str, provider: Any) -> PlanResult:
        """LLM-driven task decomposition into verified TaskDAG.

        Prompts provider to output JSON array of step objects:
        [{"id": "step1", "description": "...", "depends_on": ["step0"]}]

        Hard-cap: MAX_DECOMPOSITION_STEPS = 6. Truncates beyond that.
        Falls back to single-node DAG on any failure.
        """
        from sage.llm.base import Message, Role, LLMConfig

        log = logging.getLogger(__name__)

        prompt = (
            "Decompose the following task into 2-6 sequential or parallel subtasks. "
            "Output ONLY a JSON array. Each element: "
            '{"id": "unique_id", "description": "what to do", "depends_on": ["ids of prerequisites"]}. '
            "Keep it minimal — prefer fewer steps. "
            f"Task: {task}"
        )

        try:
            config = LLMConfig(provider="google", model="gemini-2.5-flash")
            response = await provider.generate(
                messages=[
                    Message(role=Role.SYSTEM, content="You are a task decomposition assistant. Output only valid JSON."),
                    Message(role=Role.USER, content=prompt),
                ],
                config=config,
            )
            content = response.content or ""

            # Extract JSON from response (may have markdown fences)
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in LLM response")

            steps = json.loads(json_match.group())
            if not isinstance(steps, list):
                raise ValueError("Expected JSON array")

            # Hard cap
            steps = steps[:MAX_DECOMPOSITION_STEPS]

            # Normalize step format for plan_static
            normalized = []
            for step in steps:
                if not isinstance(step, dict) or "id" not in step:
                    continue
                entry: dict[str, Any] = {
                    "id": str(step["id"]),
                    "description": str(step.get("description", step["id"])),
                }
                deps = step.get("depends_on", [])
                if deps:
                    entry["depends_on"] = [str(d) for d in deps]
                normalized.append(entry)

            if not normalized:
                raise ValueError("No valid steps after normalization")

            return self.plan_static(normalized)

        except Exception as exc:
            log.warning("plan_auto fallback to single-node DAG: %s", exc)
            # Fallback: single-node DAG with the entire task
            return self.plan_static([{"id": "main", "description": task}])
