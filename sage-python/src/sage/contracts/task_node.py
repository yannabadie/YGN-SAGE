"""TaskNode IR — VeriMAP-inspired contract for each subtask.

Each TaskNode declares:
- What it needs (input schema, capabilities)
- What it produces (output schema, provenance)
- What it's allowed to do (read/write sets, side effects)
- How to handle failure (retry, replan, compensation)
- Resource limits (tokens, cost, wall time)
- Security classification (info-flow labels)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class SecurityLabel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    TOP = 3


@dataclass
class IOSchema:
    """Named typed fields for input/output contracts."""

    fields: dict[str, str] = field(default_factory=dict)

    def validate(self, data: dict[str, Any]) -> bool:
        """Check that all required fields are present in data."""
        return all(k in data for k in self.fields)


@dataclass
class BudgetConstraint:
    max_tokens: int = 0
    max_cost_usd: float = 0.0
    max_wall_time_s: float = 0.0


@dataclass
class FailurePolicy:
    max_retries: int = 1
    replan_on_failure: bool = False
    compensation: str = ""


@dataclass
class TaskNode:
    """A single node in a task DAG with full contract."""

    node_id: str
    description: str

    # I/O contracts
    input_schema: IOSchema = field(default_factory=IOSchema)
    output_schema: IOSchema = field(default_factory=IOSchema)

    # Capability requirements
    capabilities_required: list[str] = field(default_factory=list)

    # Side-effect permissions
    read_set: list[str] = field(default_factory=list)
    write_set: list[str] = field(default_factory=list)
    idempotent: bool = True

    # Security
    security_label: SecurityLabel = SecurityLabel.LOW

    # Failure handling
    failure_policy: FailurePolicy = field(default_factory=FailurePolicy)

    # Budget
    budget: BudgetConstraint = field(default_factory=BudgetConstraint)

    # Provenance
    provenance_tags: list[str] = field(default_factory=list)
