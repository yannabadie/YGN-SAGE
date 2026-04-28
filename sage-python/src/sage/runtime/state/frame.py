from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class EvidenceRef:
    kind: Literal["tool", "node", "memory", "file", "verifier"]
    id: str
    hash: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class StateDelta:
    add_constraints: tuple[str, ...] = ()
    remove_constraints: tuple[str, ...] = ()
    add_assumptions: tuple[str, ...] = ()
    invalidate_assumptions: tuple[str, ...] = ()
    update_entities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    add_decisions: tuple[Mapping[str, Any], ...] = ()
    add_tool_facts: tuple[Mapping[str, Any], ...] = ()
    add_open_questions: tuple[str, ...] = ()
    close_open_questions: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.add_constraints
            or self.remove_constraints
            or self.add_assumptions
            or self.invalidate_assumptions
            or self.update_entities
            or self.add_decisions
            or self.add_tool_facts
            or self.add_open_questions
            or self.close_open_questions
            or self.evidence
        )


@dataclass(frozen=True)
class StateFrame:
    task_id: str
    version: int = 0
    objective: str = ""
    constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    invalidated_assumptions: tuple[str, ...] = ()
    entities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    decisions: tuple[Mapping[str, Any], ...] = ()
    tool_facts: tuple[Mapping[str, Any], ...] = ()
    open_questions: tuple[str, ...] = ()
    causal_edges: tuple[tuple[str, str, str], ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class StateApplyResult:
    before_version: int
    after: StateFrame
    conflicts: tuple[str, ...] = ()
    applied: bool = True
