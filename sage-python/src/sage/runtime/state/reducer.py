from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from sage.runtime.state.errors import StateConflict
from sage.runtime.state.frame import StateApplyResult, StateDelta, StateFrame


@dataclass(frozen=True)
class _NormalizedDelta:
    source_node_id: str | None
    is_empty: bool
    add_constraints: tuple[str, ...]
    remove_constraints: tuple[str, ...]
    add_assumptions: tuple[str, ...]
    invalidate_assumptions: tuple[str, ...]
    update_entities: dict[str, dict[str, Any]]
    add_decisions: tuple[dict[str, Any], ...]
    add_tool_facts: tuple[dict[str, Any], ...]
    add_open_questions: tuple[str, ...]
    close_open_questions: tuple[str, ...]
    has_evidence: bool


def normalize_assumption_id(value: str) -> str:
    """Return the R6 canonical assumption id form."""
    return re.sub(r"[\s-]+", "_", str(value).strip().lower())


def apply_delta(
    frame: StateFrame,
    delta: StateDelta,
    *,
    source_node_id: str | None = None,
    raise_on_conflict: bool = False,
) -> StateApplyResult:
    """Apply one state delta without mutating the input frame."""
    return apply_deltas(
        frame,
        ((source_node_id or "", delta),),
        raise_on_conflict=raise_on_conflict,
    )


def apply_deltas(
    frame: StateFrame,
    deltas: tuple[tuple[str, StateDelta], ...],
    *,
    raise_on_conflict: bool = False,
) -> StateApplyResult:
    """Atomically apply a sibling batch of deltas sorted by source node id."""
    before_version = frame.version
    ordered = sorted(
        enumerate(deltas),
        key=lambda item: (str(item[1][0]), item[0]),
    )
    normalized = tuple(
        _normalize_delta(delta, source_node_id=str(source_node_id) or None)
        for _idx, (source_node_id, delta) in ordered
    )
    conflicts = _validate_batch(normalized)
    if conflicts:
        result = StateApplyResult(
            before_version=before_version,
            after=frame,
            conflicts=tuple(conflicts),
            applied=False,
        )
        if raise_on_conflict:
            raise StateConflict("; ".join(result.conflicts))
        return result

    if not normalized or all(delta.is_empty for delta in normalized):
        return StateApplyResult(before_version=before_version, after=frame, applied=True)

    constraints = list(frame.constraints)
    assumptions = list(frame.assumptions)
    invalidated_assumptions = list(frame.invalidated_assumptions)
    entities = _copy_entities(frame.entities)
    decisions = list(_copy_mapping_tuple(frame.decisions))
    tool_facts = list(_copy_mapping_tuple(frame.tool_facts))
    open_questions = list(frame.open_questions)

    accepted_non_empty = 0
    for delta in normalized:
        if delta.is_empty:
            continue
        accepted_non_empty += 1
        for constraint in delta.add_constraints:
            if constraint not in constraints:
                constraints.append(constraint)
        if delta.remove_constraints:
            remove_set = set(delta.remove_constraints)
            constraints = [value for value in constraints if value not in remove_set]

        for assumption_id in delta.add_assumptions:
            if assumption_id not in assumptions and assumption_id not in invalidated_assumptions:
                assumptions.append(assumption_id)
        for assumption_id in delta.invalidate_assumptions:
            assumptions = [value for value in assumptions if value != assumption_id]
            if assumption_id not in invalidated_assumptions:
                invalidated_assumptions.append(assumption_id)

        for entity_id, fields in delta.update_entities.items():
            existing = dict(entities.get(entity_id, {}))
            existing.update(fields)
            entities[entity_id] = existing

        decisions.extend(dict(item) for item in delta.add_decisions)
        tool_facts.extend(dict(item) for item in delta.add_tool_facts)

        for question in delta.add_open_questions:
            if question not in open_questions:
                open_questions.append(question)
        if delta.close_open_questions:
            close_set = set(delta.close_open_questions)
            open_questions = [value for value in open_questions if value not in close_set]

    after = StateFrame(
        task_id=frame.task_id,
        version=before_version + accepted_non_empty,
        objective=frame.objective,
        constraints=tuple(constraints),
        assumptions=tuple(assumptions),
        invalidated_assumptions=tuple(invalidated_assumptions),
        entities=entities,
        decisions=tuple(decisions),
        tool_facts=tuple(tool_facts),
        open_questions=tuple(open_questions),
        causal_edges=(),
        confidence=frame.confidence,
    )
    return StateApplyResult(before_version=before_version, after=after, applied=True)


def _normalize_delta(delta: StateDelta, *, source_node_id: str | None) -> _NormalizedDelta:
    return _NormalizedDelta(
        source_node_id=source_node_id,
        is_empty=delta.is_empty(),
        add_constraints=tuple(str(value) for value in delta.add_constraints),
        remove_constraints=tuple(str(value) for value in delta.remove_constraints),
        add_assumptions=tuple(
            normalize_assumption_id(value) for value in delta.add_assumptions
        ),
        invalidate_assumptions=tuple(
            normalize_assumption_id(value) for value in delta.invalidate_assumptions
        ),
        update_entities={
            str(entity_id): dict(fields)
            for entity_id, fields in delta.update_entities.items()
        },
        add_decisions=tuple(dict(value) for value in delta.add_decisions),
        add_tool_facts=tuple(dict(value) for value in delta.add_tool_facts),
        add_open_questions=tuple(str(value) for value in delta.add_open_questions),
        close_open_questions=tuple(str(value) for value in delta.close_open_questions),
        has_evidence=bool(delta.evidence),
    )


def _validate_batch(deltas: tuple[_NormalizedDelta, ...]) -> list[str]:
    conflicts: list[str] = []
    added_constraints: set[str] = set()
    removed_constraints: set[str] = set()
    added_assumptions: set[str] = set()
    invalidated_assumptions: set[str] = set()
    entity_updates: dict[tuple[str, str], Any] = {}

    for delta in deltas:
        added_constraints.update(delta.add_constraints)
        removed_constraints.update(delta.remove_constraints)
        added_assumptions.update(delta.add_assumptions)
        invalidated_assumptions.update(delta.invalidate_assumptions)

        for entity_id, fields in delta.update_entities.items():
            for field_name, value in fields.items():
                key = (entity_id, str(field_name))
                if key in entity_updates and entity_updates[key] != value:
                    conflicts.append(
                        f"entity field conflict: {entity_id}.{field_name}"
                    )
                else:
                    entity_updates[key] = value

        for decision in delta.add_decisions:
            if (
                not delta.source_node_id
                and not delta.has_evidence
                and not _mapping_has_evidence(decision)
            ):
                conflicts.append("decision without evidence or source_node_id")

        for tool_fact in delta.add_tool_facts:
            if (
                not tool_fact.get("tool_call_id")
                and not tool_fact.get("node_id")
                and not delta.source_node_id
            ):
                conflicts.append("tool_fact without tool_call_id or node attribution")

    for constraint in sorted(added_constraints & removed_constraints):
        conflicts.append(f"constraint added and removed in batch: {constraint}")
    for assumption_id in sorted(added_assumptions & invalidated_assumptions):
        conflicts.append(f"assumption added and invalidated in batch: {assumption_id}")
    return conflicts


def _mapping_has_evidence(value: Mapping[str, Any]) -> bool:
    evidence = value.get("evidence") or value.get("evidence_ref") or value.get("evidence_refs")
    return bool(evidence)


def _copy_entities(
    entities: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(entity_id): dict(fields) for entity_id, fields in entities.items()}


def _copy_mapping_tuple(
    values: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(value) for value in values)
