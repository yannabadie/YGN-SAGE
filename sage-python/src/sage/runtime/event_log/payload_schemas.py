"""RuntimeEventLog payload schema source of truth.

The envelope schema version remains ``SCHEMA_VERSION == "1.0"``. These
records version the per-event payload contract carried by
``payload_schema_version``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Literal, TypeAlias, cast

from sage.runtime.event_log.errors import EventLogSchemaError
from sage.runtime.event_log.schema import (
    CONTROLLER_ACTIONS,
    EVENT_TYPES,
    FINAL_RESULT_STATUSES,
)
from sage.runtime.oracle.verdict import QUALITY_LABELS, VERDICT_SOURCES

PayloadSchemaVersion: TypeAlias = str
PayloadKind: TypeAlias = Literal["dict", "scalar", "none"]
PayloadJsonType: TypeAlias = Literal[
    "str",
    "int",
    "float",
    "bool",
    "null",
    "list",
    "dict",
    "any_json",
]
PayloadJsonTypeSpec: TypeAlias = PayloadJsonType | tuple[PayloadJsonType, ...]
PayloadSchemaMode: TypeAlias = Literal["audit", "strict-current"]

DEFAULT_PAYLOAD_STRING_MAX_BYTES = 4096
DEFAULT_PAYLOAD_JSON_MAX_BYTES = 4096

_PAYLOAD_SCHEMA_VERSION_RE = re.compile(
    r"^v[1-9][0-9]*(?:_[a-z0-9][a-z0-9_:-]*)?$"
)


@dataclass(frozen=True, slots=True)
class PayloadFieldSpec:
    json_type: PayloadJsonTypeSpec
    required: bool = False
    max_utf8_bytes: int | None = None
    max_json_utf8_bytes: int | None = None
    allowed_values: tuple[Any, ...] | None = None
    item_type: PayloadJsonTypeSpec | None = None
    item_max_utf8_bytes: int | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class EventPayloadSchema:
    event_type: str
    version: PayloadSchemaVersion
    allowed_fields: tuple[str, ...]
    required_fields: tuple[str, ...]
    field_specs: Mapping[str, PayloadFieldSpec]
    payload_kind: PayloadKind
    current: bool
    legacy_read_only: bool
    scalar_alias: str | None = None
    conditional_rules: tuple[str, ...] = ()
    top_level_field_specs: Mapping[str, PayloadFieldSpec] | None = None


PayloadSchemaDistributionEntry: TypeAlias = dict[str, int | str]


@dataclass(frozen=True, slots=True)
class _PayloadSchemaDistributionReport:
    mode: PayloadSchemaMode
    distribution: dict[str, dict[str, PayloadSchemaDistributionEntry]]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _f(
    json_type: PayloadJsonTypeSpec,
    *,
    required: bool = False,
    max_utf8_bytes: int | None = None,
    max_json_utf8_bytes: int | None = None,
    allowed_values: Sequence[Any] | None = None,
    item_type: PayloadJsonTypeSpec | None = None,
    item_max_utf8_bytes: int | None = None,
    notes: str = "",
) -> PayloadFieldSpec:
    return PayloadFieldSpec(
        json_type=json_type,
        required=required,
        max_utf8_bytes=max_utf8_bytes,
        max_json_utf8_bytes=max_json_utf8_bytes,
        allowed_values=tuple(allowed_values) if allowed_values is not None else None,
        item_type=item_type,
        item_max_utf8_bytes=item_max_utf8_bytes,
        notes=notes,
    )


def _schema(
    *,
    event_type: str,
    version: str,
    allowed_fields: Sequence[str],
    required_fields: Sequence[str],
    field_specs: Mapping[str, PayloadFieldSpec],
    payload_kind: PayloadKind,
    current: bool,
    legacy_read_only: bool = False,
    scalar_alias: str | None = None,
    conditional_rules: Sequence[str] = (),
    top_level_field_specs: Mapping[str, PayloadFieldSpec] | None = None,
) -> EventPayloadSchema:
    required = tuple(required_fields)
    return EventPayloadSchema(
        event_type=event_type,
        version=version,
        allowed_fields=tuple(allowed_fields),
        required_fields=required,
        field_specs={
            key: _with_required(spec, key in required)
            for key, spec in field_specs.items()
        },
        payload_kind=payload_kind,
        current=current,
        legacy_read_only=legacy_read_only,
        scalar_alias=scalar_alias,
        conditional_rules=tuple(conditional_rules),
        top_level_field_specs=top_level_field_specs,
    )


def _with_required(spec: PayloadFieldSpec, required: bool) -> PayloadFieldSpec:
    return PayloadFieldSpec(
        json_type=spec.json_type,
        required=required,
        max_utf8_bytes=spec.max_utf8_bytes,
        max_json_utf8_bytes=spec.max_json_utf8_bytes,
        allowed_values=spec.allowed_values,
        item_type=spec.item_type,
        item_max_utf8_bytes=spec.item_max_utf8_bytes,
        notes=spec.notes,
    )


def _string(max_bytes: int | None = None) -> PayloadFieldSpec:
    return _f("str", max_utf8_bytes=max_bytes)


def _string_or_null(max_bytes: int | None = None) -> PayloadFieldSpec:
    return _f(("str", "null"), max_utf8_bytes=max_bytes)


def _int() -> PayloadFieldSpec:
    return _f("int")


def _float_or_null() -> PayloadFieldSpec:
    return _f(("float", "null"))


def _bool() -> PayloadFieldSpec:
    return _f("bool")


def _list(
    *,
    max_json_utf8_bytes: int | None = None,
    item_type: PayloadJsonTypeSpec | None = None,
    item_max_utf8_bytes: int | None = None,
) -> PayloadFieldSpec:
    return _f(
        "list",
        max_json_utf8_bytes=max_json_utf8_bytes,
        item_type=item_type,
        item_max_utf8_bytes=item_max_utf8_bytes,
    )


def _dict_or_null(max_json_utf8_bytes: int | None = None) -> PayloadFieldSpec:
    return _f(("dict", "null"), max_json_utf8_bytes=max_json_utf8_bytes)


def _any_json(max_json_utf8_bytes: int | None = None) -> PayloadFieldSpec:
    return _f("any_json", max_json_utf8_bytes=max_json_utf8_bytes)


_CONTROLLER_CURRENT_FIELDS = (
    "node_id",
    "action",
    "target_node_id",
    "gate_source_id",
    "gate_target_id",
    "quality_score",
    "quality_source",
    "threshold_band",
    "reason_code",
)

_NODE_STARTED_FIELDS = (
    "topology_id",
    "node_id",
    "node_role",
    "attempt",
    "model_id",
    "provider_id",
    "predecessor_ids",
    "edge_ids",
    "predecessors_by_channel",
)

_RUN_FRAME_SUMMARY_FIELDS = (
    "run_frame_schema_version",
    "run_id",
    "task_id",
    "task_hash",
    "status",
    "topology_id",
    "graph_digest",
    "topology_history",
    "node_record_count",
    "node_records",
    "state_frame_count",
    "controller_decision_count",
    "runtime_delta_count",
    "runtime_delta_hashes",
    "final_result_seq",
    "failure_seqs",
    "terminal_failure_seq",
    "budget_snapshot",
    "feature_flags",
    "redacted",
    "oracle_verdict",
    "run_frame_hash",
)

_EVIDENCE_REF_FIELDS = frozenset(
    {
        "run_id",
        "node_run_id",
        "event_seq",
        "output_sha256",
        "tool_call_id",
        "verifier_id",
        "evidence_hash",
    }
)


PAYLOAD_SCHEMAS: dict[str, dict[PayloadSchemaVersion, EventPayloadSchema]] = {
    "task_started": {
        "v1": _schema(
            event_type="task_started",
            version="v1",
            allowed_fields=("task_text",),
            required_fields=("task_text",),
            field_specs={"task_text": _string(16384)},
            payload_kind="scalar",
            scalar_alias="task_text",
            current=True,
        )
    },
    "routing_decision": {
        "v1": _schema(
            event_type="routing_decision",
            version="v1",
            allowed_fields=("routing_source", "system", "domain", "confidence", "model_id"),
            required_fields=("routing_source", "system", "domain", "confidence", "model_id"),
            field_specs={
                "routing_source": _string(128),
                "system": _int(),
                "domain": _string(128),
                "confidence": _float_or_null(),
                "model_id": _string(256),
            },
            top_level_field_specs={
                "routing_source": _string(128),
                "system": _int(),
                "domain": _string(128),
                "confidence": _float_or_null(),
                "model_id": _string(256),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "topology_selected": {
        "v1": _schema(
            event_type="topology_selected",
            version="v1",
            allowed_fields=(
                "topology_id",
                "template_type",
                "node_count",
                "edge_count",
                "nodes",
                "edges",
            ),
            required_fields=(
                "topology_id",
                "template_type",
                "node_count",
                "edge_count",
                "nodes",
                "edges",
            ),
            field_specs={
                "topology_id": _string(256),
                "template_type": _string(128),
                "node_count": _int(),
                "edge_count": _int(),
                "nodes": _list(max_json_utf8_bytes=65536, item_type="dict"),
                "edges": _list(max_json_utf8_bytes=65536, item_type="dict"),
            },
            top_level_field_specs={
                "topology_id": _string(256),
                "template_type": _string(128),
                "node_count": _int(),
                "edge_count": _int(),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "model_assigned": {
        "v1": _schema(
            event_type="model_assigned",
            version="v1",
            allowed_fields=(
                "node_id",
                "node_role",
                "model_id",
                "provider_id",
                "required_capabilities",
            ),
            required_fields=(
                "node_id",
                "node_role",
                "model_id",
                "provider_id",
                "required_capabilities",
            ),
            field_specs={
                "node_id": _string(256),
                "node_role": _string(128),
                "model_id": _string(256),
                "provider_id": _string(128),
                "required_capabilities": _list(
                    max_json_utf8_bytes=4096,
                    item_type="str",
                ),
            },
            top_level_field_specs={
                "node_id": _string(256),
                "node_role": _string(128),
                "model_id": _string(256),
                "provider_id": _string(128),
                "required_capabilities": _list(
                    max_json_utf8_bytes=4096,
                    item_type="str",
                ),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "node_started": {
        "v1": _schema(
            event_type="node_started",
            version="v1",
            allowed_fields=_NODE_STARTED_FIELDS,
            required_fields=(
                "topology_id",
                "node_id",
                "node_role",
                "attempt",
                "model_id",
                "provider_id",
                "predecessor_ids",
                "edge_ids",
            ),
            field_specs={
                "topology_id": _string(256),
                "node_id": _string(256),
                "node_role": _string(128),
                "attempt": _int(),
                "model_id": _string(256),
                "provider_id": _string(128),
                "predecessor_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
                "edge_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
                "predecessors_by_channel": _f(
                    "dict",
                    max_json_utf8_bytes=8192,
                    notes=(
                        "StateCore ON requires exactly control/message/state keys "
                        "with sorted list[str] values; StateCore OFF forbids it."
                    ),
                ),
            },
            top_level_field_specs={
                "topology_id": _string(256),
                "node_id": _string(256),
                "node_role": _string(128),
                "attempt": _int(),
                "model_id": _string(256),
                "provider_id": _string(128),
                "predecessor_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
                "edge_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
            },
            conditional_rules=(
                "statecore_off_forbids_predecessors_by_channel",
                "statecore_on_requires_control_message_state_sorted_lists",
            ),
            payload_kind="dict",
            current=True,
        )
    },
    "node_completed": {
        "v1": _schema(
            event_type="node_completed",
            version="v1",
            allowed_fields=("output_text",),
            required_fields=("output_text",),
            field_specs={"output_text": _string(16384)},
            top_level_field_specs={
                "node_id": _string(256),
                "node_role": _string(128),
                "model_id": _string(256),
                "provider_id": _string(128),
            },
            payload_kind="scalar",
            scalar_alias="output_text",
            current=True,
        )
    },
    "controller_decision": {
        "v1_pre_allowlist_reason": _schema(
            event_type="controller_decision",
            version="v1_pre_allowlist_reason",
            allowed_fields=(
                "node_id",
                "action",
                "target_node_id",
                "gate_source_id",
                "gate_target_id",
                "quality_score",
                "quality_source",
                "threshold_band",
                "reason_code",
                "reason",
            ),
            required_fields=("action",),
            field_specs={
                "node_id": _string(256),
                "action": _f("str", allowed_values=CONTROLLER_ACTIONS),
                "target_node_id": _string(256),
                "gate_source_id": _string_or_null(256),
                "gate_target_id": _string_or_null(256),
                "quality_score": _float_or_null(),
                "quality_source": _f(
                    ("str", "null"),
                    allowed_values=("formal", "onnx", "heuristic", "external", "abstain", None),
                ),
                "threshold_band": _f(
                    ("str", "null"),
                    allowed_values=("critical", "continue", "good", None),
                ),
                "reason_code": _string(80),
                "reason": _string(4096),
            },
            top_level_field_specs={
                "node_id": _string(256),
                "action": _f("str", allowed_values=CONTROLLER_ACTIONS),
                "target_node_id": _string(256),
                "gate_source_id": _string_or_null(256),
                "gate_target_id": _string_or_null(256),
                "quality_score": _float_or_null(),
                "quality_source": _f(
                    ("str", "null"),
                    allowed_values=("formal", "onnx", "heuristic", "external", "abstain", None),
                ),
                "threshold_band": _f(
                    ("str", "null"),
                    allowed_values=("critical", "continue", "good", None),
                ),
                "reason_code": _string(80),
            },
            payload_kind="dict",
            current=False,
            legacy_read_only=True,
        ),
        "v2_allowlist_only": _schema(
            event_type="controller_decision",
            version="v2_allowlist_only",
            allowed_fields=_CONTROLLER_CURRENT_FIELDS,
            required_fields=_CONTROLLER_CURRENT_FIELDS,
            field_specs={
                "node_id": _string(256),
                "action": _f("str", allowed_values=CONTROLLER_ACTIONS),
                "target_node_id": _string(256),
                "gate_source_id": _string(256),
                "gate_target_id": _string(256),
                "quality_score": _float_or_null(),
                "quality_source": _f(
                    "str",
                    allowed_values=("formal", "onnx", "heuristic", "external", "abstain"),
                ),
                "threshold_band": _f(
                    "str",
                    allowed_values=("critical", "continue", "good"),
                ),
                "reason_code": _string(80),
            },
            top_level_field_specs={
                "node_id": _string(256),
                "action": _f("str", allowed_values=CONTROLLER_ACTIONS),
                "target_node_id": _string(256),
                "gate_source_id": _string_or_null(256),
                "gate_target_id": _string_or_null(256),
                "quality_score": _float_or_null(),
                "quality_source": _f(
                    ("str", "null"),
                    allowed_values=("formal", "onnx", "heuristic", "external", "abstain", None),
                ),
                "threshold_band": _f(
                    ("str", "null"),
                    allowed_values=("critical", "continue", "good", None),
                ),
                "reason_code": _string(80),
            },
            payload_kind="dict",
            current=True,
        ),
    },
    "state_applied": {
        "v1": _schema(
            event_type="state_applied",
            version="v1",
            allowed_fields=(
                "target_node_id",
                "source_node_ids",
                "before_version",
                "after_version",
                "delta_count",
                "conflict_count",
                "applied",
                "invalidated_assumption_ids",
            ),
            required_fields=(
                "target_node_id",
                "source_node_ids",
                "before_version",
                "after_version",
                "delta_count",
                "conflict_count",
                "applied",
                "invalidated_assumption_ids",
            ),
            field_specs={
                "target_node_id": _string(256),
                "source_node_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
                "before_version": _int(),
                "after_version": _int(),
                "delta_count": _int(),
                "conflict_count": _int(),
                "applied": _bool(),
                "invalidated_assumption_ids": _list(
                    max_json_utf8_bytes=8192,
                    item_type="str",
                ),
            },
            top_level_field_specs={
                "target_node_id": _string(256),
                "source_node_ids": _list(max_json_utf8_bytes=8192, item_type="str"),
                "before_version": _int(),
                "after_version": _int(),
                "delta_count": _int(),
                "conflict_count": _int(),
                "applied": _bool(),
                "invalidated_assumption_ids": _list(
                    max_json_utf8_bytes=8192,
                    item_type="str",
                ),
            },
            conditional_rules=(
                "after_version_gte_before_version",
                "applied_false_keeps_before_version_and_requires_conflict",
                "source_node_ids_sorted",
            ),
            payload_kind="dict",
            current=True,
        )
    },
    "failure": {
        "v1": _schema(
            event_type="failure",
            version="v1",
            allowed_fields=("kind", "error_type", "message"),
            required_fields=("kind", "error_type", "message"),
            field_specs={
                "kind": _string(128),
                "error_type": _string(256),
                "message": _string(1024),
            },
            top_level_field_specs={
                "kind": _string(128),
                "error_type": _string(256),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "budget": {
        "v1": _schema(
            event_type="budget",
            version="v1",
            allowed_fields=(
                "kind",
                "budget_limit_usd",
                "budget_remaining_usd",
                "cost_so_far_usd",
            ),
            required_fields=(
                "kind",
                "budget_limit_usd",
                "budget_remaining_usd",
                "cost_so_far_usd",
            ),
            field_specs={
                "kind": _string(128),
                "budget_limit_usd": _f("float"),
                "budget_remaining_usd": _f("float"),
                "cost_so_far_usd": _f("float"),
            },
            top_level_field_specs={
                "kind": _string(128),
                "budget_limit_usd": _f("float"),
                "budget_remaining_usd": _f("float"),
                "cost_so_far_usd": _f("float"),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "final_result": {
        "v1": _schema(
            event_type="final_result",
            version="v1",
            allowed_fields=("result_text",),
            required_fields=("result_text",),
            field_specs={"result_text": _string(16384)},
            top_level_field_specs={
                "status": _f("str", allowed_values=FINAL_RESULT_STATUSES),
            },
            payload_kind="scalar",
            scalar_alias="result_text",
            current=True,
        )
    },
    "oracle_verdict": {
        "v1": _schema(
            event_type="oracle_verdict",
            version="v1",
            allowed_fields=(
                "schema_version",
                "trainable",
                "verdict_source",
                "quality_label",
                "score",
                "confidence",
                "reason_codes",
                "evidence",
            ),
            required_fields=(
                "schema_version",
                "trainable",
                "verdict_source",
                "quality_label",
                "score",
                "confidence",
                "reason_codes",
                "evidence",
            ),
            field_specs={
                "schema_version": _string(32),
                "trainable": _bool(),
                "verdict_source": _f("str", allowed_values=VERDICT_SOURCES),
                "quality_label": _f("str", allowed_values=QUALITY_LABELS),
                "score": _float_or_null(),
                "confidence": _f("float"),
                "reason_codes": _list(
                    max_json_utf8_bytes=4096,
                    item_type="str",
                    item_max_utf8_bytes=128,
                ),
                "evidence": _list(max_json_utf8_bytes=32768, item_type="dict"),
            },
            payload_kind="dict",
            current=True,
        )
    },
    "run_frame_summary": {
        "v1": _schema(
            event_type="run_frame_summary",
            version="v1",
            allowed_fields=_RUN_FRAME_SUMMARY_FIELDS,
            required_fields=(
                "run_frame_schema_version",
                "run_frame_hash",
                "status",
                "node_record_count",
                "final_result_seq",
            ),
            field_specs={
                "run_frame_schema_version": _string(),
                "run_id": _string(256),
                "task_id": _string(256),
                "task_hash": _string(128),
                "status": _f(
                    "str",
                    allowed_values=("success", "failure", "budget_exceeded", "unknown"),
                ),
                "topology_id": _string_or_null(256),
                "graph_digest": _string_or_null(256),
                "topology_history": _list(max_json_utf8_bytes=16384, item_type="dict"),
                "node_record_count": _int(),
                "node_records": _f("dict", max_json_utf8_bytes=65536),
                "state_frame_count": _int(),
                "controller_decision_count": _int(),
                "runtime_delta_count": _int(),
                "runtime_delta_hashes": _list(
                    max_json_utf8_bytes=16384,
                    item_type="str",
                ),
                "final_result_seq": _int(),
                "failure_seqs": _list(item_type="int"),
                "terminal_failure_seq": _f(("int", "null")),
                "budget_snapshot": _dict_or_null(4096),
                "feature_flags": _f("dict", max_json_utf8_bytes=4096),
                "redacted": _bool(),
                "oracle_verdict": _dict_or_null(32768),
                "run_frame_hash": _string(),
            },
            payload_kind="dict",
            current=True,
        )
    },
}

def _current_schema_for(event_type: str) -> EventPayloadSchema:
    versions = PAYLOAD_SCHEMAS[event_type]
    current = [schema for schema in versions.values() if schema.current]
    if len(current) != 1:
        raise EventLogSchemaError(
            f"{event_type}: expected exactly one current payload schema, got {len(current)}"
        )
    return current[0]


CURRENT_PAYLOAD_SCHEMA_VERSIONS: dict[str, PayloadSchemaVersion] = {
    event_type: _current_schema_for(event_type).version for event_type in EVENT_TYPES
}


def get_current_payload_schema_version(event_type: str) -> PayloadSchemaVersion:
    try:
        return CURRENT_PAYLOAD_SCHEMA_VERSIONS[event_type]
    except KeyError as exc:
        raise EventLogSchemaError(f"unknown event type: {event_type!r}") from exc


def get_schema_for_event(
    event_type: str,
    version: PayloadSchemaVersion | None = None,
) -> EventPayloadSchema:
    if version is None:
        version = get_current_payload_schema_version(event_type)
    try:
        return PAYLOAD_SCHEMAS[event_type][version]
    except KeyError as exc:
        raise EventLogSchemaError(
            f"unknown payload schema: {event_type}.{version}"
        ) from exc


def _assert_current_payload_schema_for_emit(event_type: str) -> EventPayloadSchema:
    schema = get_schema_for_event(event_type)
    if schema.legacy_read_only or not schema.current:
        raise EventLogSchemaError(
            f"{event_type}.{schema.version} is not valid for new emission"
        )
    return schema


def _resolve_payload_schema_version(
    event_type: str,
    payload: Any,
    version: PayloadSchemaVersion | None = None,
    *,
    top_level_event: Mapping[str, Any] | None = None,
    statecore_profile: Literal["on", "off"] | None = None,
    allow_absent_payload: bool = True,
) -> tuple[EventPayloadSchema, bool]:
    """Resolve explicit or historical missing ``payload_schema_version``.

    Returns ``(schema, inferred)`` where ``inferred`` is true when the event
    envelope had no explicit payload schema version.
    """
    if version is not None:
        schema = get_schema_for_event(event_type, version)
        _validate_payload_against_schema(
            payload,
            schema,
            top_level_event=top_level_event,
            allow_absent_payload=allow_absent_payload,
            statecore_profile=statecore_profile,
        )
        return schema, False

    try:
        versions = PAYLOAD_SCHEMAS[event_type]
    except KeyError as exc:
        raise EventLogSchemaError(f"unknown event type: {event_type!r}") from exc

    matching: list[EventPayloadSchema] = []
    for candidate in versions.values():
        try:
            _validate_payload_against_schema(
                payload,
                candidate,
                top_level_event=top_level_event,
                allow_absent_payload=allow_absent_payload,
                statecore_profile=statecore_profile,
            )
        except EventLogSchemaError:
            continue
        matching.append(candidate)

    if not matching:
        raise EventLogSchemaError(
            f"{event_type}: missing payload_schema_version and payload matches no "
            "registered schema"
        )

    current_version = get_current_payload_schema_version(event_type)
    for candidate in matching:
        if candidate.version == current_version:
            return candidate, True
    return matching[0], True


def _payload_schema_distribution_for_events(
    events: Iterable[Mapping[str, Any]],
    *,
    mode: PayloadSchemaMode = "audit",
) -> _PayloadSchemaDistributionReport:
    distribution: dict[str, dict[str, PayloadSchemaDistributionEntry]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for event in events:
        event_type = event.get("event_type")
        if not isinstance(event_type, str):
            errors.append("event missing string event_type")
            _add_distribution(distribution, "<unknown>", "<unknown>", False, "unknown_rejected")
            continue

        explicit_version_raw = event.get("payload_schema_version")
        explicit_version = (
            explicit_version_raw if isinstance(explicit_version_raw, str) else None
        )
        payload = event.get("payload")

        try:
            payload_absent = "payload" not in event or event.get("payload") is None
            allow_absent_payload = (
                payload_absent
                and event.get("redaction_state") == "redacted"
                and event_type
                not in {"controller_decision", "oracle_verdict", "run_frame_summary"}
            )
            schema, inferred = _resolve_payload_schema_version(
                event_type,
                payload,
                explicit_version,
                top_level_event=event,
                allow_absent_payload=allow_absent_payload,
            )
            current_version = get_current_payload_schema_version(event_type)
        except EventLogSchemaError as exc:
            version_for_distribution = explicit_version or "<unresolved>"
            _add_distribution(
                distribution,
                event_type,
                version_for_distribution,
                explicit_version is not None,
                "unknown_rejected",
            )
            errors.append(f"{event_type}@seq{event.get('seq', '?')}: {exc}")
            continue

        if schema.version == current_version:
            status = "current"
        elif mode == "audit":
            status = "legacy_accepted"
            warnings.append(
                f"{event_type}@seq{event.get('seq', '?')}: accepted legacy "
                f"{schema.version} in audit mode"
            )
        else:
            status = "legacy_rejected_strict_current"

        _add_distribution(
            distribution,
            event_type,
            schema.version,
            not inferred,
            status,
        )

        if mode == "strict-current":
            if inferred:
                errors.append(
                    f"{event_type}@seq{event.get('seq', '?')}: strict-current "
                    "requires explicit payload_schema_version"
                )
            if schema.version != current_version:
                errors.append(
                    f"{event_type}@seq{event.get('seq', '?')}: "
                    f"{schema.version} != current {current_version}"
                )

    return _PayloadSchemaDistributionReport(
        mode=mode,
        distribution=distribution,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _add_distribution(
    distribution: dict[str, dict[str, PayloadSchemaDistributionEntry]],
    event_type: str,
    version: str,
    explicit: bool,
    status: str,
) -> None:
    by_version = distribution.setdefault(event_type, {})
    entry = by_version.setdefault(
        version,
        {
            "count": 0,
            "explicit_count": 0,
            "inferred_count": 0,
            "status": status,
        },
    )
    entry["count"] = int(entry["count"]) + 1
    if explicit:
        entry["explicit_count"] = int(entry["explicit_count"]) + 1
    else:
        entry["inferred_count"] = int(entry["inferred_count"]) + 1
    entry["status"] = status


def _validate_payload_against_schema(
    payload: Any,
    schema: EventPayloadSchema,
    *,
    top_level_event: Mapping[str, Any] | None = None,
    allow_absent_payload: bool = False,
    statecore_profile: Literal["on", "off"] | None = None,
) -> None:
    prefix = f"{schema.event_type}.{schema.version}"

    if payload is None and allow_absent_payload:
        _validate_top_level_fields(top_level_event, schema, prefix)
        return

    if schema.payload_kind == "none":
        if payload is not None:
            raise EventLogSchemaError(f"{prefix}: expected no payload")
        _validate_top_level_fields(top_level_event, schema, prefix)
        return

    if schema.payload_kind == "scalar":
        alias = schema.scalar_alias
        if alias is None:
            raise EventLogSchemaError(f"{prefix}: scalar schema missing alias")
        spec = schema.field_specs[alias]
        if payload is None:
            if spec.required and not allow_absent_payload:
                raise EventLogSchemaError(
                    f"{prefix}: missing required payload field {alias!r}"
                )
        else:
            _validate_value(alias, payload, spec, prefix)
        _validate_top_level_fields(top_level_event, schema, prefix)
        return

    if not isinstance(payload, Mapping):
        raise EventLogSchemaError(f"{prefix}: expected dict payload")

    payload_keys = set(payload.keys())
    non_string_keys = [key for key in payload_keys if not isinstance(key, str)]
    if non_string_keys:
        raise EventLogSchemaError(f"{prefix}: payload keys must be strings")

    allowed = set(schema.allowed_fields)
    extras = sorted(str(key) for key in payload_keys - allowed)
    if extras:
        raise EventLogSchemaError(f"{prefix}: extra payload field(s): {extras}")

    for required in schema.required_fields:
        if required not in payload:
            raise EventLogSchemaError(
                f"{prefix}: missing required payload field {required!r}"
            )

    for field_name, value in payload.items():
        field_spec = schema.field_specs.get(str(field_name))
        if field_spec is None:
            raise EventLogSchemaError(f"{prefix}: no field spec for {field_name!r}")
        _validate_value(str(field_name), value, field_spec, prefix)

    _validate_conditionals(
        cast(Mapping[str, Any], payload),
        schema,
        prefix,
        statecore_profile=statecore_profile,
    )
    _validate_top_level_fields(top_level_event, schema, prefix)


def _validate_top_level_fields(
    top_level_event: Mapping[str, Any] | None,
    schema: EventPayloadSchema,
    prefix: str,
) -> None:
    if top_level_event is None or schema.top_level_field_specs is None:
        return
    for field_name, spec in schema.top_level_field_specs.items():
        if field_name in top_level_event and top_level_event[field_name] is not None:
            _validate_value(f"top_level.{field_name}", top_level_event[field_name], spec, prefix)


def _validate_value(
    field_name: str,
    value: Any,
    spec: PayloadFieldSpec,
    prefix: str,
) -> None:
    allowed_types = _json_types(spec.json_type)
    if "any_json" in allowed_types:
        if not _is_json_value(value):
            raise EventLogSchemaError(f"{prefix}: {field_name} is not JSON-serializable")
    elif not any(_matches_json_type(value, type_name) for type_name in allowed_types):
        raise EventLogSchemaError(
            f"{prefix}: {field_name} has invalid type "
            f"{type(value).__name__}; expected {allowed_types}"
        )

    if spec.allowed_values is not None and value not in spec.allowed_values:
        raise EventLogSchemaError(
            f"{prefix}: {field_name}={value!r} not in allowed values "
            f"{list(spec.allowed_values)!r}"
        )

    if isinstance(value, str):
        limit = spec.max_utf8_bytes
        if limit is None and "str" in allowed_types:
            limit = DEFAULT_PAYLOAD_STRING_MAX_BYTES
        if limit is not None:
            size = len(value.encode("utf-8", errors="replace"))
            if size > limit:
                raise EventLogSchemaError(
                    f"{prefix}: {field_name} exceeds max UTF-8 bytes "
                    f"({size}>{limit})"
                )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise EventLogSchemaError(f"{prefix}: {field_name} is non-finite")

    if isinstance(value, (list, tuple)):
        if spec.item_type is not None:
            for index, item in enumerate(value):
                _validate_list_item(field_name, index, item, spec, prefix)
        limit = spec.max_json_utf8_bytes or DEFAULT_PAYLOAD_JSON_MAX_BYTES
        size = _canonical_json_size_bytes(value)
        if size > limit:
            raise EventLogSchemaError(
                f"{prefix}: {field_name} exceeds max JSON UTF-8 bytes ({size}>{limit})"
            )

    if isinstance(value, Mapping):
        limit = spec.max_json_utf8_bytes or DEFAULT_PAYLOAD_JSON_MAX_BYTES
        size = _canonical_json_size_bytes(value)
        if size > limit:
            raise EventLogSchemaError(
                f"{prefix}: {field_name} exceeds max JSON UTF-8 bytes ({size}>{limit})"
            )


def _validate_list_item(
    field_name: str,
    index: int,
    item: Any,
    spec: PayloadFieldSpec,
    prefix: str,
) -> None:
    assert spec.item_type is not None
    item_spec = PayloadFieldSpec(
        json_type=spec.item_type,
        max_utf8_bytes=spec.item_max_utf8_bytes,
    )
    _validate_value(f"{field_name}[{index}]", item, item_spec, prefix)


def _validate_conditionals(
    payload: Mapping[str, Any],
    schema: EventPayloadSchema,
    prefix: str,
    *,
    statecore_profile: Literal["on", "off"] | None,
) -> None:
    if schema.event_type == "node_started":
        _validate_node_started_conditionals(payload, prefix, statecore_profile)
    elif schema.event_type == "state_applied":
        _validate_state_applied_conditionals(payload, prefix)
    elif schema.event_type == "oracle_verdict":
        _validate_oracle_verdict_conditionals(payload, prefix)


def _validate_node_started_conditionals(
    payload: Mapping[str, Any],
    prefix: str,
    statecore_profile: Literal["on", "off"] | None,
) -> None:
    has_partition = "predecessors_by_channel" in payload
    if statecore_profile == "off" and has_partition:
        raise EventLogSchemaError(
            f"{prefix}: predecessors_by_channel is forbidden in StateCore OFF"
        )
    if statecore_profile == "on" and not has_partition:
        raise EventLogSchemaError(
            f"{prefix}: predecessors_by_channel is required in StateCore ON"
        )
    if not has_partition:
        return

    partition = payload["predecessors_by_channel"]
    if not isinstance(partition, Mapping):
        raise EventLogSchemaError(f"{prefix}: predecessors_by_channel must be a dict")
    if set(partition.keys()) != {"control", "message", "state"}:
        raise EventLogSchemaError(
            f"{prefix}: predecessors_by_channel must contain exactly "
            "control/message/state"
        )
    for channel, values in partition.items():
        if not isinstance(values, list):
            raise EventLogSchemaError(
                f"{prefix}: predecessors_by_channel.{channel} must be list[str]"
            )
        if any(not isinstance(item, str) for item in values):
            raise EventLogSchemaError(
                f"{prefix}: predecessors_by_channel.{channel} must be list[str]"
            )
        if values != sorted(values):
            raise EventLogSchemaError(
                f"{prefix}: predecessors_by_channel.{channel} must be sorted"
            )


def _validate_state_applied_conditionals(
    payload: Mapping[str, Any],
    prefix: str,
) -> None:
    before = payload.get("before_version")
    after = payload.get("after_version")
    conflict_count = payload.get("conflict_count")
    applied = payload.get("applied")
    if isinstance(before, int) and isinstance(after, int) and after < before:
        raise EventLogSchemaError(f"{prefix}: after_version must be >= before_version")
    if applied is False:
        if after != before:
            raise EventLogSchemaError(
                f"{prefix}: applied=False requires after_version == before_version"
            )
        if not isinstance(conflict_count, int) or conflict_count <= 0:
            raise EventLogSchemaError(
                f"{prefix}: applied=False requires conflict_count > 0"
            )
    sources = payload.get("source_node_ids")
    if isinstance(sources, list) and sources != sorted(sources):
        raise EventLogSchemaError(f"{prefix}: source_node_ids must be sorted")


def _validate_oracle_verdict_conditionals(
    payload: Mapping[str, Any],
    prefix: str,
) -> None:
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise EventLogSchemaError(f"{prefix}: evidence[{index}] must be a dict")
        extras = sorted(set(item.keys()) - _EVIDENCE_REF_FIELDS)
        if extras:
            raise EventLogSchemaError(
                f"{prefix}: evidence[{index}] may contain only EvidenceRef-style "
                f"metadata fields; extra={extras}"
            )


def _json_types(spec: PayloadJsonTypeSpec) -> tuple[PayloadJsonType, ...]:
    if isinstance(spec, tuple):
        return spec
    return (spec,)


def _matches_json_type(value: Any, type_name: PayloadJsonType) -> bool:
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if type_name == "list":
        return isinstance(value, (list, tuple))
    if type_name == "dict":
        return isinstance(value, Mapping)
    if type_name == "any_json":
        return _is_json_value(value)
    return False


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _canonical_json_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8", errors="replace")
    )


def _schema_manifest_canonical_text(schema: EventPayloadSchema) -> str:
    """Canonical compact text representation of a manifest, OS-stable.

    Used both to write committed manifests and to assert byte-exact drift
    in the regression test. Policy locked by cgpro 2026-04-30 cycle-8 R6.1c
    VERIFY round-1: ``sort_keys=True, ensure_ascii=False, separators=(",", ":")``
    with no trailing newline. Single-line compact JSON avoids CRLF/LF and
    indentation drift across Linux CI vs Windows local checkouts.
    """
    return json.dumps(
        _schema_to_manifest(schema),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _schema_to_manifest(schema: EventPayloadSchema) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "event_type": schema.event_type,
        "version": schema.version,
        "payload_kind": schema.payload_kind,
        "current": schema.current,
        "legacy_read_only": schema.legacy_read_only,
        "allowed_fields": list(schema.allowed_fields),
        "required_fields": list(schema.required_fields),
        "field_specs": {
            name: _field_spec_to_manifest(schema.field_specs[name])
            for name in sorted(schema.field_specs)
        },
    }
    if schema.scalar_alias is not None:
        manifest["scalar_alias"] = schema.scalar_alias
    if schema.conditional_rules:
        manifest["conditional_rules"] = list(schema.conditional_rules)
    if schema.top_level_field_specs:
        manifest["top_level_field_specs"] = {
            name: _field_spec_to_manifest(schema.top_level_field_specs[name])
            for name in sorted(schema.top_level_field_specs)
        }
    return manifest


def _field_spec_to_manifest(spec: PayloadFieldSpec) -> dict[str, Any]:
    item: dict[str, Any] = {
        "json_type": list(spec.json_type) if isinstance(spec.json_type, tuple) else spec.json_type,
        "required": spec.required,
    }
    if spec.max_utf8_bytes is not None:
        item["max_utf8_bytes"] = spec.max_utf8_bytes
    if spec.max_json_utf8_bytes is not None:
        item["max_json_utf8_bytes"] = spec.max_json_utf8_bytes
    if spec.allowed_values is not None:
        item["allowed_values"] = list(spec.allowed_values)
    if spec.item_type is not None:
        item["item_type"] = (
            list(spec.item_type) if isinstance(spec.item_type, tuple) else spec.item_type
        )
    if spec.item_max_utf8_bytes is not None:
        item["item_max_utf8_bytes"] = spec.item_max_utf8_bytes
    if spec.notes:
        item["notes"] = spec.notes
    return item


def _validate_schema_registry() -> None:
    missing = set(EVENT_TYPES) - set(PAYLOAD_SCHEMAS)
    extra = set(PAYLOAD_SCHEMAS) - set(EVENT_TYPES)
    if missing or extra:
        raise EventLogSchemaError(
            f"payload schema registry drift: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for event_type, versions in PAYLOAD_SCHEMAS.items():
        current_count = sum(1 for schema in versions.values() if schema.current)
        if current_count != 1:
            raise EventLogSchemaError(
                f"{event_type}: expected one current schema, got {current_count}"
            )
        for version, schema in versions.items():
            if version != schema.version:
                raise EventLogSchemaError(f"{event_type}: version key mismatch {version}")
            if not _PAYLOAD_SCHEMA_VERSION_RE.fullmatch(schema.version):
                raise EventLogSchemaError(
                    f"{event_type}: invalid payload schema version {schema.version!r}"
                )
            if set(schema.required_fields) - set(schema.allowed_fields):
                raise EventLogSchemaError(
                    f"{event_type}.{version}: required fields outside allowed fields"
                )
            if set(schema.allowed_fields) - set(schema.field_specs):
                raise EventLogSchemaError(
                    f"{event_type}.{version}: allowed fields missing specs"
                )


_validate_schema_registry()
