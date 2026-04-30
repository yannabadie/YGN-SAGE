from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from sage.runtime.event_log.errors import EventLogSchemaError
from sage.runtime.event_log.schema import EVENT_TYPES
from sage.runtime.event_log.payload_schemas import (
    CURRENT_PAYLOAD_SCHEMA_VERSIONS,
    PAYLOAD_SCHEMAS,
    _schema_to_manifest,
    _validate_payload_against_schema,
    get_current_payload_schema_version,
    get_schema_for_event,
)


MANIFEST_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "sage"
    / "runtime"
    / "event_log"
    / "payload_schema_manifests"
)

GOLDEN_DIR = pathlib.Path(__file__).parent / "golden" / "runtime_events"


def test_all_13_event_types_have_schemas() -> None:
    assert len(EVENT_TYPES) == 13
    for event_type in EVENT_TYPES:
        versions = PAYLOAD_SCHEMAS.get(event_type)
        assert versions, f"{event_type} has no registered payload schema"
        current = [schema for schema in versions.values() if schema.current]
        assert len(current) == 1, f"{event_type} must have exactly one current schema"
        assert CURRENT_PAYLOAD_SCHEMA_VERSIONS[event_type] == current[0].version


def test_payload_schema_manifests_match_python_sot() -> None:
    """Byte-exact manifest drift tripwire (cgpro 2026-04-30 cycle-8 R6.1c
    VERIFY round-1, fix #1).

    Compares raw UTF-8 bytes — not just ``json.loads(==)`` semantics — so
    that key order, whitespace, and trailing-newline drift across OS
    (Linux CI vs Windows local) all surface as test failures, not silent
    schema drift.
    """
    from sage.runtime.event_log.payload_schemas import _schema_manifest_canonical_text

    expected_text: dict[str, str] = {}
    expected_dict: dict[str, dict[str, Any]] = {}
    for versions in PAYLOAD_SCHEMAS.values():
        for schema in versions.values():
            name = f"{schema.event_type}.{schema.version}.json"
            expected_text[name] = _schema_manifest_canonical_text(schema)
            expected_dict[name] = _schema_to_manifest(schema)

    actual_paths = sorted(MANIFEST_DIR.glob("*.json"))
    assert {path.name for path in actual_paths} == set(expected_text)
    for path in actual_paths:
        # Read bytes (not text) to detect any encoding or line-ending drift.
        actual_bytes = path.read_bytes()
        expected_bytes = expected_text[path.name].encode("utf-8")
        assert actual_bytes == expected_bytes, (
            f"manifest byte drift in {path.name}: "
            f"file_len={len(actual_bytes)} expected_len={len(expected_bytes)}"
        )
        # Belt-and-suspenders: also assert semantic equality.
        actual_obj = json.loads(actual_bytes.decode("utf-8"))
        assert actual_obj == expected_dict[path.name]


def test_current_payload_schema_versions_are_registered() -> None:
    for event_type, version in CURRENT_PAYLOAD_SCHEMA_VERSIONS.items():
        assert get_current_payload_schema_version(event_type) == version
        assert get_schema_for_event(event_type, version).version == version
        assert get_schema_for_event(event_type).version == version


def test_controller_decision_legacy_v1_reason_schema_is_read_only() -> None:
    legacy = get_schema_for_event("controller_decision", "v1_pre_allowlist_reason")
    current = get_schema_for_event("controller_decision")

    assert legacy.legacy_read_only is True
    assert legacy.current is False
    assert "reason" in legacy.allowed_fields
    assert "reason" not in current.allowed_fields


def test_controller_decision_current_v2_rejects_reason() -> None:
    schema = get_schema_for_event("controller_decision")
    payload = {
        "node_id": "n0",
        "action": "continue",
        "target_node_id": "",
        "gate_source_id": "",
        "gate_target_id": "",
        "quality_score": None,
        "quality_source": "abstain",
        "threshold_band": "continue",
        "reason_code": "abstain_no_signal",
        "reason": "free-form text must not be current",
    }

    with pytest.raises(EventLogSchemaError, match="extra payload field"):
        _validate_payload_against_schema(payload, schema)


def test_node_started_statecore_off_forbids_predecessors_by_channel() -> None:
    schema = get_schema_for_event("node_started")
    payload = _minimal_payload_for("node_started")
    payload["predecessors_by_channel"] = {"control": [], "message": [], "state": []}

    with pytest.raises(EventLogSchemaError, match="forbidden in StateCore OFF"):
        _validate_payload_against_schema(payload, schema, statecore_profile="off")


def test_node_started_statecore_on_requires_predecessors_by_channel() -> None:
    schema = get_schema_for_event("node_started")
    payload = _minimal_payload_for("node_started")

    with pytest.raises(EventLogSchemaError, match="required in StateCore ON"):
        _validate_payload_against_schema(payload, schema, statecore_profile="on")

    payload["predecessors_by_channel"] = {
        "control": [],
        "message": ["n0"],
        "state": ["n1"],
    }
    _validate_payload_against_schema(payload, schema, statecore_profile="on")


def test_oracle_verdict_evidence_cap_and_no_raw_output() -> None:
    schema = get_schema_for_event("oracle_verdict")
    payload = _minimal_payload_for("oracle_verdict")
    payload["evidence"] = [{"run_id": "r", "raw_output": "must not be accepted"}]

    with pytest.raises(EventLogSchemaError, match="EvidenceRef-style"):
        _validate_payload_against_schema(payload, schema)

    payload = _minimal_payload_for("oracle_verdict")
    payload["evidence"] = [{"run_id": "r", "evidence_hash": "x" * 40000}]
    with pytest.raises(EventLogSchemaError, match="exceeds max JSON UTF-8 bytes"):
        _validate_payload_against_schema(payload, schema)


def test_run_frame_summary_allowed_keys() -> None:
    schema = get_schema_for_event("run_frame_summary")
    payload = _minimal_payload_for("run_frame_summary")
    payload["new_summary_key_without_schema_bump"] = True

    with pytest.raises(EventLogSchemaError, match="extra payload field"):
        _validate_payload_against_schema(payload, schema)


def test_golden_runtime_event_fixtures_validate_against_payload_schemas() -> None:
    for fixture_name in sorted(GOLDEN_DIR.glob("*.json")):
        fixture = json.loads(fixture_name.read_text(encoding="utf-8"))
        event_type = fixture["event_type"]
        schema = get_schema_for_event(event_type)
        required_payload = set(fixture.get("_required_in_payload", []))
        assert required_payload <= set(schema.allowed_fields)
        if event_type in {"controller_decision", "run_frame_summary", "oracle_verdict"}:
            assert required_payload <= set(schema.required_fields)

    off_fixture = json.loads(
        (GOLDEN_DIR / "statecore_off_node_started.json").read_text(encoding="utf-8")
    )
    on_fixture = json.loads(
        (GOLDEN_DIR / "statecore_on_node_started.json").read_text(encoding="utf-8")
    )
    assert "predecessors_by_channel" not in off_fixture["_required_in_payload"]
    assert "predecessors_by_channel" in on_fixture["_required_in_payload"]


def _minimal_payload_for(event_type: str) -> dict[str, Any]:
    if event_type == "node_started":
        return {
            "topology_id": "topo",
            "node_id": "n1",
            "node_role": "worker",
            "attempt": 1,
            "model_id": "model",
            "provider_id": "provider",
            "predecessor_ids": ["n0"],
            "edge_ids": ["e0"],
        }
    if event_type == "oracle_verdict":
        return {
            "schema_version": "0",
            "trainable": True,
            "verdict_source": "exact",
            "quality_label": "pass",
            "score": 1.0,
            "confidence": 1.0,
            "reason_codes": ["exact_pass"],
            "evidence": [{"run_id": "r", "evidence_hash": "h"}],
        }
    if event_type == "run_frame_summary":
        return {
            "run_frame_schema_version": "0",
            "run_frame_hash": "h",
            "status": "success",
            "node_record_count": 0,
            "final_result_seq": 1,
        }
    raise AssertionError(f"no minimal payload helper for {event_type}")
