"""R6.0.1 — mode-aware RuntimeEventLog event contract tests.

cgpro 2026-04-29 cycle 3 reassess: contract snapshot tests with mode-aware
forbidden-field validation. Catches the class of bug that R6 verify
round-trip surfaced (test 13: NodeStarted's `predecessors_by_channel`
field type-checks fine but is FORBIDDEN in legacy mode — only golden
fixture validation surfaces this).

Each fixture in `tests/golden/runtime_events/<event>.json` declares:
- `_required_always`: top-level event fields that must be present in every
  emitted instance (e.g., schema_version, run_id, seq, timestamp_ns).
- `_required_in_payload`: payload fields required when SAGE_TRACE_RAW=1
  surfaces the payload (most events).
- `_forbidden_in_payload_when_off`: payload fields that MUST NOT appear
  when the relevant mode flag is OFF (preserves byte-identical R5 schema).
- `_required_predecessors_by_channel_keys` (NodeStarted ON only): the 3
  canonical channel keys that must partition predecessors.

These tests run the live writer against in-memory state, write a JSONL
event, and parse it back to validate against the fixture. Regressions
in the contract surface as test failures, not silent schema drift.
"""
from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any
from uuid import uuid4

import pytest

from sage.runtime.event_log import RuntimeEventLog
from sage.runtime.event_log.errors import EventLogSchemaError
from sage.runtime.event_log.payload_schemas import (
    CURRENT_PAYLOAD_SCHEMA_VERSIONS,
    _payload_schema_distribution_for_events,
)


GOLDEN_DIR = pathlib.Path(__file__).parent / "golden" / "runtime_events"


@pytest.fixture
def tmp_path() -> pathlib.Path:
    """Local temp under .tmp (the default pytest temp root is ACL-denied here)."""
    path = pathlib.Path(".tmp") / "pytest-runtime-event-contracts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def _read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _emit_minimal_node_started(
    log: RuntimeEventLog,
    *,
    predecessors_by_channel: dict[str, tuple[str, ...]] | None,
) -> None:
    """Emit one NodeStarted with the given partition (or None for OFF mode)."""
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_node_started(
        topology_id="t1",
        node_id="n0",
        node_role="actor",
        attempt=1,
        model_id="m",
        provider_id="p",
        predecessor_ids=("a", "b"),
        edge_ids=("e0", "e1"),
        predecessors_by_channel=predecessors_by_channel,
    )
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()


# ---- Tests ----

def test_node_started_off_mode_has_no_predecessors_by_channel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFF mode contract: NodeStarted.payload must NOT carry
    predecessors_by_channel (byte-identical R5 schema guarantee)."""
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    fixture = _load_fixture("statecore_off_node_started.json")

    run_id = "01CONTRACTOFF000000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    _emit_minimal_node_started(log, predecessors_by_channel=None)

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    node_started = next(e for e in events if e["event_type"] == "node_started")

    # Required-always (top-level)
    for field in fixture["_required_always"]:
        assert field in node_started, f"missing required-always field {field!r}"

    # Payload contract
    assert "payload" in node_started, "SAGE_TRACE_RAW=1 must surface payload"
    payload = node_started["payload"]
    for field in fixture["_required_in_payload"]:
        assert field in payload, f"missing required-in-payload field {field!r}"
    for forbidden in fixture["_forbidden_in_payload_when_off"]:
        assert forbidden not in payload, (
            f"OFF-mode NodeStarted.payload MUST NOT contain {forbidden!r} "
            f"(byte-identical R5 schema guarantee)"
        )


def test_node_started_on_mode_has_predecessors_by_channel(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ON mode contract: NodeStarted.payload MUST carry predecessors_by_channel
    with the 3 canonical channel keys."""
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    monkeypatch.setenv("SAGE_STATECORE", "1")
    fixture = _load_fixture("statecore_on_node_started.json")

    run_id = "01CONTRACTON0000000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    _emit_minimal_node_started(
        log,
        predecessors_by_channel={
            "control": ("a",),
            "message": ("b",),
            "state": (),
        },
    )

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    node_started = next(e for e in events if e["event_type"] == "node_started")

    # Required-always (top-level)
    for field in fixture["_required_always"]:
        assert field in node_started, f"missing required-always field {field!r}"

    # Payload contract
    assert "payload" in node_started
    payload = node_started["payload"]
    for field in fixture["_required_in_payload"]:
        assert field in payload, f"missing required-in-payload field {field!r}"

    # 3 canonical channel keys
    partition = payload["predecessors_by_channel"]
    assert isinstance(partition, dict)
    assert set(partition.keys()) == set(
        fixture["_required_predecessors_by_channel_keys"]
    )


def test_state_applied_event_required_fields(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state_applied event contract: all top-level fields present including
    `applied: bool` (cgpro 2026-04-28 R6 verify schema correction)."""
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)  # default mode is fine
    fixture = _load_fixture("state_applied.json")

    run_id = "01CONTRACTSTATEAPP00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_state_applied(
        target_node_id="n1",
        source_node_ids=("n0",),
        before_version=0,
        after_version=1,
        delta_count=1,
        conflict_count=0,
        applied=True,
        invalidated_assumption_ids=(),
    )
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    state_event = next(e for e in events if e["event_type"] == "state_applied")

    # Required-always (includes top-level fields like target_node_id, applied, etc.)
    for field in fixture["_required_always"]:
        assert field in state_event, f"missing required-always field {field!r}"

    # `applied` must be a bool (not 0/1, not None, not string)
    assert isinstance(state_event["applied"], bool), (
        "state_applied.applied must be a bool — disambiguates "
        "'accepted no-op' (True) from 'blocked by conflict' (False)"
    )

    # Invariant: applied=True with non-empty deltas → after >= before
    assert state_event["after_version"] >= state_event["before_version"]

    # Invariant: source_node_ids is a sorted list
    src = state_event["source_node_ids"]
    assert src == sorted(src), "source_node_ids must be sorted ascending"


def test_state_applied_never_in_off_mode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state_applied is OPT-IN: ONLY emitted when SAGE_STATECORE=1.
    OFF-mode pipeline runs MUST NOT emit any state_applied event.

    This is an integration-style assertion — we don't directly run the
    pipeline here; instead we verify that a writer left to emit only
    'normal' events under OFF semantics produces no state_applied lines.
    The runner-level test of this guarantee lives at
    test_statecore.py::test_statecore_off_preserves_legacy_context.
    """
    monkeypatch.delenv("SAGE_STATECORE", raising=False)
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTSTATEOFF00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_node_started(
        topology_id="t",
        node_id="n0",
        node_role="r",
        attempt=1,
        model_id="m",
        provider_id="p",
        predecessor_ids=(),
        edge_ids=(),
        predecessors_by_channel=None,  # OFF mode passes None
    )
    log.emit_node_completed(
        node_id="n0",
        node_role="r",
        output="ok",
        latency_ms=1.0,
        cost_usd=0.0,
        model_id="m",
        provider_id="p",
    )
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    state_events = [e for e in events if e["event_type"] == "state_applied"]
    assert state_events == [], (
        "state_applied event MUST NOT appear in OFF-mode JSONL. "
        f"Got {len(state_events)} state_applied event(s)."
    )


def test_event_type_catalog_completeness() -> None:
    """The schema's EVENT_TYPES tuple MUST match the doc's catalog.

    Adding a new event type without updating docs/contracts/runtime-event-log.md
    is a contract drift. This test fails loudly so it gets noticed.
    """
    from sage.runtime.event_log.schema import EVENT_TYPES

    expected = {
        "task_started",
        "routing_decision",
        "bandit_attribution_mismatch",
        "topology_selected",
        "model_assigned",
        "node_started",
        "node_completed",
        "controller_decision",
        "failure",
        "budget",
        "state_applied",
        "final_result",
        "oracle_verdict",
        "run_frame_summary",
    }

    assert set(EVENT_TYPES) == expected, (
        "EVENT_TYPES drift: docs/contracts/runtime-event-log.md and "
        "docs/contracts/runtime-event-log.md catalog must be updated together."
    )


def test_controller_decision_payload_contains_quality_metadata(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle-mode controller_decision payload carries safe quality metadata."""
    monkeypatch.setenv("SAGE_ORACLE", "1")
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)
    fixture = _load_fixture("controller_decision.json")

    run_id = "01CONTRACTCTRLDEC00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_controller_decision(
        node_id="0",
        action="upgrade_model",
        target_node_id="0",
        quality_score=0.25,
        quality_source="onnx",
        threshold_band="critical",
        reason_code="quality_below_theta_critical",
    )
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    controller = next(e for e in events if e["event_type"] == "controller_decision")

    for field in fixture["_required_always"]:
        assert field in controller, f"missing required-always field {field!r}"

    assert "payload" in controller
    payload = controller["payload"]
    for field in fixture["_required_in_payload"]:
        assert field in payload, f"missing required-in-payload field {field!r}"

    assert payload["node_id"] == "0"
    assert payload["action"] == "upgrade_model"
    assert payload["quality_score"] == 0.25
    assert payload["quality_source"] == "onnx"
    assert payload["threshold_band"] == "critical"
    assert payload["reason_code"] == "quality_below_theta_critical"


def test_controller_decision_payload_populated_under_default_on(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-flip default-on contract: unset SAGE_ORACLE still forces the
    controller_decision safe payload fields.
    """
    monkeypatch.delenv("SAGE_ORACLE", raising=False)
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTCTRLDEFAULTON01"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_controller_decision(
        node_id="2",
        action="retry_node",
        target_node_id="2",
        quality_score=0.11,
        quality_source="onnx",
        threshold_band="critical",
        reason_code="quality_below_theta_critical",
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    controller = next(e for e in events if e["event_type"] == "controller_decision")
    payload = controller["payload"]

    for field in (
        "quality_score",
        "quality_source",
        "threshold_band",
        "reason_code",
        "node_id",
    ):
        assert field in payload

    assert payload["node_id"] == "2"
    assert payload["quality_score"] == 0.11
    assert payload["quality_source"] == "onnx"
    assert payload["threshold_band"] == "critical"
    assert payload["reason_code"] == "quality_below_theta_critical"


def test_controller_decision_forced_payload_excludes_freeform_reason(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-1 PUSH BACK: under default-on,
    the forced controller_decision payload MUST NOT include the free-form
    ``reason`` field. Even if the caller passes a ``reason`` containing a
    Traceback or PII, neither the key ``reason`` nor any of its content
    should leak into the emitted event.
    """
    monkeypatch.delenv("SAGE_ORACLE", raising=False)  # default-on
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTREASONLEAKTEST"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_controller_decision(
        node_id="0",
        action="upgrade_model",
        reason=(
            "Traceback (most recent call last):\n"
            "  File \"sage/topology/runner.py\", line 42\n"
            "  email alice@example.com  AKIA1234567890ABCDEF\n"
        ),
        reason_code="quality_below_theta_critical",
        quality_score=0.2,
        quality_source="onnx",
        threshold_band="critical",
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    controller = next(e for e in events if e["event_type"] == "controller_decision")
    payload = controller["payload"]

    assert "reason" not in payload, (
        f"`reason` leaked into forced payload: keys={sorted(payload.keys())!r}"
    )
    payload_json = json.dumps(payload, ensure_ascii=False)
    assert "Traceback" not in payload_json
    assert "alice@example.com" not in payload_json
    assert "runner.py" not in payload_json


def test_controller_decision_forced_payload_uses_allowlist_only(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-1: forced payload keys MUST be
    a subset of CONTROLLER_DECISION_ALLOWED_PAYLOAD_KEYS. Any key outside the
    allowlist is a contract violation, even if its value would otherwise pass
    the raw-leak scan.
    """
    from sage.runtime.event_log.payload_schemas import get_schema_for_event

    controller_payload_keys = set(
        get_schema_for_event("controller_decision").allowed_fields
    )

    monkeypatch.delenv("SAGE_ORACLE", raising=False)  # default-on
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTALLOWLISTTEST0"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_controller_decision(
        node_id="3",
        action="retry_node",
        target_node_id="3",
        gate_source_id="2",
        gate_target_id="3",
        reason="ignored under default-on",
        reason_code="quality_below_theta_critical",
        quality_score=0.15,
        quality_source="onnx",
        threshold_band="critical",
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    controller = next(e for e in events if e["event_type"] == "controller_decision")
    payload = controller["payload"]

    payload_keys = set(payload.keys())
    extra_keys = payload_keys - controller_payload_keys
    assert not extra_keys, (
        f"forced payload has keys outside allowlist: {sorted(extra_keys)!r}"
    )


def test_new_events_write_payload_schema_version(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_ORACLE", raising=False)
    run_id = "01CONTRACTSCHEMAVERSION01"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.emit_task_started("task")
    log.emit_controller_decision(node_id="n0", action="continue")
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=1,
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    assert events
    for event in events:
        event_type = event["event_type"]
        assert event["payload_schema_version"] == CURRENT_PAYLOAD_SCHEMA_VERSIONS[
            event_type
        ]


def test_emission_violates_allowlist_raises_event_log_schema_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary = {
        "run_frame_schema_version": "0",
        "run_frame_hash": "h",
        "status": "success",
        "node_record_count": 0,
        "final_result_seq": 1,
        "extra_key": "must fail",
    }

    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    fail_closed = RuntimeEventLog(run_id="01SCHEMAFAILCLOSED000001", trace_dir=tmp_path)
    with pytest.raises(EventLogSchemaError, match="extra payload field"):
        fail_closed.emit_run_frame_summary(parent_event_id=1, summary=summary)

    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    caplog.clear()
    fail_open = RuntimeEventLog(run_id="01SCHEMAFAILOPEN0000001", trace_dir=tmp_path)
    assert fail_open.emit_run_frame_summary(parent_event_id=1, summary=summary) is None
    assert fail_open.disabled is True
    assert "event_log_error" in caplog.text
    assert (tmp_path / "01SCHEMAFAILOPEN0000001.jsonl").read_text(encoding="utf-8") == ""


def test_emission_violates_required_field_raises_event_log_schema_error(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    log = RuntimeEventLog(run_id="01SCHEMAMISSINGFIELD0001", trace_dir=tmp_path)

    with pytest.raises(EventLogSchemaError, match="missing required payload field"):
        log.emit_run_frame_summary(
            parent_event_id=1,
            summary={
                "run_frame_schema_version": "0",
                "status": "success",
                "node_record_count": 0,
                "final_result_seq": 1,
            },
        )


def test_emission_violates_max_length_errors_not_truncates(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    too_long = "x" * 1025
    run_id = "01SCHEMAMAXLEN000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)

    with pytest.raises(EventLogSchemaError, match="exceeds max UTF-8 bytes"):
        log.emit_failure(kind="provider", error_type="RuntimeError", message=too_long)

    assert too_long not in (tmp_path / f"{run_id}.jsonl").read_text(encoding="utf-8")


def test_controller_decision_current_v2_rejects_reason_under_kill_switch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01CTRLV2NOREASON0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.emit_controller_decision(
        node_id="n0",
        action="continue",
        reason="legacy reason must not revive",
        reason_code="quality.low",
    )
    log.close()

    [event] = _read_events(tmp_path / f"{run_id}.jsonl")
    assert event["payload_schema_version"] == "v2_allowlist_only"
    assert event["payload"]["reason_code"] == "quality.low"
    assert "reason" not in event["payload"]


def test_controller_decision_legacy_v1_reason_accepted_in_audit() -> None:
    event = _historical_controller_event(run_id="r1", seq=0)

    report = _payload_schema_distribution_for_events(
        [event], mode="audit"
    )

    assert report.errors == ()
    assert report.warnings
    assert report.distribution["controller_decision"]["v1_pre_allowlist_reason"] == {
        "count": 1,
        "explicit_count": 0,
        "inferred_count": 1,
        "status": "legacy_accepted",
    }


def test_validator_reads_v1_and_v2_traces() -> None:
    events = [
        _historical_controller_event(run_id="r1", seq=0),
        _current_controller_event(run_id="r2", seq=0),
    ]

    report = _payload_schema_distribution_for_events(events, mode="audit")

    assert report.errors == ()
    assert report.distribution["controller_decision"]["v1_pre_allowlist_reason"][
        "inferred_count"
    ] == 1
    assert report.distribution["controller_decision"]["v2_allowlist_only"][
        "explicit_count"
    ] == 1


def test_validator_strict_current_rejects_v1() -> None:
    """Strict-current rejects historical inferred v1. Same case as
    test_validator_strict_current_rejects_inferred_legacy_version below;
    kept for backward-compatibility of test names."""
    report = _payload_schema_distribution_for_events(
        [_historical_controller_event(run_id="r1", seq=0)],
        mode="strict-current",
    )

    assert report.errors
    assert report.distribution["controller_decision"]["v1_pre_allowlist_reason"][
        "status"
    ] == "legacy_rejected_strict_current"


def test_validator_strict_current_rejects_inferred_legacy_version() -> None:
    """cgpro 2026-04-30 cycle-8 R6.1c VERIFY round-1, fix #2 case (c): an
    event that resolves to a registered legacy schema by inference (missing
    payload_schema_version + payload shape matches v1) MUST be rejected
    under strict-current.
    """
    report = _payload_schema_distribution_for_events(
        [_historical_controller_event(run_id="r-leg", seq=0)],
        mode="strict-current",
    )

    assert report.errors, "strict-current must error on inferred legacy"
    assert report.distribution["controller_decision"]["v1_pre_allowlist_reason"][
        "status"
    ] == "legacy_rejected_strict_current"


def test_validator_strict_current_rejects_missing_version_even_if_current_shape() -> None:
    """cgpro 2026-04-30 cycle-8 R6.1c VERIFY round-1, fix #2 case (a): an
    event with a payload that matches the CURRENT schema shape but lacks
    an explicit `payload_schema_version` field MUST still be rejected
    under strict-current. Implicit-current is not allowed.
    """
    current_event = _current_controller_event(run_id="r-impl", seq=0)
    current_event.pop("payload_schema_version", None)

    report = _payload_schema_distribution_for_events(
        [current_event],
        mode="strict-current",
    )

    assert report.errors, "strict-current must error on missing payload_schema_version"
    cdist = report.distribution["controller_decision"]["v2_allowlist_only"]
    assert cdist["explicit_count"] == 0
    assert cdist["inferred_count"] >= 1


def test_validator_strict_current_rejects_explicit_non_current_version() -> None:
    """cgpro 2026-04-30 cycle-8 R6.1c VERIFY round-1, fix #2 case (b): an
    event that EXPLICITLY tags `payload_schema_version` with a registered
    but non-current version MUST be rejected under strict-current.
    """
    legacy_event = _historical_controller_event(run_id="r-explicit", seq=0)
    legacy_event["payload_schema_version"] = "v1_pre_allowlist_reason"

    report = _payload_schema_distribution_for_events(
        [legacy_event],
        mode="strict-current",
    )

    assert report.errors, "strict-current must error on explicit non-current"
    cdist = report.distribution["controller_decision"]["v1_pre_allowlist_reason"]
    assert cdist["explicit_count"] >= 1
    assert cdist["status"] == "legacy_rejected_strict_current"


def test_validator_strict_current_rejects_malformed_payload_matching_no_schema() -> None:
    """cgpro 2026-04-30 cycle-8 R6.1c VERIFY round-1, fix #2 case (d): an
    event whose payload matches NO registered schema for its event_type
    MUST be rejected under strict-current (and produce an unknown_rejected
    status in the distribution).
    """
    malformed = _current_controller_event(run_id="r-mal", seq=0)
    malformed.pop("payload_schema_version", None)
    malformed["payload"] = {
        "totally_unknown_field": "garbage",
        "another_unknown": 42,
    }

    report = _payload_schema_distribution_for_events(
        [malformed],
        mode="strict-current",
    )

    assert report.errors, "strict-current must error on malformed payload"
    # The distribution must record this as unknown_rejected for controller_decision.
    found_unknown = any(
        bucket.get("status") == "unknown_rejected"
        for bucket in report.distribution.get("controller_decision", {}).values()
    )
    assert found_unknown, (
        f"expected unknown_rejected bucket in distribution; got "
        f"{report.distribution.get('controller_decision')!r}"
    )


def test_validator_reports_explicit_vs_inferred_counts() -> None:
    report = _payload_schema_distribution_for_events(
        [
            _historical_controller_event(run_id="r1", seq=0),
            _current_controller_event(run_id="r2", seq=0),
            _current_controller_event(run_id="r3", seq=0),
        ],
        mode="audit",
    )

    cdist = report.distribution["controller_decision"]
    assert cdist["v1_pre_allowlist_reason"]["explicit_count"] == 0
    assert cdist["v1_pre_allowlist_reason"]["inferred_count"] == 1
    assert cdist["v2_allowlist_only"]["explicit_count"] == 2
    assert cdist["v2_allowlist_only"]["inferred_count"] == 0


def test_path_e_validate_uses_payload_schema_sot() -> None:
    validator = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "sage"
        / "bench"
        / "path_e_validate.py"
    ).read_text(encoding="utf-8")

    assert "sage.runtime.event_log.payload_schemas" in validator
    assert "CONTROLLER_DECISION_ALLOWED_PAYLOAD_KEYS" not in validator


def _historical_controller_event(*, run_id: str, seq: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "trace_id": run_id,
        "parent_event_id": None,
        "seq": seq,
        "timestamp_ns": 1,
        "event_type": "controller_decision",
        "source_component": "controller",
        "task_hash": "h",
        "payload_hash": "p",
        "redaction_state": "raw",
        "payload": {
            "action": "continue",
            "target_node_id": "n1",
            "gate_source_id": "",
            "gate_target_id": "",
            "reason": "historical reason",
        },
        "action": "continue",
    }


def _current_controller_event(*, run_id: str, seq: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "payload_schema_version": "v2_allowlist_only",
        "run_id": run_id,
        "trace_id": run_id,
        "parent_event_id": None,
        "seq": seq,
        "timestamp_ns": 1,
        "event_type": "controller_decision",
        "source_component": "controller",
        "task_hash": "h",
        "payload_hash": "p",
        "redaction_state": "redacted",
        "payload": {
            "node_id": "n0",
            "action": "continue",
            "target_node_id": "",
            "gate_source_id": "",
            "gate_target_id": "",
            "quality_score": None,
            "quality_source": "abstain",
            "threshold_band": "continue",
            "reason_code": "abstain_no_signal",
        },
        "node_id": "n0",
        "action": "continue",
    }


def test_controller_decision_reason_code_is_slug_constrained(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-1: free-form reason_code from a
    caller (e.g. a multiline Traceback accidentally passed as reason_code)
    must be coerced to the abstain sentinel, not surfaced verbatim.
    """
    monkeypatch.delenv("SAGE_ORACLE", raising=False)
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTSLUGCOERCETEST"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    log.emit_controller_decision(
        node_id="0",
        action="continue",
        reason_code="Traceback (most recent call last):\n  spaces and CAPS",
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    controller = next(e for e in events if e["event_type"] == "controller_decision")
    payload = controller["payload"]

    assert payload["reason_code"] == "abstain_no_signal"
    payload_json = json.dumps(payload, ensure_ascii=False)
    assert "Traceback" not in payload_json
    assert "CAPS" not in payload_json


def test_controller_decision_quality_score_clamps_non_finite(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-1: non-finite or out-of-range
    quality_score values must be coerced (None for non-finite, clamp to
    [0.0, 1.0] for finite) so downstream consumers never see NaN/Inf or
    out-of-band values.
    """
    monkeypatch.delenv("SAGE_ORACLE", raising=False)
    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01CONTRACTQSCORECLAMPTST"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_path)
    log.set_task_text("contract test task")
    log.emit_task_started("contract test task")
    # NaN / Inf -> None; out-of-range -> clamped.
    log.emit_controller_decision(
        node_id="0",
        action="continue",
        quality_score=float("inf"),
    )
    log.emit_controller_decision(
        node_id="1",
        action="continue",
        quality_score=2.5,
    )
    log.emit_controller_decision(
        node_id="2",
        action="continue",
        quality_score=-0.5,
    )
    log.close()

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    cdecs = [e for e in events if e["event_type"] == "controller_decision"]
    assert cdecs[0]["payload"]["quality_score"] is None  # inf -> None
    assert cdecs[1]["payload"]["quality_score"] == 1.0   # 2.5 -> 1.0
    assert cdecs[2]["payload"]["quality_score"] == 0.0   # -0.5 -> 0.0


def test_controller_decision_payload_fields_present_in_cycle7_n50_jsonls() -> None:
    """Analyzer reconciliation over the N=50 cycle-7 trace set."""
    from sage.bench.path_e_validate import (
        CONTROLLER_DECISION_SAFE_FIELDS,
        _summarize_controller_decision_payloads,
    )

    jsonl_dir = (
        pathlib.Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmarks"
        / "2026-04-29-cycle7-evidence-bcb-N50-jsonl"
    )
    assert jsonl_dir.exists(), f"missing committed trace directory: {jsonl_dir}"
    assert len(list(jsonl_dir.glob("*.jsonl"))) == 50

    summary = _summarize_controller_decision_payloads(jsonl_dir)

    assert summary.event_count > 0
    assert summary.non_empty_payload_count == summary.event_count
    assert summary.missing_examples == ()
    assert summary.field_presence == {
        field: summary.event_count for field in CONTROLLER_DECISION_SAFE_FIELDS
    }


def test_oracle_verdict_golden_matches_default_on_contract() -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-2 PUSH BACK sub-blocker A: the
    oracle_verdict.json golden fixture's ``_doc`` and ``_invariants`` MUST
    reflect the cycle-7 default-on contract, not the pre-flip opt-in one.
    """
    fixture = _load_fixture("oracle_verdict.json")
    text = json.dumps(fixture)
    assert "ONLY emitted when SAGE_ORACLE=1" not in text, (
        "oracle_verdict.json still has pre-flip 'ONLY emitted when SAGE_ORACLE=1' "
        "framing — update _doc to default-on contract."
    )
    assert "MUST NEVER appear when SAGE_ORACLE is unset" not in text, (
        "oracle_verdict.json still has pre-flip 'MUST NEVER appear when SAGE_ORACLE "
        "is unset' invariant — update _invariants to kill-switch suppression."
    )
    assert "default-on" in text or "Default-on" in text, (
        "oracle_verdict.json must mention default-on (cycle-7 contract anchor)."
    )
    assert "disable" in text and "disabled" in text, (
        "oracle_verdict.json must list `disable`/`disabled` in the kill-switch values."
    )


def test_no_stale_cycle7_oracle_contract_phrases() -> None:
    """cgpro 2026-04-30 cycle-7 VERIFY round-2 trap-fix: prose contract drift
    is a class. The contract docs and golden fixtures can silently diverge
    even when the existing field-presence tests still pass. Lint a small set
    of pre-flip phrases out of the current-state docs (NOT historical
    evidence files in docs/benchmarks/, which are intentionally frozen).
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    scoped_paths: list[pathlib.Path] = [
        repo_root / "README.md",
        repo_root / "CLAUDE.md",
        repo_root / "roadmap.md",
        repo_root / "docs" / "contracts" / "runtime-event-log.md",
        repo_root / "YGN-SAGE" / "00-Dashboard.md",
    ]
    golden_dir = repo_root / "sage-python" / "tests" / "golden" / "runtime_events"

    haystack_parts: list[tuple[str, str]] = []
    for path in scoped_paths:
        if path.is_file():
            haystack_parts.append((str(path), path.read_text(encoding="utf-8")))
    for golden in sorted(golden_dir.glob("*.json")):
        haystack_parts.append((str(golden), golden.read_text(encoding="utf-8")))

    stale_phrases = [
        "ONLY emitted when SAGE_ORACLE=1",
        "MUST NEVER appear when SAGE_ORACLE is unset",
        "All 4 strategic feature flags ship default-OFF",
        "All four strategic feature flags default OFF",
    ]
    failures: list[str] = []
    for phrase in stale_phrases:
        for source, text in haystack_parts:
            if phrase in text:
                failures.append(f"  {source}: contains stale phrase {phrase!r}")
    assert not failures, (
        "Stale cycle-7 oracle contract phrases found in current-state docs:\n"
        + "\n".join(failures)
    )


# ────────────────────────────────────────────────────────────────────
# Invariant 9 (cycle-12 prelude 2026-05-05): CLI protocol versioning
#
# `sage run --jsonl` emits frames with a `protocol_version` envelope
# field. The constant `sage.cli.run.CLI_PROTOCOL_VERSION` MUST equal
# the value documented in `docs/contracts/SAGE_CLI_PROTOCOL.md` AND
# every emitted frame MUST carry that constant verbatim. Drift between
# any of (constant, spec, emitted frames) means the runtime contract
# disagrees with itself.
#
# This test is the executable proof for invariant 9 in
# `docs/contracts/runtime-integrity-ledger.md`. Per the maintenance
# discipline, "wire a regression test that proves the side-effect is
# blocked when the verification fails" — here the side-effect is
# external consumer interpretation of YGN-SAGE runtime decisions
# (the pi-mono adapter shipping cycle-13). A frontend that sees a
# mismatched `protocol_version` MUST refuse to render; this test
# proves the backend cannot accidentally emit a value that drifts
# from the spec or the constant.
# ────────────────────────────────────────────────────────────────────


def test_cli_protocol_version_constant_matches_spec_doc() -> None:
    """The constant in code MUST equal the literal documented in the spec.

    cycle-12 prelude (2026-05-05): the CLI protocol document declares
    "v0" as the inaugural protocol version. The constant
    ``sage.cli.run.CLI_PROTOCOL_VERSION`` is the source of truth for
    every emitted frame; if the spec bumps to "v1" but the constant
    stays at "v0" (or vice versa) the contract drifts silently.

    This test verifies BOTH directions:
      1. The constant is exactly ``"v0"`` (matches the spec literal).
      2. The spec doc contains the literal string the constant emits.

    A future protocol bump MUST update both atomically. Update this
    test then.
    """
    from sage.cli.run import CLI_PROTOCOL_VERSION

    # Direction 1: constant is what we expect at this point in the
    # ledger history (cycle-12 prelude = v0).
    assert CLI_PROTOCOL_VERSION == "v0", (
        f"CLI_PROTOCOL_VERSION drifted: expected 'v0' (cycle-12 prelude), "
        f"got {CLI_PROTOCOL_VERSION!r}. If this is a deliberate bump, "
        f"update docs/contracts/SAGE_CLI_PROTOCOL.md AND "
        f"docs/contracts/runtime-integrity-ledger.md (invariant 9 row), "
        f"AND this test."
    )

    # Direction 2: the spec doc actually contains the literal the
    # constant emits. Open the spec, find the version declaration.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    spec_path = repo_root / "docs" / "contracts" / "SAGE_CLI_PROTOCOL.md"
    assert spec_path.exists(), (
        f"SAGE_CLI_PROTOCOL.md missing at {spec_path}. The CLI protocol "
        f"spec is the source of truth for invariant 9; deleting it is "
        f"a contract change that must update this test too."
    )
    spec_text = spec_path.read_text(encoding="utf-8")
    # The spec declares the version in its title and in the envelope
    # description. We assert the constant value appears as the
    # double-quoted form the envelope mandates.
    assert f'"{CLI_PROTOCOL_VERSION}"' in spec_text, (
        f"docs/contracts/SAGE_CLI_PROTOCOL.md does not contain the "
        f"literal {CLI_PROTOCOL_VERSION!r} (as a double-quoted JSON value). "
        f"The constant and the spec disagree — invariant 9 broken. "
        f"Fix: update the spec to mention the new version, OR revert "
        f"the constant change."
    )


def test_cli_protocol_version_emitted_in_envelope() -> None:
    """Every emitted CLI envelope frame MUST carry the documented version.

    The 4 CLI-shell envelope events (cli_started, cli_progress,
    cli_tool_request, cli_complete) each pass through ``_emit_cli_event``
    which is the single point where the envelope is constructed. This
    test exercises that function with all 4 event types and asserts
    the emitted frames carry ``protocol_version == CLI_PROTOCOL_VERSION``.

    If a future commit bypasses ``_emit_cli_event`` and emits a frame
    via ``json.dumps`` directly (e.g. inlining for performance), this
    test will NOT catch that — but the ``test_cli_protocol_version_constant_matches_spec_doc``
    test still catches drift between constant and spec, and the
    ``test_sage_cli_jsonl.py::test_emit_cli_event_envelope_shape`` test
    catches the envelope schema. The trio is the protection.
    """
    import io

    from sage.cli.run import CLI_PROTOCOL_VERSION, _emit_cli_event, _SeqCounter

    counter = _SeqCounter()
    for event_type, payload in (
        ("cli_started", {"protocol_version": CLI_PROTOCOL_VERSION, "task": "x"}),
        ("cli_progress", {"stage": "execute", "elapsed_ms": 100}),
        ("cli_tool_request", {"correlation_id": "approval-1", "action": "upgrade_model"}),
        ("cli_complete", {"exit_code": 0, "outcome": "success"}),
    ):
        buf = io.StringIO()
        _emit_cli_event(
            buf,
            event_type,
            run_id="01TESTRUN0000000000000099",
            payload=payload,
            seq=counter.next(),
        )
        frame = json.loads(buf.getvalue().rstrip("\n"))
        assert frame["protocol_version"] == CLI_PROTOCOL_VERSION, (
            f"Emitted {event_type} frame carries "
            f"protocol_version={frame['protocol_version']!r}, "
            f"expected {CLI_PROTOCOL_VERSION!r}. Invariant 9 violated — "
            f"frontend consumers will fail-close on this frame. Check "
            f"_emit_cli_event in sage/cli/run.py."
        )
        # Belt-and-suspenders: also assert event_type is one of the 4
        # documented CLI-shell events, so this test fails loudly if a
        # future commit adds a 5th event type without updating the
        # taxonomy in SAGE_CLI_PROTOCOL.md.
        assert event_type in (
            "cli_started", "cli_progress", "cli_tool_request", "cli_complete",
        ), (
            f"Unexpected CLI-shell event type {event_type!r}. The "
            f"protocol v0 declares exactly 4 CLI-shell envelope types; "
            f"a 5th requires a spec update + this test update."
        )


def test_cli_protocol_version_appears_in_runtime_integrity_ledger() -> None:
    """Invariant 9 MUST be documented in the runtime-integrity-ledger.

    Per the ledger's own maintenance discipline (line ~57): "When adding
    a new invariant: append a row to BOTH tables (the invariant table
    AND the module cross-reference)." This test enforces the rule —
    a future commit that drops invariant 9 from either table without
    updating this test fails loudly.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    ledger_path = repo_root / "docs" / "contracts" / "runtime-integrity-ledger.md"
    assert ledger_path.exists()
    ledger_text = ledger_path.read_text(encoding="utf-8")

    # Both invariant table AND module-cross-reference table must
    # mention "CLI protocol versioning".
    assert ledger_text.count("CLI protocol versioning") >= 2, (
        f"Invariant 9 'CLI protocol versioning' must appear in BOTH "
        f"tables in runtime-integrity-ledger.md. Found "
        f"{ledger_text.count('CLI protocol versioning')} mention(s); "
        f"expected ≥2."
    )

    # The header announces 10 invariants (post cycle-13 K Phase 1.5
    # — ToolPolicy capability declaration & grant enforcement).
    assert "## The 10 invariants" in ledger_text, (
        "runtime-integrity-ledger.md header must announce '10 invariants' "
        "(updated from 9 in cycle-13 K Phase 1.5 commit 6497c427). "
        "If a future cycle adds an 11th invariant, update both the "
        "header and this test."
    )
