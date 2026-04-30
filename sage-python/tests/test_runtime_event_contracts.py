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
    from sage.bench.path_e_validate import CONTROLLER_DECISION_ALLOWED_PAYLOAD_KEYS

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
    extra_keys = payload_keys - CONTROLLER_DECISION_ALLOWED_PAYLOAD_KEYS
    assert not extra_keys, (
        f"forced payload has keys outside allowlist: {sorted(extra_keys)!r}"
    )


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
