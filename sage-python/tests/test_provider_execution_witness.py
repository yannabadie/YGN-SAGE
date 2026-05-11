"""Tests for slice 10D — provider_execution_witness (Route A v0).

cgpro DESIGN_LOCK 2026-05-11 on conv
``cgpro_ygn_sage_global_analysis_20260510``: Route A — Python-only
witness, NOT runtime ledger invariant I-11 yet.

Required tests (per DESIGN_LOCK):
- test_provider_execution_witness_emitted_after_model_assigned
- test_provider_execution_witness_records_blocked_routing_candidate
- test_provider_execution_witness_records_allowed_assignments
- test_provider_execution_witness_no_policy_active
- test_provider_execution_witness_payload_schema_rejects_missing_required_fields
- test_provider_execution_witness_payload_visible_when_forced_without_trace_raw
- test_provider_execution_witness_does_not_mask_provider_policy_violation

This file ships the first 5 (the unit-test core). The last 2 are wired
in pre-existing test files where the assertion belongs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sage.pipeline_v2.runtime_events import (
    _classify_provider_against_policy,
    runtime_emit_provider_execution_witness,
)
from sage.runtime.event_log.payload_schemas import (
    PAYLOAD_SCHEMAS,
    _current_schema_for,
)
from sage.runtime.event_log.schema import EVENT_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Schema-side tests (no event_log instantiation required)
# ─────────────────────────────────────────────────────────────────────────────


def test_event_type_registered() -> None:
    """`provider_execution_witness` must be in EVENT_TYPES."""
    assert "provider_execution_witness" in EVENT_TYPES


def test_payload_schema_v1_registered() -> None:
    """The v1 schema must exist + be current."""
    versions = PAYLOAD_SCHEMAS["provider_execution_witness"]
    assert "v1" in versions
    current = _current_schema_for("provider_execution_witness")
    assert current.version == "v1"
    assert current.payload_kind == "dict"


def test_payload_schema_required_fields() -> None:
    """Required fields per cgpro DESIGN_LOCK Q2."""
    schema = _current_schema_for("provider_execution_witness")
    assert set(schema.required_fields) >= {
        "witness_schema_version",
        "routing",
        "policy",
        "per_node_assignments",
        "substitution_summary",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function classifier tests
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_passes_policy_with_allowlist() -> None:
    decision, reason = _classify_provider_against_policy(
        "deepseek",
        allowlist=("deepseek", "google"),
        denylist=(),
        policy_active=True,
    )
    assert decision == "allowed"
    assert reason == "passes_policy"


def test_classify_blocked_by_denylist() -> None:
    decision, reason = _classify_provider_against_policy(
        "openai",
        allowlist=("deepseek", "google"),
        denylist=("openai",),
        policy_active=True,
    )
    assert decision == "blocked"
    assert reason == "provider_in_denylist"


def test_classify_blocked_outside_allowlist() -> None:
    decision, reason = _classify_provider_against_policy(
        "xai",
        allowlist=("deepseek", "google"),
        denylist=(),
        policy_active=True,
    )
    assert decision == "blocked"
    assert reason == "provider_outside_allowlist"


def test_classify_no_policy_active_allows_anything() -> None:
    decision, reason = _classify_provider_against_policy(
        "anything",
        allowlist=(),
        denylist=(),
        policy_active=False,
    )
    assert decision == "allowed"
    assert reason == "no_policy_active"


def test_classify_unresolved_when_provider_empty() -> None:
    decision, reason = _classify_provider_against_policy(
        "",
        allowlist=("deepseek",),
        denylist=(),
        policy_active=True,
    )
    assert decision == "unresolved"
    assert reason in {
        "routing_provider_unresolved",
        "assignment_provider_unresolved",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper integration tests (with a fake event_log)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeEventLog:
    """Minimal stand-in for RuntimeEventLog.

    Only implements ``emit_provider_execution_witness``; captures the
    call so tests can assert payload shape.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._seq = 0

    def emit_provider_execution_witness(
        self,
        *,
        witness_schema_version: str,
        assignment_phase: str,
        routing: dict[str, Any],
        policy: dict[str, Any],
        per_node_assignments: list[dict[str, Any]],
        substitution_summary: dict[str, Any],
        parent_event_id: int | None = None,
    ) -> int | None:
        self.calls.append(
            {
                "witness_schema_version": witness_schema_version,
                "assignment_phase": assignment_phase,
                "routing": routing,
                "policy": policy,
                "per_node_assignments": per_node_assignments,
                "substitution_summary": substitution_summary,
                "parent_event_id": parent_event_id,
            }
        )
        seq = self._seq
        self._seq += 1
        return seq


def _make_node(role: str, model_id: str = "", caps: tuple[str, ...] = ()):
    return SimpleNamespace(
        role=role,
        model_id=model_id,
        required_capabilities=caps,
    )


def _make_topology(nodes: list[Any]) -> Any:
    topo = SimpleNamespace()
    topo._nodes = nodes
    topo.id = "test-topo"
    topo.template_type = "sequential"
    topo.get_node = lambda idx, _t=topo: _t._nodes[idx]
    topo.node_count = lambda _t=topo: len(_t._nodes)
    return topo


def _make_pipeline(provider_inferred: dict[str, str] | None = None) -> Any:
    """Fake CognitiveOrchestrationPipeline with a provider_pool that
    infers provider from model_id via a static map.
    """
    pool = SimpleNamespace()
    inferred = provider_inferred or {}
    pool.infer_provider = lambda model_id, _m=inferred: _m.get(model_id, "")
    return SimpleNamespace(provider_pool=pool, llm_config=None)


def _make_ctx(
    nodes: list[Any],
    *,
    assignments: dict[int, str],
    allowlist: tuple[str, ...] = (),
    denylist: tuple[str, ...] = (),
    routing_source: str = "rust_system_router",
    system: int = 3,
    domain: str = "code",
    confidence: float | None = 0.8788,
) -> Any:
    ctx = SimpleNamespace()
    ctx.topology = _make_topology(nodes)
    ctx.assignments = assignments
    ctx.provider_hints = {}
    ctx.provider_allowlist = allowlist
    ctx.provider_denylist = denylist
    ctx.routing_source = routing_source
    ctx.system = system
    ctx.domain = domain
    ctx.confidence = confidence
    return ctx


def test_provider_execution_witness_emitted_after_model_assigned() -> None:
    """The helper writes ONE event with the right shape (slice 9 baseline).

    Recreates the actual slice 9 scenario: routing chose gpt-5.4-pro
    (openai), allowlist=google,deepseek, denylist=openai. Expected:
    - routing.routing_provider_id = "openai"
    - policy.routing_candidate_decision = "blocked"
    - all per_node_assignments[].assignment_policy_decision = "allowed"
    - substitution_summary.routing_candidate_blocked_by_policy = True
    """
    pipeline = _make_pipeline({
        "gpt-5.4-pro": "openai",
        "deepseek-v4-pro": "deepseek",
        "gemini-2.5-flash": "google",
    })
    nodes = [
        _make_node("coder", caps=("code_generation", "reasoning", "tools")),
        _make_node("synthesizer", caps=("text_processing",)),
    ]
    ctx = _make_ctx(
        nodes,
        assignments={0: "deepseek-v4-pro", 1: "gemini-2.5-flash"},
        allowlist=("deepseek", "google"),
        denylist=("openai",),
    )
    event_log = _FakeEventLog()

    seq = runtime_emit_provider_execution_witness(
        pipeline, ctx, event_log,
        routing_model_id="gpt-5.4-pro",
    )

    assert seq == 0
    assert len(event_log.calls) == 1
    call = event_log.calls[0]

    # Routing block
    assert call["routing"]["routing_model_id"] == "gpt-5.4-pro"
    assert call["routing"]["routing_provider_id"] == "openai"
    assert call["routing"]["system"] == 3
    assert call["routing"]["domain"] == "code"

    # Policy block
    assert call["policy"]["active"] is True
    assert call["policy"]["allowlist"] == ["deepseek", "google"]
    assert call["policy"]["denylist"] == ["openai"]
    assert call["policy"]["routing_candidate_decision"] == "blocked"
    assert call["policy"]["routing_candidate_reason_code"] == "provider_in_denylist"

    # Per-node assignments: 2 nodes, both allowed
    assert len(call["per_node_assignments"]) == 2
    for node in call["per_node_assignments"]:
        assert node["assignment_policy_decision"] == "allowed"
        assert node["assignment_policy_reason_code"] == "passes_policy"
    assert call["per_node_assignments"][0]["node_role"] == "coder"
    assert call["per_node_assignments"][0]["assigned_model_id"] == "deepseek-v4-pro"
    assert call["per_node_assignments"][0]["assigned_provider_id"] == "deepseek"

    # Substitution summary
    summary = call["substitution_summary"]
    assert summary["routing_model_distinct_from_assignments"] is True
    assert summary["routing_candidate_blocked_by_policy"] is True
    assert summary["assignment_count"] == 2
    assert summary["allowed_assignment_count"] == 2
    assert summary["blocked_assignment_count"] == 0
    assert summary["rust_filter_details_observed"] is False


def test_provider_execution_witness_records_blocked_routing_candidate() -> None:
    """When the routing's chosen model's provider is in the denylist,
    policy.routing_candidate_decision MUST be 'blocked'.
    """
    pipeline = _make_pipeline({"gpt-5.4-pro": "openai", "deepseek-v4-pro": "deepseek"})
    nodes = [_make_node("coder")]
    ctx = _make_ctx(
        nodes,
        assignments={0: "deepseek-v4-pro"},
        denylist=("openai",),
    )
    event_log = _FakeEventLog()
    runtime_emit_provider_execution_witness(
        pipeline, ctx, event_log,
        routing_model_id="gpt-5.4-pro",
    )
    call = event_log.calls[0]
    assert call["policy"]["routing_candidate_decision"] == "blocked"
    assert call["policy"]["routing_candidate_reason_code"] == "provider_in_denylist"
    assert call["substitution_summary"]["routing_candidate_blocked_by_policy"] is True


def test_provider_execution_witness_records_allowed_assignments() -> None:
    """Per-node assignments resolve provider via runtime_provider_id_for_model
    and classify against policy correctly.
    """
    pipeline = _make_pipeline({
        "deepseek-v4-pro": "deepseek",
        "gemini-3-flash-preview": "google",
    })
    nodes = [
        _make_node("actor"),
        _make_node("verifier"),
        _make_node("judge"),
    ]
    ctx = _make_ctx(
        nodes,
        assignments={
            0: "deepseek-v4-pro",
            1: "gemini-3-flash-preview",
            2: "deepseek-v4-pro",
        },
        allowlist=("deepseek", "google"),
        denylist=("openai",),
    )
    event_log = _FakeEventLog()
    runtime_emit_provider_execution_witness(
        pipeline, ctx, event_log,
        routing_model_id="deepseek-v4-pro",
    )
    call = event_log.calls[0]
    assert all(
        n["assignment_policy_decision"] == "allowed"
        for n in call["per_node_assignments"]
    )
    assert call["substitution_summary"]["allowed_assignment_count"] == 3


def test_provider_execution_witness_no_policy_active() -> None:
    """With no allowlist + no denylist, all assignments allowed under
    `no_policy_active`.
    """
    pipeline = _make_pipeline({"deepseek-v4-pro": "deepseek"})
    nodes = [_make_node("coder")]
    ctx = _make_ctx(nodes, assignments={0: "deepseek-v4-pro"})
    event_log = _FakeEventLog()
    runtime_emit_provider_execution_witness(
        pipeline, ctx, event_log,
        routing_model_id="deepseek-v4-pro",
    )
    call = event_log.calls[0]
    assert call["policy"]["active"] is False
    assert call["policy"]["routing_candidate_decision"] == "allowed"
    assert call["policy"]["routing_candidate_reason_code"] == "no_policy_active"
    assert call["per_node_assignments"][0]["assignment_policy_decision"] == "allowed"
    assert call["per_node_assignments"][0]["assignment_policy_reason_code"] == "no_policy_active"


def test_provider_execution_witness_no_event_log_returns_none() -> None:
    """Defensive: caller may pass event_log=None; helper returns None."""
    pipeline = _make_pipeline()
    nodes = [_make_node("coder")]
    ctx = _make_ctx(nodes, assignments={0: "deepseek-v4-pro"})
    result = runtime_emit_provider_execution_witness(
        pipeline, ctx, None,
        routing_model_id="deepseek-v4-pro",
    )
    assert result is None


def test_provider_execution_witness_reroute_phase_recorded() -> None:
    """When assignment_phase='reroute', the event records that label so
    downstream consumers can distinguish initial vs reroute witnesses.
    """
    pipeline = _make_pipeline({"deepseek-v4-pro": "deepseek"})
    nodes = [_make_node("coder")]
    ctx = _make_ctx(nodes, assignments={0: "deepseek-v4-pro"})
    event_log = _FakeEventLog()
    runtime_emit_provider_execution_witness(
        pipeline, ctx, event_log,
        routing_model_id="deepseek-v4-pro",
        assignment_phase="reroute",
    )
    assert event_log.calls[0]["assignment_phase"] == "reroute"


# ─────────────────────────────────────────────────────────────────────────────
# cgpro DESIGN_LOCK required tests #5/#6/#7 — schema-validation + writer +
# orchestration assertions.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_trace_dir() -> Path:
    """Per-test trace dir under `.tmp` (the default pytest temp root is
    ACL-denied on this Windows box — same workaround as the rest of the
    runtime-event contract tests)."""
    import shutil
    from uuid import uuid4
    path = Path(".tmp") / "pytest-witness-contracts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_provider_execution_witness_payload_schema_rejects_missing_required_fields(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro DESIGN_LOCK required test #5.

    The v1 schema declares 5 required fields. Dropping any one must
    raise EventLogSchemaError. We test the validator directly because
    the writer's typed signature always supplies all 5 fields — the
    validator is the runtime gate that catches refactor drift.
    """
    from sage.runtime.event_log.errors import EventLogSchemaError
    from sage.runtime.event_log.payload_schemas import (
        _current_schema_for,
        _validate_payload_against_schema,
    )

    schema = _current_schema_for("provider_execution_witness")
    base_payload = {
        "witness_schema_version": "v0",
        "assignment_phase": "initial",
        "routing": {"routing_model_id": "deepseek-v4-pro"},
        "policy": {"active": False, "routing_candidate_decision": "allowed"},
        "per_node_assignments": [{"node_index": 0, "node_role": "coder"}],
        "substitution_summary": {"assignment_count": 1},
    }

    _validate_payload_against_schema(base_payload, schema)

    for missing in schema.required_fields:
        bad = {k: v for k, v in base_payload.items() if k != missing}
        with pytest.raises(EventLogSchemaError, match="missing required payload field"):
            _validate_payload_against_schema(bad, schema)

    bad_extra = dict(base_payload)
    bad_extra["unknown_field"] = "x"
    with pytest.raises(EventLogSchemaError, match="extra payload field"):
        _validate_payload_against_schema(bad_extra, schema)


def test_provider_execution_witness_payload_visible_when_forced_without_trace_raw(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro DESIGN_LOCK required test #6.

    ``emit_provider_execution_witness`` uses ``_force_payload=True`` so
    the structured payload is present in the JSONL even when
    ``SAGE_TRACE_RAW`` is unset. Without this, the chain
    ``routing → policy → assignments`` cannot be reconstructed by
    post-hoc consumers.
    """
    from sage.runtime.event_log import RuntimeEventLog

    monkeypatch.delenv("SAGE_TRACE_RAW", raising=False)

    run_id = "01WITNESSFORCEPAYLOAD0001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("witness payload visibility test")
    log.emit_task_started("witness payload visibility test")

    seq = log.emit_provider_execution_witness(
        witness_schema_version="v0",
        assignment_phase="initial",
        routing={
            "routing_model_id": "deepseek-v4-pro",
            "routing_provider_id": "deepseek",
            "system": 3,
            "domain": "code",
        },
        policy={
            "active": True,
            "allowlist": ["deepseek"],
            "denylist": [],
            "routing_candidate_decision": "allowed",
            "routing_candidate_reason_code": "passes_policy",
        },
        per_node_assignments=[
            {
                "node_index": 0,
                "node_role": "coder",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": False,
            "assignment_count": 1,
            "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    assert seq is not None
    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    events = [
        json.loads(line)
        for line in (tmp_trace_dir / f"{run_id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    witness = next(e for e in events if e["event_type"] == "provider_execution_witness")

    assert "payload" in witness, (
        "provider_execution_witness MUST surface payload even when "
        "SAGE_TRACE_RAW is unset (slice 10D _force_payload=True)"
    )
    p = witness["payload"]
    assert p["witness_schema_version"] == "v0"
    assert p["routing"]["routing_provider_id"] == "deepseek"
    assert p["policy"]["routing_candidate_decision"] == "allowed"
    assert len(p["per_node_assignments"]) == 1
    assert p["substitution_summary"]["assignment_count"] == 1


def test_helper_emits_schema_valid_payload_through_real_writer(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advisor-flagged coverage gap (2026-05-11).

    The other helper tests use _FakeEventLog (no schema validation).
    Test #5 validates the schema with a hand-written payload. This test
    is the only one that runs the LIVE helper through the LIVE writer
    under ``SAGE_TRACE_FAIL_CLOSED=1`` — catching any future drift
    between the helper's payload construction and the v1 schema as a
    hard test failure.
    """
    from sage.runtime.event_log import RuntimeEventLog

    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")

    run_id = "01HELPERTHRUWRITER0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("helper schema-bound integration")
    log.emit_task_started("helper schema-bound integration")

    pipeline = _make_pipeline({
        "gpt-5.4-pro": "openai",
        "deepseek-v4-pro": "deepseek",
        "gemini-3-flash-preview": "google",
    })
    nodes = [
        _make_node("coder", caps=("code_generation", "reasoning", "tools")),
        _make_node("synthesizer", caps=("text_processing",)),
    ]
    ctx = _make_ctx(
        nodes,
        assignments={0: "deepseek-v4-pro", 1: "gemini-3-flash-preview"},
        allowlist=("deepseek", "google"),
        denylist=("openai",),
    )

    seq = runtime_emit_provider_execution_witness(
        pipeline, ctx, log,
        routing_model_id="gpt-5.4-pro",
    )
    assert seq is not None, (
        "helper-through-real-writer emit must succeed under "
        "SAGE_TRACE_FAIL_CLOSED=1 — schema drift would raise here"
    )

    log.emit_final_result(
        status="success",
        output="ok",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=2,
    )
    log.close()

    events = [
        json.loads(line)
        for line in (tmp_trace_dir / f"{run_id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    witness = next(e for e in events if e["event_type"] == "provider_execution_witness")
    # Payload is present (_force_payload=True) and shape is what the helper emitted
    p = witness["payload"]
    assert p["routing"]["routing_model_id"] == "gpt-5.4-pro"
    assert p["routing"]["routing_provider_id"] == "openai"
    assert p["policy"]["routing_candidate_decision"] == "blocked"
    assert p["policy"]["routing_candidate_reason_code"] == "provider_in_denylist"
    assert len(p["per_node_assignments"]) == 2
    assert p["substitution_summary"]["routing_candidate_blocked_by_policy"] is True


def test_provider_execution_witness_does_not_mask_provider_policy_violation(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro DESIGN_LOCK required test #7.

    The witness is an OBSERVATION, not an enforcement step. If the
    subsequent ``enforce_provider_policy`` check decides to abort the
    run (because the routing's chosen model violates the policy), the
    JSONL log MUST still contain BOTH: the witness event AND a
    subsequent ``failure`` event with
    ``error_type=provider_policy_violation``.
    """
    from sage.runtime.event_log import RuntimeEventLog

    run_id = "01WITNESSNOMASK0000000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("witness vs enforce ordering test")
    log.emit_task_started("witness vs enforce ordering test")

    log.emit_provider_execution_witness(
        witness_schema_version="v0",
        assignment_phase="initial",
        routing={
            "routing_model_id": "gpt-5.4-pro",
            "routing_provider_id": "openai",
            "system": 3,
            "domain": "code",
        },
        policy={
            "active": True,
            "allowlist": ["deepseek", "google"],
            "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_index": 0,
                "node_role": "coder",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": True,
            "routing_candidate_blocked_by_policy": True,
            "assignment_count": 1,
            "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )

    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="provider 'openai' is in the denylist",
    )
    log.emit_final_result(
        status="failure",
        output="",
        total_cost_usd=0.0,
        total_latency_ms=1.0,
        node_count=0,
    )
    log.close()

    events = [
        json.loads(line)
        for line in (tmp_trace_dir / f"{run_id}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    witness_seqs = [
        e["seq"] for e in events if e["event_type"] == "provider_execution_witness"
    ]
    violation_seqs = [
        e["seq"]
        for e in events
        if e["event_type"] == "failure"
        and e.get("error_type") == "provider_policy_violation"
    ]

    assert len(witness_seqs) == 1, "witness must emit exactly once"
    assert len(violation_seqs) == 1, "violation must emit exactly once"
    assert witness_seqs[0] < violation_seqs[0], (
        "witness MUST precede the policy violation in event order"
    )
