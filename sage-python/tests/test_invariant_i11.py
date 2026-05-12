"""Tests for slice 10D Invariant I-11 (cgpro DESIGN_LOCKED 2026-05-11).

Binds ``provider_execution_witness.policy.routing_candidate_decision``
to the evaluated provider-policy decision via inline assertion in
``enforce_provider_policy`` + close-time audit on the JSONL trace.

Test surface per cgpro DESIGN_LOCK Q3 "Tests" column:
- Existing test_provider_execution_witness_does_not_mask_provider_policy_violation
  (already in test_provider_execution_witness.py — covers pairing
  semantics at the event log level)
- I-11 tests for initial/reroute/upgrade blocked witnesses
- Inline witness-vs-enforcement mismatch
- Close-time blocked-without-failure audit failure
- SAGE_TRACE_FAIL_CLOSED gate behavior on/off
- no_policy_active exclusion
- Preservation of the existing ProviderPolicyViolation path
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from sage.pipeline_v2.provider_policy import (
    ProviderPolicyViolation,
    enforce_provider_policy,
)
from sage.runtime.event_log import RuntimeEventLog
from sage.runtime.event_log.errors import EventLogInvariantViolation


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_trace_dir() -> Path:
    """Per-test trace dir under .tmp (default pytest temp ACL-denied here)."""
    path = Path(".tmp") / "pytest-i11" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _make_pipeline_with_policy(
    *, allowlist: tuple[str, ...] = (), denylist: tuple[str, ...] = (),
    assigner_provider_map: dict[str, str] | None = None,
) -> Any:
    """Build a fake pipeline with a provider policy + provider pool.

    Attrs are underscore-prefixed because `effective_provider_policy`
    reads `_provider_policy_source`, `_provider_allowlist`,
    `_provider_denylist` (see provider_policy.py line 154-157).
    """
    pool = SimpleNamespace()
    inferred = assigner_provider_map or {}
    pool.infer_provider = lambda mid, _m=inferred: _m.get(mid, "")
    pipeline = SimpleNamespace()
    pipeline.provider_pool = pool
    has_policy = bool(allowlist or denylist)
    pipeline._provider_allowlist = tuple(sorted(allowlist)) if allowlist else None
    pipeline._provider_denylist = tuple(sorted(denylist)) if denylist else ()
    pipeline._provider_policy_source = "cli" if has_policy else ""
    pipeline.llm_config = None
    return pipeline


def _make_ctx(
    *, assignments: dict[int, str] | None = None,
    nodes: list[Any] | None = None,
) -> Any:
    ctx = SimpleNamespace()
    ctx.assignments = assignments or {}
    ctx.provider_hints = {}
    # Topology with the given nodes
    if nodes:
        topo = SimpleNamespace()
        topo._nodes = nodes
        topo.get_node = lambda idx, _t=topo: _t._nodes[idx]
        topo.node_count = lambda _t=topo: len(_t._nodes)
        ctx.topology = topo
    else:
        ctx.topology = None
    return ctx


def _seed_witness_state(
    event_log: RuntimeEventLog,
    *,
    decision: str,
    routing_provider_id: str,
    active: bool = True,
    phase: str = "initial",
    reason_code: str | None = "provider_in_denylist",
) -> None:
    """Inject a witness state directly (bypass full witness emit)."""
    event_log._last_witness_state = {
        "phase": phase,
        "active": active,
        "decision": decision,
        "reason_code": reason_code,
        "routing_model_id": "gpt-5.4-pro",
        "routing_provider_id": routing_provider_id,
        "witness_seq": 1,
    }


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Inline binding tests (the authoritative I-11 path per cgpro Q1=C lock)
# ─────────────────────────────────────────────────────────────────────────────


def test_i11_witness_blocked_verified_blocked_emits_assertion_pass(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slice 9 normal flow: witness records blocked, policy re-evaluation
    also says blocked → assertion verdict=pass. The pipeline still
    continues via ModelAssigner substitution (per-node assignments
    pass policy).
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11WITNESSBLOCKEDPASS01"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("i11 inline pass")

    log.emit_task_started("i11 inline pass")
    _seed_witness_state(log, decision="blocked", routing_provider_id="openai")

    pipeline = _make_pipeline_with_policy(
        allowlist=("deepseek", "google"),
        denylist=("openai",),
        assigner_provider_map={
            "deepseek-v4-pro": "deepseek",
            "gemini-3-flash-preview": "google",
        },
    )
    ctx = _make_ctx(assignments={0: "deepseek-v4-pro", 1: "gemini-3-flash-preview"})

    # Substituted assignments pass policy → enforce returns OK
    enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert len(assertions) == 1, "exactly one I-11 assertion expected"
    p = assertions[0]["payload"]
    assert p["invariant_id"] == "I-11"
    assert p["verdict"] == "pass"
    assert p["declared_decision"] == "blocked"
    assert p["verified_decision"] == "blocked"
    assert p["fail_closed"] is False


def test_i11_witness_blocked_verified_allowed_emits_assertion_fail(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Witness lied: declared blocked, but live policy says allowed.
    Without the FAIL_CLOSED gate, emits assertion=fail and continues.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11WITNESSLIEDFAIL00001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("i11 inline fail")

    log.emit_task_started("i11 inline fail")
    # Witness claims openai is blocked, but our pipeline policy allows openai
    _seed_witness_state(log, decision="blocked", routing_provider_id="openai")

    pipeline = _make_pipeline_with_policy(
        allowlist=("openai",),  # openai IS allowed
        denylist=(),
        assigner_provider_map={"gpt-5.4-pro": "openai"},
    )
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert len(assertions) == 1
    p = assertions[0]["payload"]
    assert p["verdict"] == "fail"
    assert p["declared_decision"] == "blocked"
    assert p["verified_decision"] == "allowed"


def test_i11_witness_allowed_verified_allowed_emits_assertion_pass(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both witness and verified say allowed → verdict pass."""
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11ALLOWEDPASS00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    _seed_witness_state(
        log, decision="allowed", routing_provider_id="deepseek",
        reason_code="passes_policy",
    )

    pipeline = _make_pipeline_with_policy(
        allowlist=("deepseek",),
        assigner_provider_map={"deepseek-v4-pro": "deepseek"},
    )
    ctx = _make_ctx(assignments={0: "deepseek-v4-pro"})

    enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert len(assertions) == 1
    assert assertions[0]["payload"]["verdict"] == "pass"


def test_i11_fail_closed_mismatch_raises_event_log_invariant_violation(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under SAGE_TRACE_FAIL_CLOSED=1, a mismatch raises
    EventLogInvariantViolation BEFORE the existing
    ProviderPolicyViolation logic runs.
    """
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    run_id = "01I11FAILCLOSEDRAISES0001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    _seed_witness_state(log, decision="blocked", routing_provider_id="openai")

    pipeline = _make_pipeline_with_policy(
        allowlist=("openai",),  # openai IS allowed → mismatch
        assigner_provider_map={"gpt-5.4-pro": "openai"},
    )
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    with pytest.raises(EventLogInvariantViolation, match="I-11"):
        enforce_provider_policy(pipeline, ctx, log)


def test_i11_no_policy_active_excluded_from_assertion(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When witness was emitted under policy.active=false, I-11
    no-ops — no assertion emitted, no enforcement gate.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11NOPOLICYACTIVE000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    _seed_witness_state(
        log,
        decision="allowed",  # always allowed under no_policy
        routing_provider_id="openai",
        active=False,
        reason_code="no_policy_active",
    )

    pipeline = _make_pipeline_with_policy()  # empty policy
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert assertions == [], "no I-11 assertion when witness.policy.active=false"


def test_i11_no_witness_yet_no_assertion_emitted(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No witness emitted yet → no assertion. Safe under both gate
    states.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11NOWITNESSYET00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    # No _seed_witness_state — no witness emitted

    pipeline = _make_pipeline_with_policy(
        allowlist=("deepseek",),
        assigner_provider_map={"deepseek-v4-pro": "deepseek"},
    )
    ctx = _make_ctx(assignments={0: "deepseek-v4-pro"})

    enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()
    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert assertions == []


def test_i11_phase_reroute_recorded_in_assertion(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the witness was emitted with assignment_phase=reroute,
    the I-11 assertion records that phase faithfully."""
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11REROUTEPHASE0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    _seed_witness_state(
        log, decision="blocked", routing_provider_id="openai", phase="reroute",
    )

    pipeline = _make_pipeline_with_policy(
        denylist=("openai",),
        assigner_provider_map={"deepseek-v4-pro": "deepseek"},
    )
    ctx = _make_ctx(assignments={0: "deepseek-v4-pro"})
    enforce_provider_policy(pipeline, ctx, log)
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertion = next(e for e in events if e["event_type"] == "runtime_integrity_assertion")
    assert assertion["payload"]["phase"] == "reroute"


def test_i11_phase_upgrade_recorded_in_assertion(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the witness was emitted with assignment_phase=upgrade
    (FrugalGPT cascade), the I-11 assertion records that phase.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11UPGRADEPHASE0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    _seed_witness_state(
        log, decision="blocked", routing_provider_id="openai", phase="upgrade",
    )

    pipeline = _make_pipeline_with_policy(
        denylist=("openai",),
        assigner_provider_map={"deepseek-v4-pro": "deepseek"},
    )
    ctx = _make_ctx(assignments={0: "deepseek-v4-pro"})
    enforce_provider_policy(pipeline, ctx, log)
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    assertion = next(e for e in events if e["event_type"] == "runtime_integrity_assertion")
    assert assertion["payload"]["phase"] == "upgrade"


def test_i11_does_not_downgrade_provider_policy_violation(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro Q2 lock: SAGE_TRACE_FAIL_CLOSED only gates I-11, NEVER
    downgrades a real ProviderPolicyViolation into an allow. When
    per-node assignments actually violate policy, the existing
    ProviderPolicyViolation path MUST fire regardless of FAIL_CLOSED
    state.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11PRESERVESPPV00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    # Witness agrees with verified — no I-11 mismatch
    _seed_witness_state(log, decision="blocked", routing_provider_id="openai")

    pipeline = _make_pipeline_with_policy(
        denylist=("openai",),
        assigner_provider_map={"gpt-5.4-pro": "openai"},  # per-node also openai
    )
    # Per-node still points at openai → enforce_provider_policy finds
    # a violation → ProviderPolicyViolation MUST fire even though
    # FAIL_CLOSED is off.
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    with pytest.raises(ProviderPolicyViolation):
        enforce_provider_policy(pipeline, ctx, log)


# ─────────────────────────────────────────────────────────────────────────────
# Close-time audit tests (the backstop per cgpro Q1=C lock)
# ─────────────────────────────────────────────────────────────────────────────


def test_i11_close_time_audit_blocked_witness_with_failure_passes(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked policy-active witness followed by a
    provider_policy_violation failure event passes the close-time
    audit (returns []).
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11CLOSEAUDITPASS000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": ["deepseek"], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "coder",
                "assigned_model_id": "gpt-5.4-pro",
                "assigned_provider_id": "openai",
                "required_capabilities": [],
                "assignment_policy_decision": "blocked",
                "assignment_policy_reason_code": "provider_in_denylist",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 0,
            "blocked_assignment_count": 1,
            "rust_filter_details_observed": False,
        },
    )
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="openai denied",
    )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    violations = log.validate_invariants()
    log.close()
    assert violations == []


def test_i11_close_time_audit_blocked_witness_without_failure_returns_violation(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked policy-active witness WITHOUT a subsequent
    provider_policy_violation failure event is recorded as a
    violation by close-time audit. Without the FAIL_CLOSED gate,
    audit returns the list (does not raise).
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11CLOSEAUDITFAILS00001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": ["deepseek"], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "coder",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "required_capabilities": [],
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": True,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": True,
            "assignment_count": 1, "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    # No failure event — ModelAssigner silently substituted (slice 9 normal flow)
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    violations = log.validate_invariants()
    log.close()
    assert len(violations) == 1
    v = violations[0]
    assert v["invariant_id"] == "I-11"
    assert v["witness_phase"] == "initial"


def test_i11_fail_closed_verified_blocked_emits_provider_policy_failure_before_event_log_invariant_violation(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro VERIFY 2026-05-11 EDIT_REQUIRED Blocker 1: fail-closed
    ordering. When the per-node policy ALSO blocks AND the witness
    mismatches verified (declared=allowed, verified=blocked), the
    trace MUST contain BOTH the `provider_policy_violation` failure
    event AND the I-11 EventLogInvariantViolation — failure event
    landed FIRST in the trace, then exception raised.

    Required event order:
      provider_execution_witness
      → runtime_integrity_assertion(verdict=fail)
      → failure(error_type=provider_policy_violation)
      → EventLogInvariantViolation
    """
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    run_id = "01I11FCBLOCKEDORDER000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    # Witness "lied" — declared=allowed but live policy denies.
    # Per-node assignment ALSO violates → existing decision.violations
    # path runs → emit_failure FIRST, then EventLogInvariantViolation.
    _seed_witness_state(
        log,
        decision="allowed",
        routing_provider_id="openai",
        reason_code="passes_policy",
    )
    pipeline = _make_pipeline_with_policy(
        allowlist=("deepseek",),
        assigner_provider_map={"gpt-5.4-pro": "openai"},
    )
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    with pytest.raises(EventLogInvariantViolation):
        enforce_provider_policy(pipeline, ctx, log)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    types_in_order = [e["event_type"] for e in events]
    assert "runtime_integrity_assertion" in types_in_order
    failure_events = [
        e for e in events
        if e["event_type"] == "failure"
        and e.get("error_type") == "provider_policy_violation"
    ]
    assert len(failure_events) == 1, (
        "EXACTLY ONE provider_policy_violation failure event must be "
        "in the trace — audit evidence preserved per cgpro Blocker 1"
    )
    assertion_seq = next(
        e["seq"] for e in events
        if e["event_type"] == "runtime_integrity_assertion"
    )
    failure_seq = failure_events[0]["seq"]
    assert assertion_seq < failure_seq, (
        "assertion event MUST precede failure event"
    )


def test_i11_close_time_audit_rejects_unrelated_provider_policy_failure(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro Blocker 2: a provider_policy_violation failure that
    arrives AFTER a `node_started` (provider dispatch already
    happened) MUST NOT pair with an earlier blocked witness.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11AUDITUNRELATED000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": ["deepseek"], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "coder",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "required_capabilities": [],
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": True,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": True,
            "assignment_count": 1, "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    log.emit_node_started(
        topology_id="t1", node_id="n0", node_role="actor",
        attempt=1, model_id="deepseek-v4-pro", provider_id="deepseek",
        predecessor_ids=(), edge_ids=(),
        predecessors_by_channel=None,
    )
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="unrelated late failure",
    )
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=1)

    violations = log.validate_invariants()
    log.close()
    assert len(violations) == 1
    assert "provider dispatch" in violations[0]["message"]


def test_i11_close_time_audit_lifo_pairing_for_multiple_witnesses(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro Blocker 2 LIFO contract: when multiple blocked witnesses
    each have their own subsequent provider_policy_violation failure,
    each pair matches in proper order — not FIFO.

    Trace shape:
      witness A (blocked, phase=initial)
      failure A — pairs with A
      witness B (blocked, phase=reroute) — A is matched, B is open
      failure B — pairs with B
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11AUDITLIFOPAIR0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    for phase in ("initial", "reroute"):
        log.emit_provider_execution_witness(
            witness_schema_version="v0", assignment_phase=phase,
            routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
            policy={
                "active": True, "source": "cli",
                "allowlist": [], "denylist": ["openai"],
                "routing_candidate_decision": "blocked",
                "routing_candidate_reason_code": "provider_in_denylist",
            },
            per_node_assignments=[
                {
                    "node_id": "0", "node_role": "actor",
                    "assigned_model_id": "gpt-5.4-pro",
                    "assigned_provider_id": "openai",
                    "required_capabilities": [],
                    "assignment_policy_decision": "blocked",
                    "assignment_policy_reason_code": "provider_in_denylist",
                }
            ],
            substitution_summary={
                "routing_model_distinct_from_assignments": False,
                "routing_candidate_blocked_by_policy": True,
                "executed_models_distinct_from_routing": False,
                "assignment_count": 1, "allowed_assignment_count": 0,
                "blocked_assignment_count": 1,
                "rust_filter_details_observed": False,
            },
        )
        log.emit_failure(
            kind="provider_policy",
            error_type="provider_policy_violation",
            message=f"openai denied ({phase})",
        )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    violations = log.validate_invariants()
    log.close()
    assert violations == []


def test_i11_close_time_audit_raises_under_fail_closed(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under SAGE_TRACE_FAIL_CLOSED=1, the close-time audit raises
    EventLogInvariantViolation instead of returning the list.
    """
    monkeypatch.setenv("SAGE_TRACE_FAIL_CLOSED", "1")
    run_id = "01I11CLOSEAUDITRAISES0001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")

    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": ["deepseek"], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "coder",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "required_capabilities": [],
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": True,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": True,
            "assignment_count": 1, "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    with pytest.raises(EventLogInvariantViolation, match="I-11"):
        log.validate_invariants()


def test_emit_failure_includes_correlation_witness_seq_when_passed(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro DESIGN_LOCKED 2026-05-12 I11_FAILURE_CORRELATION_METADATA:
    `emit_failure(..., correlation_witness_seq=N)` MUST emit a
    failure event whose payload carries the correlation field. Legacy
    callers (no kwarg) MUST still emit valid failures without the
    field.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11FAILCORRSEQ00000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")

    # With correlation
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="denied",
        correlation_witness_seq=42,
    )
    # Legacy (no correlation)
    log.emit_failure(
        kind="other",
        error_type="some_other_error",
        message="x",
    )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    failures = [e for e in events if e["event_type"] == "failure"]
    assert len(failures) == 2
    # First failure has correlation
    assert failures[0]["payload"]["correlation_witness_seq"] == 42
    # Second has no correlation key (or it's null)
    assert (
        "correlation_witness_seq" not in failures[1]["payload"]
        or failures[1]["payload"]["correlation_witness_seq"] is None
    )


def test_i11_close_time_audit_uses_correlation_when_present(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the failure event carries `correlation_witness_seq`, the
    close-time audit MUST pair by witness identity, not by LIFO
    window. This proves the fail-secure semantics when the
    correlation metadata is available.

    Trace shape (correlation explicit, LIFO would pair wrong):
      W_initial (seq=N1, blocked)
      W_reroute (seq=N2, blocked)
      F (provider_policy_violation, correlation=N1)
    → W_initial pairs with F (identity), W_reroute remains unmatched
      and yields a violation.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11AUDITCORRPAIR000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    # Witness 1 (initial)
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": [], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "actor",
                "assigned_model_id": "gpt-5.4-pro",
                "assigned_provider_id": "openai",
                "required_capabilities": [],
                "assignment_policy_decision": "blocked",
                "assignment_policy_reason_code": "provider_in_denylist",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 0,
            "blocked_assignment_count": 1,
            "rust_filter_details_observed": False,
        },
    )
    witness1_state = log._last_witness_state
    assert witness1_state is not None
    w1_seq = witness1_state["witness_seq"]

    # Witness 2 (reroute) — same shape, new seq
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="reroute",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": [], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "actor",
                "assigned_model_id": "gpt-5.4-pro",
                "assigned_provider_id": "openai",
                "required_capabilities": [],
                "assignment_policy_decision": "blocked",
                "assignment_policy_reason_code": "provider_in_denylist",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 0,
            "blocked_assignment_count": 1,
            "rust_filter_details_observed": False,
        },
    )

    # Failure references the FIRST witness explicitly — but the
    # intervening reroute witness flipped dispatch_seen=True on
    # witness 1. Per cgpro VERIFY
    # I11_CORRELATION_TEMPORAL_ORDERING_PATCH 2026-05-12: identity
    # pairing does NOT waive temporal ordering. The correlated
    # failure is rejected, witness 1 stays unmatched (violation),
    # witness 2 has no failure pairing (violation).
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="openai denied (late correlation)",
        correlation_witness_seq=w1_seq,
    )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    violations = log.validate_invariants()
    log.close()
    # Both witnesses unmatched — w1 because correlation arrived
    # after dispatch sentinel (intervening w2), w2 because no
    # failure pairing.
    assert len(violations) == 2
    phases_violated = sorted(v["witness_phase"] for v in violations)
    assert phases_violated == ["initial", "reroute"]


def test_i11_close_time_audit_correlated_failure_before_dispatch_passes(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro VERIFY 2026-05-12 I11_CORRELATION_TEMPORAL_ORDERING_PATCH
    required test #1: explicit correlation BEFORE any dispatch
    sentinel passes audit cleanly.

    Trace:
      W(seq=N, blocked)
      F(error_type=provider_policy_violation, correlation_witness_seq=N)

    Expected: no I-11 violation.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11CORRBEFOREDISP000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": [], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "actor",
                "assigned_model_id": "gpt-5.4-pro",
                "assigned_provider_id": "openai",
                "required_capabilities": [],
                "assignment_policy_decision": "blocked",
                "assignment_policy_reason_code": "provider_in_denylist",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 0,
            "blocked_assignment_count": 1,
            "rust_filter_details_observed": False,
        },
    )
    witness_seq = log._last_witness_state["witness_seq"]
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="openai denied (timely)",
        correlation_witness_seq=witness_seq,
    )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)
    violations = log.validate_invariants()
    log.close()
    assert violations == [], (
        "correlated failure arriving BEFORE dispatch is valid evidence — "
        "audit should pass cleanly"
    )


def test_i11_close_time_audit_correlated_failure_after_dispatch_fails(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro VERIFY 2026-05-12 I11_CORRELATION_TEMPORAL_ORDERING_PATCH
    required test #2: explicit correlation AFTER node_started
    (provider dispatch) MUST fail. Identity pairing does not waive
    temporal ordering — late denial evidence cannot retroactively
    validate a trace where the protected side effect already
    happened.

    Trace:
      W(seq=N, blocked)
      node_started
      F(error_type=provider_policy_violation, correlation_witness_seq=N)

    Expected: I-11 violation mentioning provider dispatch.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11CORRAFTERDISP0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
        policy={
            "active": True, "source": "cli",
            "allowlist": [], "denylist": ["openai"],
            "routing_candidate_decision": "blocked",
            "routing_candidate_reason_code": "provider_in_denylist",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "actor",
                "assigned_model_id": "gpt-5.4-pro",
                "assigned_provider_id": "openai",
                "required_capabilities": [],
                "assignment_policy_decision": "blocked",
                "assignment_policy_reason_code": "provider_in_denylist",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": True,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 0,
            "blocked_assignment_count": 1,
            "rust_filter_details_observed": False,
        },
    )
    witness_seq = log._last_witness_state["witness_seq"]
    log.emit_node_started(
        topology_id="t1", node_id="n0", node_role="actor",
        attempt=1, model_id="gpt-5.4-pro", provider_id="openai",
        predecessor_ids=(), edge_ids=(),
        predecessors_by_channel=None,
    )
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="openai denied (LATE — after dispatch)",
        correlation_witness_seq=witness_seq,
    )
    log.emit_final_result(status="success", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=1)

    violations = log.validate_invariants()
    log.close()
    assert len(violations) == 1, (
        "correlated failure AFTER dispatch MUST be rejected per "
        "I-11 temporal ordering contract (cgpro 2026-05-12 lock)"
    )
    assert "provider dispatch" in violations[0]["message"]


def test_i11_close_time_audit_rejects_correlation_to_non_blocked_or_missing_witness(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro VERIFY 2026-05-12 I11_CORRELATION_TEMPORAL_ORDERING_PATCH
    required test #3: when correlation_witness_seq points to a
    witness that is NOT policy-active + blocked (or doesn't exist),
    the failure MUST NOT be accepted as valid I-11 evidence.

    Sub-trace A — correlation to allowed witness:
      W(seq=N, ALLOWED)
      F(error_type=provider_policy_violation, correlation_witness_seq=N)
    → no blocked witness in unmatched → failure silently dropped.

    Sub-trace B — correlation to non-existent witness seq:
      F(error_type=provider_policy_violation, correlation_witness_seq=999999)
    → no matching witness → failure silently dropped.

    Neither case produces an I-11 violation because there is no
    blocked-witness-without-failure to report. The failure event
    itself is just orphaned evidence.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11CORRWRONGTARGET00001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    # Allowed witness — not in `unmatched` since it's not blocked
    log.emit_provider_execution_witness(
        witness_schema_version="v0", assignment_phase="initial",
        routing={"routing_model_id": "deepseek-v4-pro", "routing_provider_id": "deepseek"},
        policy={
            "active": True, "source": "cli",
            "allowlist": ["deepseek"], "denylist": [],
            "routing_candidate_decision": "allowed",
            "routing_candidate_reason_code": "passes_policy",
        },
        per_node_assignments=[
            {
                "node_id": "0", "node_role": "actor",
                "assigned_model_id": "deepseek-v4-pro",
                "assigned_provider_id": "deepseek",
                "required_capabilities": [],
                "assignment_policy_decision": "allowed",
                "assignment_policy_reason_code": "passes_policy",
            }
        ],
        substitution_summary={
            "routing_model_distinct_from_assignments": False,
            "routing_candidate_blocked_by_policy": False,
            "executed_models_distinct_from_routing": False,
            "assignment_count": 1, "allowed_assignment_count": 1,
            "blocked_assignment_count": 0,
            "rust_filter_details_observed": False,
        },
    )
    w_seq = log._last_witness_state["witness_seq"]
    # Sub-trace A: correlate to the allowed witness
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="orphaned failure correlated to allowed witness",
        correlation_witness_seq=w_seq,
    )
    # Sub-trace B: correlate to a non-existent witness seq
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="orphaned failure correlated to missing witness",
        correlation_witness_seq=999999,
    )
    log.emit_final_result(status="success", output="ok", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    violations = log.validate_invariants()
    log.close()
    # No blocked-witness-without-failure → no violations. The two
    # orphaned failures are silently dropped (not paired, not
    # reported as I-11 evidence).
    assert violations == []


def test_reroute_rebuild_blocked_candidate_chain_witness_assertion_failure_no_dispatch(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro FUTURE_HARDENING_ID=REROUTE_REBUILD_RUNTIME_INTEGRATION_SMOKE
    (non-blocker per VERIFY 2026-05-12). Drives the actual sequence
    that REROUTE_REBUILD invokes in `execute.py:387-411`:

      runtime_emit_provider_execution_witness(phase=reroute, blocked)
      → enforce_provider_policy(pipeline, ctx, event_log)
        → _maybe_emit_i11_assertion (verdict=pass: both blocked)
        → emit_failure(provider_policy_violation)
        → raise ProviderPolicyViolation

    Asserts:
      1. witness with phase=reroute + decision=blocked emitted
      2. runtime_integrity_assertion with verdict=pass + phase=reroute
      3. failure(provider_policy_violation) emitted with
         correlation_witness_seq pointing at the reroute witness
      4. ProviderPolicyViolation raised → control flow aborts BEFORE
         the next runner.run() call (proven by absence of
         node_started/runner_timeout events in this trace)

    Not driving the full `execute()` function because that requires
    the Rust topology engine + a TopologyController returning
    "__REROUTE__". This helper-level test proves the same functional
    contract: the runtime helpers that the reroute path invokes
    enforce I-11 correctly when the reroute candidate is blocked.
    The source inspection test
    (`test_reroute_rebuild_path_calls_enforce_provider_policy`)
    separately proves that `execute.py` DOES invoke these helpers
    in the right order.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    monkeypatch.setenv("SAGE_TRACE_RAW", "1")
    run_id = "01I11REROUTECHAIN0000001"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("reroute integration smoke")
    log.emit_task_started("reroute integration smoke")

    # Build pipeline with active provider policy that denies openai.
    # The "reroute" candidate is gpt-5.4-pro (openai) which the
    # router proposed during a hypothetical reroute attempt.
    pipeline = _make_pipeline_with_policy(
        allowlist=("deepseek",),
        denylist=("openai",),
        assigner_provider_map={
            "gpt-5.4-pro": "openai",
            "deepseek-v4-pro": "deepseek",
        },
    )
    # ModelAssigner did NOT substitute — per-node still openai →
    # decision.violations will be non-empty → enforce_provider_policy
    # emits failure + raises.
    ctx = _make_ctx(assignments={0: "gpt-5.4-pro"})

    # Step 1: emit reroute witness (mirrors execute.py:387-395)
    from sage.pipeline_v2.runtime_events import (
        runtime_emit_provider_execution_witness,
    )
    runtime_emit_provider_execution_witness(
        pipeline, ctx, log,
        routing_model_id="gpt-5.4-pro",
        assignment_phase="reroute",
    )

    # Step 2: enforce_provider_policy (mirrors execute.py:408-410)
    # Should:
    #   - emit runtime_integrity_assertion (witness blocked + verified
    #     blocked → verdict=pass)
    #   - emit failure(provider_policy_violation) with correlation
    #   - raise ProviderPolicyViolation
    with pytest.raises(ProviderPolicyViolation):
        enforce_provider_policy(pipeline, ctx, log)

    log.emit_final_result(
        status="failure", output="", total_cost_usd=0,
        total_latency_ms=1.0, node_count=0,
    )
    log.close()

    events = _read_events(tmp_trace_dir / f"{run_id}.jsonl")
    event_types = [e["event_type"] for e in events]

    # Assertion 1: witness emitted with phase=reroute, decision=blocked
    witnesses = [e for e in events if e["event_type"] == "provider_execution_witness"]
    assert len(witnesses) == 1
    w_payload = witnesses[0]["payload"]
    assert w_payload["assignment_phase"] == "reroute"
    assert w_payload["policy"]["active"] is True
    assert w_payload["policy"]["routing_candidate_decision"] == "blocked"

    # Assertion 2: runtime_integrity_assertion present, verdict=pass
    # (witness blocked, verified also blocked → I-11 invariant holds)
    assertions = [e for e in events if e["event_type"] == "runtime_integrity_assertion"]
    assert len(assertions) == 1
    a_payload = assertions[0]["payload"]
    assert a_payload["invariant_id"] == "I-11"
    assert a_payload["verdict"] == "pass"
    assert a_payload["phase"] == "reroute"
    assert a_payload["declared_decision"] == "blocked"
    assert a_payload["verified_decision"] == "blocked"
    witness_seq = witnesses[0]["seq"]
    assert a_payload["witness_seq"] == witness_seq

    # Assertion 3: failure(provider_policy_violation) emitted with
    # correlation pointing at the reroute witness
    failures = [
        e for e in events
        if e["event_type"] == "failure"
        and e.get("error_type") == "provider_policy_violation"
    ]
    assert len(failures) == 1
    f_payload = failures[0].get("payload", {})
    assert f_payload.get("correlation_witness_seq") == witness_seq, (
        "failure event MUST carry correlation_witness_seq pointing at "
        "the reroute witness — per cgpro NEXT_HARDENING_ID failure "
        "schema v1_1 lock"
    )

    # Assertion 4: no node_started — provider dispatch was blocked
    # by ProviderPolicyViolation BEFORE runner2.run() would have been
    # called. This is the core safety property the reroute binding
    # exists to guarantee.
    assert "node_started" not in event_types, (
        "blocked reroute candidate MUST NOT reach node_started — "
        "this would mean provider dispatch happened despite the "
        "policy denial"
    )

    # Bonus: close-time audit passes cleanly (witness paired with
    # failure via correlation, no orphaned blocked witness).
    violations = log.validate_invariants()
    assert violations == []


def test_reroute_rebuild_path_calls_enforce_provider_policy() -> None:
    """cgpro VERIFY 2026-05-12 NEXT_BLOCK_ID=REROUTE_REBUILD_I11_INLINE_BINDING
    acceptance #3: the REROUTE_REBUILD path MUST invoke
    `enforce_provider_policy` between the reroute witness emit and
    the runner2.run() dispatch. Without this, a blocked reroute
    candidate could reach node_started without I-11 inline binding
    firing.

    Source-inspection test: drives a real pipeline reroute end-to-end
    would require the Rust topology engine + a controller decision;
    the source guard catches refactor drift without that
    infrastructure. The existing
    `test_execute_py_reroute_path_emits_witness_with_reroute_phase`
    proves the witness emit is there; this test proves the
    enforcement call is right after it on the same path.
    """
    import inspect
    from sage.pipeline_v2 import execute as execute_mod

    src = inspect.getsource(execute_mod)
    # Locate the REROUTE_REBUILD witness emit position
    reroute_witness_marker = 'assignment_phase="reroute"'
    assert reroute_witness_marker in src
    after_reroute_emit = src.split(reroute_witness_marker, 1)[1]
    # Within the next ~50 lines on the same path, enforce_provider_policy
    # MUST be called. The runner2 dispatch happens after that.
    enforce_call = "provider_policy_mod.enforce_provider_policy"
    runner2_dispatch = "runner2 = TopologyRunner"
    enforce_pos = after_reroute_emit.find(enforce_call)
    runner2_pos = after_reroute_emit.find(runner2_dispatch)
    assert enforce_pos != -1, (
        "REROUTE_REBUILD path MUST call provider_policy_mod."
        "enforce_provider_policy after the reroute witness emit "
        "(cgpro 2026-05-12 NEXT_BLOCK_ID=REROUTE_REBUILD_I11_INLINE_BINDING)"
    )
    assert runner2_pos != -1, "expected runner2 dispatch on reroute path"
    assert enforce_pos < runner2_pos, (
        "enforce_provider_policy MUST precede runner2 dispatch on "
        "the REROUTE_REBUILD path — provider policy denial must "
        "block dispatch (cgpro 2026-05-12 acceptance criterion #5: "
        "blocked reroute MUST NOT reach node_started)"
    )


def test_witness_reads_policy_from_pipeline_underscore_attrs() -> None:
    """Production-drift regression (2026-05-12 paid canary discovery
    at HEAD `381445fd`): the witness helper MUST read the policy via
    `effective_provider_policy(pipeline)` so it sees the
    pipeline._provider_allowlist / _denylist attrs that the CLI sets
    via `configure_pipeline_provider_policy`. Reading directly from
    `ctx.provider_allowlist` missed this path, and the canary trace
    showed `policy.active=False` even though Rust had recorded 18
    real rust_filter_rejections. Without this fix, the I-11 inline
    binding skips in production (no_policy_active exclusion).
    """
    from sage.pipeline_v2.runtime_events import (
        runtime_emit_provider_execution_witness,
    )

    # Pipeline-level policy (matches CLI / production path)
    pool = SimpleNamespace()
    pool.infer_provider = lambda mid: (
        "deepseek" if mid == "deepseek-v4-pro" else
        "openai" if mid == "gpt-5.4-pro" else ""
    )
    pipeline = SimpleNamespace()
    pipeline.provider_pool = pool
    pipeline._provider_allowlist = ("deepseek",)
    pipeline._provider_denylist = ("openai",)
    pipeline._provider_policy_source = "cli"
    pipeline.llm_config = None

    # ctx has NO provider_allowlist / _denylist — production CLI flow
    nodes = [SimpleNamespace(role="coder", model_id="deepseek-v4-pro",
                             required_capabilities=())]
    topo = SimpleNamespace()
    topo._nodes = nodes
    topo.id = "t1"
    topo.template_type = "sequential"
    topo.get_node = lambda idx, _t=topo: _t._nodes[idx]
    topo.node_count = lambda _t=topo: len(_t._nodes)
    ctx = SimpleNamespace()
    ctx.topology = topo
    ctx.assignments = {0: "deepseek-v4-pro"}
    ctx.provider_hints = {}
    ctx.routing_source = "rust_system_router"
    ctx.system = 3
    ctx.domain = "code"
    ctx.confidence = 0.85

    class _Recorder:
        def __init__(self):
            self.calls = []

        def emit_provider_execution_witness(self, **kw):
            self.calls.append(kw)
            return 1

    log = _Recorder()
    runtime_emit_provider_execution_witness(
        pipeline, ctx, log, routing_model_id="gpt-5.4-pro",
    )
    assert len(log.calls) == 1
    pol = log.calls[0]["policy"]
    assert pol["active"] is True, (
        "policy.active MUST be True when "
        "pipeline._provider_allowlist/_denylist are set even if "
        "ctx.provider_allowlist is absent — production CLI flow drift "
        "discovered in the 2026-05-12 paid canary"
    )
    assert pol["allowlist"] == ["deepseek"]
    assert pol["denylist"] == ["openai"]
    assert pol["routing_candidate_decision"] == "blocked"
    assert pol["routing_candidate_reason_code"] == "provider_in_denylist"


def test_i11_close_time_audit_intervening_witness_invalidates_prior_unmatched_blocked_witness(
    tmp_trace_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cgpro VERIFY 2026-05-11 non-blocking hardening: explicit
    intervening-witness invalidation.

    Trace:
      W_initial (blocked)
      W_reroute (blocked) — intervening witness flips
        dispatch_seen=True on W_initial
      F (provider_policy_violation)

    Expected:
      W_reroute pairs with F (most recent unmatched).
      W_initial remains unmatched and yields an I-11 audit violation
      with the "dispatch / another witness" reason in the message.

    This is the LIFO+dispatch-sentinel contract made explicit at the
    multi-witness boundary. cgpro flagged this as "make the window
    semantics explicit" — not a blocker for slice 10D closure.
    """
    monkeypatch.delenv("SAGE_TRACE_FAIL_CLOSED", raising=False)
    run_id = "01I11AUDITINTERVENWITNES1"
    log = RuntimeEventLog(run_id=run_id, trace_dir=tmp_trace_dir)
    log.set_task_text("t")
    log.emit_task_started("t")
    for phase in ("initial", "reroute"):
        log.emit_provider_execution_witness(
            witness_schema_version="v0", assignment_phase=phase,
            routing={"routing_model_id": "gpt-5.4-pro", "routing_provider_id": "openai"},
            policy={
                "active": True, "source": "cli",
                "allowlist": [], "denylist": ["openai"],
                "routing_candidate_decision": "blocked",
                "routing_candidate_reason_code": "provider_in_denylist",
            },
            per_node_assignments=[
                {
                    "node_id": "0", "node_role": "actor",
                    "assigned_model_id": "gpt-5.4-pro",
                    "assigned_provider_id": "openai",
                    "required_capabilities": [],
                    "assignment_policy_decision": "blocked",
                    "assignment_policy_reason_code": "provider_in_denylist",
                }
            ],
            substitution_summary={
                "routing_model_distinct_from_assignments": False,
                "routing_candidate_blocked_by_policy": True,
                "executed_models_distinct_from_routing": False,
                "assignment_count": 1, "allowed_assignment_count": 0,
                "blocked_assignment_count": 1,
                "rust_filter_details_observed": False,
            },
        )
    # Single failure event — must pair with the MORE RECENT witness
    # (reroute), leaving the initial witness unmatched + invalidated.
    log.emit_failure(
        kind="provider_policy",
        error_type="provider_policy_violation",
        message="openai denied (reroute)",
    )
    log.emit_final_result(status="failure", output="", total_cost_usd=0, total_latency_ms=1.0, node_count=0)

    violations = log.validate_invariants()
    log.close()
    assert len(violations) == 1, (
        "exactly one violation expected — W_initial invalidated by "
        "intervening W_reroute, no failure available to pair it"
    )
    v = violations[0]
    assert v["witness_phase"] == "initial"
    assert "dispatch / another witness" in v["message"]
