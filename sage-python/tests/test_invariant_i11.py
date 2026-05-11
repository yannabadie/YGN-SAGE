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
