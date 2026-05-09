"""Learning Side-Effect Ledger v0 contract tests."""
from __future__ import annotations

import pathlib
import shutil
from uuid import uuid4

import pytest

from sage.pipeline import PipelineContext
from sage.pipeline_v2.learning_side_effects import emit_decision
from sage.runtime.credit_assignment.schema import (
    LearningSideEffectSchemaError,
    validate_record_shape,
)
from sage.runtime.credit_assignment.validate import validate_trace_dir
from sage.runtime.event_log import RuntimeEventLog, install_event_log
from sage.runtime.event_log.schema import EVENT_TYPES
from sage.runtime.oracle.verdict import EvidenceRef, OracleVerdict


@pytest.fixture
def trace_dir() -> pathlib.Path:
    path = pathlib.Path(".tmp") / "pytest-learning-side-effects" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_sidecar_writes_hash_chain_and_validates_parent_refs(trace_dir: pathlib.Path) -> None:
    log = RuntimeEventLog(run_id="01LSELEDGERVALID000000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger task")
        log.emit_task_started(ctx.task)
        log.emit_routing_decision(
            routing_source="test",
            system=2,
            domain="code",
            confidence=1.0,
            model_id="model-a",
        )
        _store(ctx, "routing_decision", log)
        final_seq = log.emit_final_result(
            status="success",
            output="ok",
            total_cost_usd=0.01,
            total_latency_ms=10.0,
            node_count=1,
        )
        _store(ctx, "final_result", log)
        verdict = OracleVerdict(
            trainable=True,
            verdict_source="exact",
            quality_label="pass",
            score=1.0,
            confidence=1.0,
            reason_codes=("exact_pass",),
            evidence=(EvidenceRef(run_id=log.run_id, evidence_hash="sha256:e1"),),
        )
        log.emit_oracle_verdict(parent_event_id=final_seq, verdict=verdict)
        _store(ctx, "oracle_verdict", log)
        ctx.oracle_verdict = verdict
        ctx.executed_model_id = "model-a"
        ctx.executed_template = "single_agent"
        ctx.cost = 0.01
        ctx.latency_ms = 10.0

        emit_decision(
            object(),  # pipeline is only carried for signature symmetry.
            ctx,
            side_effect="map_elites_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=1.0,
        )
    finally:
        token.var.reset(token)
        log.close()

    records = validate_trace_dir(trace_dir, run_id=log.run_id)

    assert len(records) == 1
    assert records[0]["side_effect"] == "map_elites_record_outcome"
    assert records[0]["prev_record_hash"] is None
    assert records[0]["record_hash"].startswith("sha256:")
    assert records[0]["oracle_verdict_ref"]["trainable"] is True


def test_validator_rejects_allowed_update_on_untrainable_oracle(
    trace_dir: pathlib.Path,
) -> None:
    log = RuntimeEventLog(run_id="01LSELEDGERBAD0000000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger bad task")
        log.emit_task_started(ctx.task)
        final_seq = log.emit_final_result(
            status="success",
            output="ok",
            total_cost_usd=0.0,
            total_latency_ms=1.0,
            node_count=1,
        )
        _store(ctx, "final_result", log)
        verdict = OracleVerdict(
            trainable=False,
            verdict_source="abstain",
            quality_label="unknown",
            score=None,
            confidence=1.0,
            reason_codes=("abstain",),
            evidence=(),
        )
        log.emit_oracle_verdict(parent_event_id=final_seq, verdict=verdict)
        _store(ctx, "oracle_verdict", log)
        ctx.oracle_verdict = verdict

        emit_decision(
            object(),
            ctx,
            side_effect="map_elites_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=0.0,
        )
    finally:
        token.var.reset(token)
        log.close()

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="requires trainable oracle verdict",
    ):
        validate_trace_dir(trace_dir, run_id=log.run_id)


def test_result_summary_forbids_raw_payload_keys(trace_dir: pathlib.Path) -> None:
    record = _minimal_record()
    record["result_summary"] = {"raw_output": "secret"}

    with pytest.raises(LearningSideEffectSchemaError, match="forbidden key"):
        validate_record_shape(record)


def test_validator_rejects_allowed_oracle_on_update_without_oracle_ref(
    trace_dir: pathlib.Path,
) -> None:
    log = RuntimeEventLog(run_id="01LSELEDGERNOREF00000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger missing oracle ref")
        log.emit_task_started(ctx.task)
        log.emit_final_result(
            status="success",
            output="ok",
            total_cost_usd=0.0,
            total_latency_ms=1.0,
            node_count=1,
        )
        _store(ctx, "final_result", log)
        ctx.oracle_verdict = None

        emit_decision(
            object(),
            ctx,
            side_effect="map_elites_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=1.0,
        )
    finally:
        token.var.reset(token)
        log.close()

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="requires oracle verdict ref",
    ):
        validate_trace_dir(trace_dir, run_id=log.run_id)


def test_runtime_event_taxonomy_remains_v0_15_types() -> None:
    assert len(EVENT_TYPES) == 15
    assert "learning_side_effect" not in EVENT_TYPES
    assert "credit_assignment" not in EVENT_TYPES


def _store(ctx: PipelineContext, event_type: str, log: RuntimeEventLog) -> None:
    ref = log.last_event_ref()
    assert ref is not None
    assert ref.event_type == event_type
    ctx.runtime_event_refs[event_type] = ref.to_dict()


def _minimal_record() -> dict:
    return {
        "schema_version": "learning_side_effect.v0",
        "seq": 0,
        "timestamp_ns": 1,
        "run_id": "r",
        "trace_id": "r",
        "task_hash": "h",
        "parent_event_refs": [],
        "oracle_verdict_ref": None,
        "policy_ref": {
            "decision_id": None,
            "routing_decision_ref": None,
            "policy_snapshot_hash": "sha256:" + "0" * 64,
            "candidate_set_hash": "sha256:" + "1" * 64,
            "selection_probability": None,
            "selection_probability_reason": "not_logged",
        },
        "subject": {
            "model_id": None,
            "provider_id": None,
            "template": None,
            "topology_id": None,
            "node_id": None,
            "tool_path_hash": None,
        },
        "side_effect": "bandit_record_outcome",
        "decision": "skipped",
        "reason_code": "no_pending_decision",
        "attempted": False,
        "gate": {
            "oracle_enabled": True,
            "oracle_trainable": False,
            "allow_training_updates": False,
            "quality_source": "none",
        },
        "metrics": {"quality": None, "cost_usd": None, "latency_ms": None},
        "result_summary": {"status": "observed", "redacted": True},
        "redaction_state": "redacted",
        "prev_record_hash": None,
        "record_hash": "sha256:x",
    }
