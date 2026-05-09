"""Learning Side-Effect Ledger v0 contract tests."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from uuid import uuid4

import pytest

from sage.pipeline import PipelineContext
from sage.pipeline_v2.learning_side_effects import emit_decision
from sage.runtime.credit_assignment.schema import (
    LearningSideEffectSchemaError,
    canonical_json,
    record_hash_input,
    validate_record_shape,
)
from sage.runtime.credit_assignment.validate import (
    validate_evidence_boundary,
    validate_trace_dir,
)
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


def test_stage5_evidence_boundary_accepts_minimal_oracle_backed_decisions(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSESTAGE5VALID000000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger stage5 task")
        _seed_trainable_oracle_trace(ctx, log)
        ctx.executed_model_id = "model-a"
        ctx.executed_template = "single_agent"

        emit_decision(
            object(),
            ctx,
            side_effect="bandit_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=1.0,
        )
        emit_decision(
            object(),
            ctx,
            side_effect="map_elites_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=1.0,
        )
        emit_decision(
            object(),
            ctx,
            side_effect="online_evolution_should_evolve",
            decision="skipped",
            reason_code="should_evolve_false",
            attempted=True,
            quality=1.0,
        )
    finally:
        token.var.reset(token)
        log.close()

    records = validate_evidence_boundary(
        trace_dir,
        run_id=log.run_id,
        expect_default_pipeline_learn=True,
    )

    assert {
        "bandit_record_outcome",
        "map_elites_record_outcome",
        "online_evolution_should_evolve",
    } <= {record["side_effect"] for record in records}


def test_stage5_evidence_boundary_rejects_missing_minimal_decisions(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSESTAGE5MISSING000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger missing stage5 task")
        _seed_trainable_oracle_trace(ctx, log)

        emit_decision(
            object(),
            ctx,
            side_effect="bandit_record_outcome",
            decision="allowed",
            reason_code="oracle_trainable",
            attempted=True,
            quality=1.0,
        )
    finally:
        token.var.reset(token)
        log.close()

    assert len(validate_trace_dir(trace_dir, run_id=log.run_id)) == 1
    with pytest.raises(
        LearningSideEffectSchemaError,
        match="missing required stage5 side-effect decision",
    ):
        validate_evidence_boundary(
            trace_dir,
            run_id=log.run_id,
            expect_default_pipeline_learn=True,
        )


def test_evidence_boundary_requires_run_id(trace_dir: pathlib.Path) -> None:
    with pytest.raises(LearningSideEffectSchemaError, match="requires run_id"):
        validate_evidence_boundary(trace_dir, run_id="")


def test_validator_rejects_forged_oracle_ref_trainable(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSEFORGEDTRAINABLE0001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger forged oracle task")
        _seed_oracle_trace(ctx, log, trainable=False)
    finally:
        token.var.reset(token)
        log.close()

    oracle_ref = ctx.runtime_event_refs["oracle_verdict"]
    record = _minimal_record()
    record.update(
        {
            "run_id": log.run_id,
            "trace_id": log.run_id,
            "task_hash": "forged-task-hash",
            "parent_event_refs": [
                ctx.runtime_event_refs["final_result"],
                oracle_ref,
            ],
            "oracle_verdict_ref": {
                "seq": oracle_ref["seq"],
                "payload_hash": oracle_ref["payload_hash"],
                "trainable": True,
                "verdict_source": "exact",
                "quality_label": "fail",
                "score": 0.0,
                "evidence_hashes": [],
            },
            "side_effect": "map_elites_record_outcome",
            "decision": "allowed",
            "reason_code": "oracle_trainable",
            "attempted": True,
            "gate": {
                "oracle_enabled": True,
                "oracle_trainable": True,
                "allow_training_updates": True,
                "quality_source": "oracle",
            },
            "metrics": {"quality": 0.0, "cost_usd": 0.0, "latency_ms": 1.0},
        }
    )
    _write_single_record(trace_dir, record)

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="does not match RuntimeEventLog payload",
    ):
        validate_trace_dir(trace_dir, run_id=log.run_id)


def test_validator_rejects_duplicate_runtime_event_keys(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSEDUPLICATEKEY000001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger duplicate runtime key task")
        _seed_trainable_oracle_trace(ctx, log)
    finally:
        token.var.reset(token)
        log.close()

    canonical_lines = (trace_dir / f"{log.run_id}.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    (trace_dir / "spoofed-sibling.jsonl").write_text(
        canonical_lines[0] + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="duplicate RuntimeEventLog event key",
    ):
        validate_trace_dir(trace_dir, run_id=log.run_id)


def test_evidence_boundary_uses_canonical_runtime_log_not_spoofed_sibling(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSECANONICALONLY00001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger canonical runtime source task")
        _seed_oracle_trace(ctx, log, trainable=False)
    finally:
        token.var.reset(token)
        log.close()

    oracle_ref = ctx.runtime_event_refs["oracle_verdict"]
    spoof_hash = "sha256:" + "f" * 64
    spoofed_oracle = _runtime_event(trace_dir, log.run_id, "oracle_verdict")
    spoofed_oracle["payload_hash"] = spoof_hash
    spoofed_oracle["payload"]["trainable"] = True
    (trace_dir / "spoofed-sibling.jsonl").write_text(
        json.dumps(spoofed_oracle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    record = _minimal_record()
    record.update(
        {
            "run_id": log.run_id,
            "trace_id": log.run_id,
            "task_hash": "canonical-only-task-hash",
            "parent_event_refs": [ctx.runtime_event_refs["final_result"]],
            "oracle_verdict_ref": {
                "seq": oracle_ref["seq"],
                "payload_hash": spoof_hash,
                "trainable": True,
                "verdict_source": "exact",
                "quality_label": "pass",
                "score": 1.0,
                "evidence_hashes": [],
            },
            "side_effect": "map_elites_record_outcome",
            "decision": "allowed",
            "reason_code": "oracle_trainable",
            "attempted": True,
            "gate": {
                "oracle_enabled": True,
                "oracle_trainable": True,
                "allow_training_updates": True,
                "quality_source": "oracle",
            },
            "metrics": {"quality": 1.0, "cost_usd": 0.0, "latency_ms": 1.0},
        }
    )
    _write_single_record(trace_dir, record)

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="oracle_verdict payload_hash mismatch",
    ):
        validate_evidence_boundary(trace_dir, run_id=log.run_id)


def test_evidence_boundary_requires_per_record_oracle_ref(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSEBOUNDARYNOREF0001", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger boundary missing oracle ref task")
        _seed_trainable_oracle_trace(ctx, log)
    finally:
        token.var.reset(token)
        log.close()

    record = _minimal_record()
    record.update(
        {
            "run_id": log.run_id,
            "trace_id": log.run_id,
            "task_hash": "missing-record-oracle-ref",
            "parent_event_refs": [ctx.runtime_event_refs["final_result"]],
            "side_effect": "online_evolution_should_evolve",
            "decision": "skipped",
            "reason_code": "should_evolve_false",
            "attempted": True,
            "gate": {
                "oracle_enabled": True,
                "oracle_trainable": True,
                "allow_training_updates": True,
                "quality_source": "oracle",
            },
            "metrics": {"quality": 1.0, "cost_usd": 0.0, "latency_ms": 1.0},
        }
    )
    _write_single_record(trace_dir, record)

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="evidence-boundary record requires oracle_verdict_ref",
    ):
        validate_evidence_boundary(trace_dir, run_id=log.run_id)


def test_evidence_boundary_requires_oracle_payload_trainable(
    trace_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    log = RuntimeEventLog(run_id="01LSEBOUNDARYPAYLOAD01", trace_dir=trace_dir)
    token = install_event_log(log)
    try:
        ctx = PipelineContext(task="ledger boundary missing oracle payload task")
        _seed_trainable_oracle_trace(ctx, log)
    finally:
        token.var.reset(token)
        log.close()

    _remove_runtime_event_payload(trace_dir, log.run_id, "oracle_verdict")
    oracle_ref = ctx.runtime_event_refs["oracle_verdict"]
    record = _minimal_record()
    record.update(
        {
            "run_id": log.run_id,
            "trace_id": log.run_id,
            "task_hash": "missing-oracle-payload-trainable",
            "parent_event_refs": [ctx.runtime_event_refs["final_result"]],
            "oracle_verdict_ref": {
                "seq": oracle_ref["seq"],
                "payload_hash": oracle_ref["payload_hash"],
                "trainable": True,
                "verdict_source": "exact",
                "quality_label": "pass",
                "score": 1.0,
                "evidence_hashes": ["sha256:e1"],
            },
            "side_effect": "map_elites_record_outcome",
            "decision": "allowed",
            "reason_code": "oracle_trainable",
            "attempted": True,
            "gate": {
                "oracle_enabled": True,
                "oracle_trainable": True,
                "allow_training_updates": True,
                "quality_source": "oracle",
            },
            "metrics": {"quality": 1.0, "cost_usd": 0.0, "latency_ms": 1.0},
        }
    )
    _write_single_record(trace_dir, record)

    with pytest.raises(
        LearningSideEffectSchemaError,
        match="evidence-boundary oracle_verdict requires payload.trainable",
    ):
        validate_evidence_boundary(trace_dir, run_id=log.run_id)


def test_runtime_event_taxonomy_remains_v0_15_types() -> None:
    assert len(EVENT_TYPES) == 15
    assert "learning_side_effect" not in EVENT_TYPES
    assert "credit_assignment" not in EVENT_TYPES


def _seed_trainable_oracle_trace(ctx: PipelineContext, log: RuntimeEventLog) -> None:
    _seed_oracle_trace(ctx, log, trainable=True)


def _seed_oracle_trace(
    ctx: PipelineContext,
    log: RuntimeEventLog,
    *,
    trainable: bool,
) -> None:
    log.emit_task_started(ctx.task)
    _store(ctx, "task_started", log)
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
        trainable=trainable,
        verdict_source="exact" if trainable else "abstain",
        quality_label="pass" if trainable else "unknown",
        score=1.0 if trainable else None,
        confidence=1.0,
        reason_codes=("exact_pass",) if trainable else ("abstain",),
        evidence=(
            (EvidenceRef(run_id=log.run_id, evidence_hash="sha256:e1"),)
            if trainable
            else ()
        ),
    )
    log.emit_oracle_verdict(parent_event_id=final_seq, verdict=verdict)
    _store(ctx, "oracle_verdict", log)
    ctx.oracle_verdict = verdict
    ctx.cost = 0.01
    ctx.latency_ms = 10.0


def _write_single_record(trace_dir: pathlib.Path, record: dict) -> None:
    record["record_hash"] = _record_hash(record)
    (trace_dir / "learning_side_effects.jsonl").write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _record_hash(record: dict) -> str:
    digest = hashlib.sha256(
        canonical_json(record_hash_input(record)).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _runtime_event(trace_dir: pathlib.Path, run_id: str, event_type: str) -> dict:
    for line in (trace_dir / f"{run_id}.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        event = json.loads(line)
        if event["event_type"] == event_type:
            return event
    raise AssertionError(f"missing {event_type}")


def _remove_runtime_event_payload(
    trace_dir: pathlib.Path,
    run_id: str,
    event_type: str,
) -> None:
    path = trace_dir / f"{run_id}.jsonl"
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["event_type"] == event_type:
            event.pop("payload", None)
        lines.append(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


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
