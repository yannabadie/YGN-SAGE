"""R9 OracleStack v0 acceptance tests (16 contract tests)."""
from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from sage.pipeline import CognitiveOrchestrationPipeline
from sage.runtime.evidence import (
    RuntimeDelta,
    produce_test_parser_deltas,
    produce_tool_deltas,
)
from sage.runtime.oracle import (
    ORACLE_VERDICT_SCHEMA_VERSION,
    EvidenceRef,
    OracleConfig,
    OracleVerdict,
    evaluate,
)
from sage.runtime.state import StateFrame


@pytest.fixture
def tmp_path() -> pathlib.Path:
    path = pathlib.Path(".tmp") / "pytest-oracle-stack" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@dataclass(frozen=True)
class FakeRunFrameView:
    run_id: str = "01ORACLEVIEW0000000000001"
    final_result_seq: int | None = 7
    runtime_deltas: tuple[RuntimeDelta, ...] = ()
    state_frames: Mapping[str, StateFrame] = field(default_factory=dict)
    node_records: Mapping[str, Any] = field(default_factory=dict)


class FakeGraph:
    id = "oracle-topology"
    template_type = "single_agent"

    def node_count(self) -> int:
        return 1

    def get_node(self, idx: int) -> Any:
        return SimpleNamespace(
            role="solo",
            model_id="model-a",
            provider_id="provider-a",
            system=1,
            required_capabilities=[],
        )

    def get_predecessors(self, idx: int) -> list[int]:
        return []

    def get_edges(self) -> list[tuple[int, int, str]]:
        return []


class _RustRouterProxy:
    """A14b: wire _rust_router so record_outcome_checked reaches the test bandit."""

    def __init__(self, bandit: Any) -> None:
        self._bandit = bandit

    def route_integrated(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            decision_id="decision-1",
            model_id="model-a",
            template="single_agent",
            selected_template="single_agent",
            system=1,
            confidence=0.9,
            estimated_cost=0.001,
        )

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        if self._bandit is not None:
            self._bandit.record_outcome_checked(
                decision_id, executed_model_id, executed_template, quality, cost, latency_ms
            )

    def cancel_bandit_decision(self, decision_id: str) -> bool:
        if self._bandit is not None and hasattr(self._bandit, "cancel_decision"):
            return bool(self._bandit.cancel_decision(decision_id))
        return True


class RecordingBandit:
    def __init__(self) -> None:
        self.checked_records: list[tuple[str, str, str, float, float, float]] = []
        self.cancelled: list[str] = []

    def record_outcome_checked(
        self,
        decision_id: str,
        executed_model_id: str,
        executed_template: str,
        quality: float,
        cost: float,
        latency_ms: float,
    ) -> None:
        self.checked_records.append(
            (decision_id, executed_model_id, executed_template, quality, cost, latency_ms)
        )

    def cancel_decision(self, decision_id: str) -> bool:
        self.cancelled.append(decision_id)
        return True

    def record_outcome(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unchecked bandit.record_outcome must not be called")


class RecordingEngine:
    def __init__(self, *, should_evolve: bool = False) -> None:
        self.record_outcome_calls = 0
        self.should_evolve_calls = 0
        self.evolve_calls = 0
        self._cells = 0
        self._should_evolve = should_evolve

    def record_outcome(self, *_args: Any, **_kwargs: Any) -> None:
        self.record_outcome_calls += 1
        self._cells += 1

    def archive_cell_count(self) -> int:
        return self._cells

    def archive_coverage(self) -> float:
        return 0.1 if self._cells else 0.0

    def should_evolve(self, *_args: Any, **_kwargs: Any) -> bool:
        self.should_evolve_calls += 1
        return self._should_evolve

    def evolve(self, *_args: Any, **_kwargs: Any) -> None:
        self.evolve_calls += 1


class RecordingEpisodicMemory:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    def add(self, **kwargs: Any) -> None:
        self.add_calls.append(kwargs)


def _read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_pipeline(
    *,
    output: str = "oracle output",
    bench_result: Mapping[str, Any] | None = None,
    bandit: Any | None = None,
    engine: Any | None = None,
    episodic_memory: Any | None = None,
    consolidator: Any | None = None,
    quality_estimator: Any | None = None,
) -> CognitiveOrchestrationPipeline:
    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=engine,
        assigner=None,
        provider_pool=None,
        bandit=bandit,
        quality_estimator=quality_estimator,
        llm_provider=MagicMock(),
        episodic_memory=episodic_memory,
        consolidator=consolidator,
    )

    def _classify(ctx: Any) -> Any:
        ctx.system = 1
        ctx.domain = "code"
        return ctx

    async def _decompose(ctx: Any) -> Any:
        return ctx

    def _select(ctx: Any) -> Any:
        ctx.topology = FakeGraph()
        ctx.topology_id = "oracle-topology"
        return ctx

    def _assign(ctx: Any) -> Any:
        ctx.assignments = {0: "model-a"}
        ctx.provider_hints = {0: "provider-a"}
        ctx.bandit_decision_id = "decision-1"
        ctx.bandit_model_id = "model-a"
        ctx.bandit_template = "single_agent"
        return ctx

    async def _execute(ctx: Any, **_kwargs: Any) -> Any:
        ctx.result = output
        ctx.cost = 0.25
        ctx.executed_model_id = "model-a"
        ctx.executed_template = "single_agent"
        ctx.bench_result = bench_result
        return ctx

    pipeline._stage_classify = _classify  # type: ignore[method-assign]
    pipeline._stage_decompose = _decompose  # type: ignore[method-assign]
    pipeline._stage_select_topology = _select  # type: ignore[method-assign]
    pipeline._stage_assign_models = _assign  # type: ignore[method-assign]
    pipeline._stage_execute = _execute  # type: ignore[method-assign]
    # A14b: _rust_router handles record_outcome_checked; wire proxy to bandit.
    if bandit is not None:
        pipeline._rust_router = _RustRouterProxy(bandit)
    return pipeline


def test_evaluate_exact_pass_returns_trainable_true() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="accepted",
        bench_result={
            "passed": True,
            "score": 1.0,
            "tool_call_id": "tool-1",
            "verifier_id": "bench",
            "output_sha256": "abc123",
        },
    )

    assert verdict.schema_version == ORACLE_VERDICT_SCHEMA_VERSION
    assert verdict.trainable is True
    assert verdict.verdict_source == "exact"
    assert verdict.quality_label == "pass"
    assert verdict.score == 1.0
    assert verdict.evidence[0].event_seq == 7
    assert verdict.evidence[0].tool_call_id == "tool-1"


def test_evaluate_exact_fail_returns_trainable_true_negative() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="rejected",
        bench_result={"passed": False, "score": 0.0, "verifier_id": "bench"},
    )

    assert verdict.trainable is True
    assert verdict.verdict_source == "exact"
    assert verdict.quality_label == "fail"
    assert verdict.score == 0.0
    assert verdict.reason_codes == ("exact_test_fail",)


def test_evaluate_no_evidence_returns_abstain() -> None:
    verdict = evaluate(FakeRunFrameView(), final_output="plain output")

    assert verdict.trainable is False
    assert verdict.verdict_source == "abstain"
    assert verdict.quality_label == "unknown"
    assert verdict.score is None


# cgpro 2026-04-29 cycle-7 flip review push-back: regression tests pinning
# that ``_exact_oracle`` never propagates raw bench_result["reason"] into
# oracle_verdict.reason_codes (would leak harness traceback / stderr into
# JSONL traces).


def test_exact_oracle_raw_reason_does_not_leak_into_reason_codes() -> None:
    raw_traceback = (
        "Traceback (most recent call last):\n"
        "  File \"/tmp/test.py\", line 42, in test_foo\n"
        "    self.assertEqual(actual, expected)\n"
        "AssertionError: 1 != 2\n"
        "SECRET_RAW_TEST_OUTPUT_canary"
    )
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="bad",
        bench_result={
            "passed": False,
            "score": 0.0,
            "verifier_id": "bench",
            "reason": raw_traceback,
        },
    )

    assert verdict.trainable is True
    assert verdict.verdict_source == "exact"
    assert verdict.quality_label == "fail"
    # reason_codes must contain only the structured tags, not the raw text.
    assert verdict.reason_codes == (
        "exact_test_fail",
        "exact_harness_detail_present",
    )
    for code in verdict.reason_codes:
        assert "Traceback" not in code
        assert "AssertionError" not in code
        assert "SECRET_RAW_TEST_OUTPUT_canary" not in code
    # Hash carries the audit pointer instead.
    assert verdict.evidence[0].evidence_hash is not None
    assert len(verdict.evidence[0].evidence_hash) == 64  # SHA-256 hex


def test_exact_oracle_pre_classified_reason_code_passes_through() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="bad",
        bench_result={
            "passed": False,
            "score": 0.0,
            "verifier_id": "bench",
            "reason_code": "bcb_unittest_fail",
        },
    )

    assert verdict.reason_codes == ("exact_test_fail", "bcb_unittest_fail")
    # No raw reason ⇒ no harness_detail_present sentinel
    assert "exact_harness_detail_present" not in verdict.reason_codes
    # No reason_sha256 supplied ⇒ evidence_hash stays None
    assert verdict.evidence[0].evidence_hash is None


def test_exact_oracle_pre_supplied_reason_sha256_takes_precedence() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="bad",
        bench_result={
            "passed": False,
            "score": 0.0,
            "verifier_id": "bench",
            "reason_code": "bcb_timeout",
            "reason_sha256": "a" * 64,
        },
    )

    assert verdict.evidence[0].evidence_hash == "a" * 64
    assert verdict.reason_codes == ("exact_test_fail", "bcb_timeout")


def test_exact_oracle_reason_code_with_raw_text_is_rejected() -> None:
    """cgpro 2026-04-29 flip approval hardening: bench seams must not put
    raw stderr or traceback into ``bench_result["reason_code"]``. Anything
    not matching the structured-token regex is replaced with the sentinel
    ``exact_reason_code_rejected`` so the audit log still surfaces the
    rejection.
    """
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="bad",
        bench_result={
            "passed": False,
            "score": 0.0,
            "verifier_id": "bench",
            "reason_code": "Traceback (most recent call last)\n  File \"x.py\"",
        },
    )

    assert verdict.reason_codes == (
        "exact_test_fail",
        "exact_reason_code_rejected",
    )
    for code in verdict.reason_codes:
        assert "Traceback" not in code
        assert '  File "' not in code


def test_exact_oracle_non_string_reason_code_is_rejected() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="bad",
        bench_result={
            "passed": False,
            "score": 0.0,
            "verifier_id": "bench",
            "reason_code": 12345,  # int, not a string token
        },
    )
    assert "exact_reason_code_rejected" in verdict.reason_codes


@pytest.mark.asyncio
async def test_abstain_blocks_bandit_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    bandit = RecordingBandit()
    pipeline = _make_pipeline(
        output="non-verifiable output",
        bandit=bandit,
        quality_estimator=MagicMock(),
    )

    await pipeline.run("task")

    assert bandit.checked_records == []


@pytest.mark.asyncio
async def test_exact_pass_triggers_bandit_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    bandit = RecordingBandit()
    pipeline = _make_pipeline(
        bench_result={"passed": True, "score": 1.0},
        bandit=bandit,
    )

    await pipeline.run("task")

    assert [record[3] for record in bandit.checked_records] == [1.0]


@pytest.mark.asyncio
async def test_exact_fail_triggers_bandit_update_with_zero_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    bandit = RecordingBandit()
    pipeline = _make_pipeline(
        bench_result={"passed": False, "score": 0.0},
        bandit=bandit,
    )

    await pipeline.run("task")

    assert [record[3] for record in bandit.checked_records] == [0.0]


@pytest.mark.asyncio
async def test_oracle_exception_returns_abstain_does_not_change_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sage.runtime.oracle as oracle_module

    monkeypatch.setenv("SAGE_ORACLE", "1")
    monkeypatch.setattr(
        oracle_module,
        "evaluate",
        MagicMock(side_effect=RuntimeError("oracle unavailable")),
    )
    result, frame = await _make_pipeline(output="final answer").run_with_frame("task")

    assert result == "final answer"
    assert frame.oracle_verdict is not None
    assert frame.oracle_verdict.verdict_source == "abstain"
    assert frame.oracle_verdict.trainable is False


def test_hierarchy_returns_first_confident_verdict() -> None:
    view = FakeRunFrameView(
        state_frames={
            "node-1": StateFrame(
                task_id="task",
                invalidated_assumptions=("ASSUMPTION_X",),
            )
        }
    )

    verdict = evaluate(
        view,
        final_output="ASSUMPTION_X is still true",
        bench_result={"passed": True, "score": 1.0},
    )

    assert verdict.verdict_source == "exact"
    assert verdict.quality_label == "pass"
    assert "state_contradiction" not in verdict.reason_codes


def test_lexical_fallback_off_by_default_means_abstain() -> None:
    verdict = evaluate(
        FakeRunFrameView(),
        final_output="pytest passed, all tests green",
        config=OracleConfig(enable_lexical_fallback=False),
    )

    assert verdict.verdict_source == "abstain"
    assert verdict.trainable is False


def _pytest_delta(stdout: str) -> RuntimeDelta:
    result = produce_test_parser_deltas(
        run_id="01ORACLEVIEW0000000000001",
        node_run_id="node-tests",
        event_seq=3,
        source_id="pytest:unit",
        framework="pytest",
        stdout=stdout,
        stderr="",
        exit_code=0,
        suite_id="unit",
    )
    assert result.rejected_reason is None
    return result.deltas[0]


def _legacy_pytest_v0_pass_delta() -> RuntimeDelta:
    return RuntimeDelta(
        schema_version="0",
        producer="test_parser",
        delta_kind="tests_passed",
        polarity="positive",
        confidence=1.0,
        run_id="01ORACLEVIEW0000000000001",
        node_run_id="node-tests",
        event_seq=3,
        source_id="pytest:unit",
        payload={
            "framework": "pytest",
            "parser_id": "pytest_summary_v0",
            "suite_id": "unit",
            "passed_count": 1,
            "failed_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "duration_ms": 100.0,
        },
    )


def _tool_fatal_delta(fatal_scope: str) -> RuntimeDelta:
    result = produce_tool_deltas(
        run_id="01ORACLEVIEW0000000000001",
        node_run_id="node-tool",
        event_seq=4,
        source_id="tool:runner",
        exit_code=-1,
        timed_out=False,
        duration_ms=2.0,
        tool_error_class="RuntimeError",
        fatal_scope=fatal_scope,
    )
    assert result.rejected_reason is None
    return result.deltas[0]


def test_tool_oracle_accepts_pytest_summary_v1_parser() -> None:
    parser_pass = _pytest_delta("==== 1 passed in 0.10s ====")

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_pass,)),
        final_output="tests passed",
    )

    assert verdict.verdict_source == "tool"
    assert verdict.quality_label == "pass"
    assert verdict.trainable is True


def test_tool_oracle_keeps_pytest_summary_v0_backcompat() -> None:
    parser_pass = _legacy_pytest_v0_pass_delta()

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_pass,)),
        final_output="legacy parser pass",
    )

    assert verdict.verdict_source == "tool"
    assert verdict.quality_label == "pass"
    assert verdict.trainable is True


def test_tool_oracle_parser_pass_with_incidental_fatal_still_passes() -> None:
    parser_pass = _pytest_delta("==== 1 passed in 0.10s ====")
    incidental_fatal = _tool_fatal_delta("incidental_tool_call")

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_pass, incidental_fatal)),
        final_output="tests passed despite incidental tool failure",
    )

    assert verdict.verdict_source == "tool"
    assert verdict.quality_label == "pass"
    assert verdict.trainable is True


def test_tool_oracle_parser_pass_with_unknown_fatal_abstains() -> None:
    parser_pass = _pytest_delta("==== 1 passed in 0.10s ====")
    unknown_fatal = _tool_fatal_delta("unknown")

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_pass, unknown_fatal)),
        final_output="tests passed with unscoped fatal",
    )

    assert verdict.verdict_source == "abstain"
    assert verdict.trainable is False


def test_tool_oracle_parser_pass_with_claimed_task_output_fatal_abstains() -> None:
    parser_pass = _pytest_delta("==== 1 passed in 0.10s ====")
    scoped_fatal = _tool_fatal_delta("claimed_task_output")

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_pass, scoped_fatal)),
        final_output="tests passed but artifact fatal invalidates confidence",
    )

    assert verdict.verdict_source == "abstain"
    assert verdict.trainable is False


def test_tool_oracle_parser_fail_with_incidental_fatal_still_fails() -> None:
    parser_fail = _pytest_delta("==== 1 failed in 0.10s ====")
    incidental_fatal = _tool_fatal_delta("incidental_tool_call")

    verdict = evaluate(
        FakeRunFrameView(runtime_deltas=(parser_fail, incidental_fatal)),
        final_output="tests failed",
    )

    assert verdict.verdict_source == "tool"
    assert verdict.quality_label == "fail"
    assert verdict.trainable is True


@pytest.mark.asyncio
async def test_oracle_verdict_event_emitted_with_parent_seq(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "01ORACLEEVENT000000000001"
    monkeypatch.setenv("SAGE_ORACLE", "1")
    monkeypatch.setenv("SAGE_RUN_FRAME", "1")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: run_id)

    result, frame = await _make_pipeline(
        bench_result={"passed": True, "score": 1.0}
    ).run_with_frame("task")

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    final = next(event for event in events if event["event_type"] == "final_result")
    oracle = next(event for event in events if event["event_type"] == "oracle_verdict")
    summary = next(event for event in events if event["event_type"] == "run_frame_summary")
    assert result == "oracle output"
    assert final["seq"] < oracle["seq"] < summary["seq"]
    assert oracle["parent_event_id"] == final["seq"]
    assert oracle["payload"]["trainable"] is True
    assert frame.oracle_verdict is not None


def test_run_frame_includes_oracle_verdict_when_present() -> None:
    from sage.runtime.event_log.redaction import _hash_text
    from sage.runtime.run_frame.builder import _RunFrameBuilder

    builder = _RunFrameBuilder(
        run_id="01ORACLEFRAME000000000001",
        task_id="01ORACLEFRAME000000000001",
        task_hash=_hash_text("task"),
    )
    verdict = OracleVerdict(
        trainable=True,
        verdict_source="exact",
        quality_label="pass",
        score=1.0,
        confidence=1.0,
        reason_codes=("exact_test_pass",),
        evidence=(EvidenceRef(run_id=builder.run_id),),
    )
    builder.record_oracle_verdict(seq=12, verdict=verdict)

    frame = builder.finalize()
    summary = frame.to_summary_dict()

    assert frame.oracle_verdict == verdict
    assert summary["oracle_verdict"]["verdict_source"] == "exact"
    assert summary["oracle_verdict"]["schema_version"] == "0"


@pytest.mark.asyncio
async def test_oracle_off_via_killswitch_preserves_legacy_behavior(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post cycle-7 default-on flip: SAGE_ORACLE unset is now ON (default).
    The legacy "oracle skipped" path is reached by the explicit kill-switch
    ``SAGE_ORACLE=0``. This test pins the kill-switch contract.
    """
    run_id = "01ORACLEOFF0000000000001"
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_TRACE_JSONL_DIR", str(tmp_path))
    monkeypatch.setattr("sage.pipeline._new_runtime_run_id", lambda: run_id)
    bandit = RecordingBandit()

    result, frame = await _make_pipeline(
        output="legacy output",
        bandit=bandit,
        quality_estimator=MagicMock(return_value=None),
    ).run_with_frame("task")

    events = _read_events(tmp_path / f"{run_id}.jsonl")
    assert result == "legacy output"
    assert frame.oracle_verdict is None
    assert "oracle_verdict" not in [event["event_type"] for event in events]


def test_evidence_required_when_trainable_post_init_validation() -> None:
    with pytest.raises(ValueError, match="at least one EvidenceRef"):
        OracleVerdict(
            trainable=True,
            verdict_source="exact",
            quality_label="pass",
            score=1.0,
            confidence=1.0,
            reason_codes=("exact_test_pass",),
            evidence=(),
        )

    with pytest.raises(ValueError, match="trainable=False"):
        OracleVerdict(
            trainable=False,
            verdict_source="exact",
            quality_label="unknown",
            score=None,
            confidence=1.0,
            reason_codes=("not_training",),
            evidence=(EvidenceRef(run_id="run"),),
        )


@pytest.mark.asyncio
async def test_episodic_memory_records_abstain_with_training_flag_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    episodic = RecordingEpisodicMemory()

    await _make_pipeline(
        output="unverified output",
        episodic_memory=episodic,
    ).run("task")

    assert len(episodic.add_calls) == 1
    content = json.loads(episodic.add_calls[0]["content"])
    assert content["is_training_evidence"] is False


def test_spec_oracle_lexical_substring_does_not_train() -> None:
    """cgpro 2026-04-29 R6.1a verify push-back: the lexical substring path
    (assumption_id ∈ output_text + invalidation-marker check) is removed.

    For v1, spec oracle abstains unless a structured claim-dependency channel
    proves the final output reasserts the invalidated assumption. Until that
    channel exists (cycle 7+), every state_frame + final_output pair MUST
    collapse to abstain — both the "reassertion" wording (was trainable=True
    in R9) AND the "mention-as-invalidated" wording (was abstain in R9).
    """
    view = FakeRunFrameView(
        state_frames={
            "node-a": StateFrame(
                task_id="task",
                invalidated_assumptions=("ASSUMPTION_42",),
            )
        }
    )

    outputs = [
        # Previously trainable=True in R9 (reassertion) — now abstain.
        "We can rely on ASSUMPTION_42.",
        # Previously abstain in R9 (mention-as-invalidated) — still abstain.
        "We learned that ASSUMPTION_42 was wrong, so we tried a different approach.",
        "ASSUMPTION_42 is no longer valid; revising plan.",
        "ASSUMPTION_42 was retracted by the verifier.",
        "ASSUMPTION_42 was invalidated upstream; pursuing alternative.",
        "ASSUMPTION_42 was found to be incorrect.",
        "ASSUMPTION_42 doesn't hold under the new constraints.",
        # No claim-dependency channel ⇒ no trainable spec verdict regardless
        # of phrasing. The substring scan is gone for both directions.
    ]
    for output in outputs:
        verdict = evaluate(view, final_output=output)
        assert verdict.verdict_source == "abstain", (
            f"output {output!r} must NOT produce a trainable spec verdict in "
            f"R6.1a v1 (no structured claim-dependency channel exists yet); "
            f"got verdict_source={verdict.verdict_source!r}"
        )
        assert verdict.trainable is False, (
            f"output {output!r} must collapse to trainable=False; "
            f"got trainable={verdict.trainable}"
        )


def test_oracle_v0_evidence_starved_default_falls_through_to_abstain() -> None:
    """R9.0.1 (cgpro 2026-04-29 cycle 5 reassess): documents the v0
    evidence-starved state explicitly. With SAGE_ORACLE=1 but NO evidence
    sources active (no bench_result, Tool/Formal oracles return None v0,
    no StateCore contradiction, LLMJudge stubbed), the hierarchy MUST fall
    through to Abstain — and Abstain MUST gate all downstream training.

    This test pins the intended R6.1a insertion point: when R6.1a deterministic
    delta producers ship, Tool/Formal oracles will start returning real
    verdicts, and runs that previously hit Abstain may become trainable. Until
    then, every non-bench path is starved-Abstain by design.

    Documents the cycle 5→6 handoff state.
    """
    view = FakeRunFrameView()  # default: no state_frames, no bench evidence

    # No bench result. No state contradiction. Tool + Formal oracles are
    # placeholders that return None in v0. LLMJudge stubbed.
    verdict = evaluate(view, final_output="any free-form output", bench_result=None)

    assert verdict.verdict_source == "abstain", (
        f"v0 evidence-starved evaluate must Abstain; got verdict_source="
        f"{verdict.verdict_source!r}. If this assertion fails after R6.1a "
        f"ships, update the test rationale to reflect the new evidence sources."
    )
    assert verdict.trainable is False
    assert verdict.score is None
    assert verdict.quality_label == "unknown"

    # Reason codes should reflect "hierarchy exhausted" or similar — the
    # placeholder fallthrough path. Don't lock the exact string; just check
    # it's one of the expected v0 abstain reasons.
    assert verdict.reason_codes, "Abstain verdict must carry at least one reason_code"
    abstain_reasons = {"hierarchy_exhausted", "hierarchy_low_confidence_only"}
    assert any(rc in abstain_reasons for rc in verdict.reason_codes), (
        f"v0 evidence-starved abstain expected one of {abstain_reasons}, "
        f"got reason_codes={verdict.reason_codes!r}"
    )


@pytest.mark.asyncio
async def test_abstain_blocks_all_training_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    from sage.constants import CONSOLIDATION_INTERVAL_STEPS

    monkeypatch.setenv("SAGE_ORACLE", "1")
    bandit = RecordingBandit()
    engine = RecordingEngine(should_evolve=True)
    consolidator = SimpleNamespace(consolidate=AsyncMock())
    pipeline = _make_pipeline(
        output="unverified output",
        bandit=bandit,
        engine=engine,
        consolidator=consolidator,
        quality_estimator=MagicMock(),
    )
    pipeline._task_count = CONSOLIDATION_INTERVAL_STEPS - 1
    pipeline._record_bandit_outcome_checked = MagicMock(
        wraps=pipeline._record_bandit_outcome_checked
    )

    await pipeline.run("task")

    assert pipeline._record_bandit_outcome_checked.call_count == 0
    assert bandit.checked_records == []
    assert engine.record_outcome_calls == 0
    assert engine.should_evolve_calls == 0
    assert engine.evolve_calls == 0
    consolidator.consolidate.assert_not_awaited()
