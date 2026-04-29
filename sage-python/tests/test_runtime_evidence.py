"""R6.1a deterministic RuntimeDelta producer acceptance tests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pytest

from sage.agent_loop_execution import execute_tool_call
from sage.runtime.evidence import (
    EvidenceError,
    RuntimeDelta,
    produce_code_node_deltas,
    produce_diff_verifier_deltas,
    produce_formal_verifier_deltas,
    produce_planner_deltas,
    produce_test_parser_deltas,
    produce_tool_deltas,
)
from sage.runtime.evidence.delta import RUNTIME_DELTA_SCHEMA_VERSION
from sage.runtime.oracle import evaluate
from sage.runtime.run_frame.builder import _RunFrameBuilder


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "runtime_evidence"


@dataclass(frozen=True)
class FakeRunFrameView:
    run_id: str = "run-oracle"
    final_result_seq: int | None = 99
    runtime_deltas: tuple[RuntimeDelta, ...] = ()
    state_frames: Mapping[str, Any] = field(default_factory=dict)
    node_records: Mapping[str, Any] = field(default_factory=dict)


def _canonical(value: Any) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _evidence_hash(delta: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "schema_version": delta["schema_version"],
            "producer": delta["producer"],
            "delta_kind": delta["delta_kind"],
            "polarity": delta["polarity"],
            "source_id": delta["source_id"],
            "payload": delta["payload"],
        }
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _delta_to_plain(delta: RuntimeDelta) -> dict[str, Any]:
    return {
        "schema_version": delta.schema_version,
        "producer": delta.producer,
        "delta_kind": delta.delta_kind,
        "polarity": delta.polarity,
        "confidence": delta.confidence,
        "run_id": delta.run_id,
        "node_run_id": delta.node_run_id,
        "event_seq": delta.event_seq,
        "source_id": delta.source_id,
        "evidence_hash": delta.evidence_hash,
        "payload": _plain(delta.payload),
    }


def _fixture(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = json.loads((FIXTURE_DIR / f"{name}.input.json").read_text())
    expected = json.loads((FIXTURE_DIR / f"{name}.expected.json").read_text())
    return input_payload, expected


def _materialize_expected(
    expected: dict[str, Any],
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    out = json.loads(json.dumps(expected))
    return_hash = _sha256_json(input_payload.get("kwargs", {}).get("return_value", {}))
    for delta in out["deltas"]:
        payload = delta["payload"]
        if payload.get("return_hash") == "$RETURN_HASH":
            payload["return_hash"] = return_hash
        if delta.get("evidence_hash") == "$AUTO":
            delta["evidence_hash"] = _evidence_hash(delta)
    return out


def _produce_from_fixture(input_payload: Mapping[str, Any]):
    kwargs = dict(input_payload["kwargs"])
    producer = input_payload["producer"]
    if producer == "tool_execution":
        return produce_tool_deltas(**kwargs)
    if producer == "test_parser":
        return produce_test_parser_deltas(**kwargs)
    if producer == "diff_verifier":
        return produce_diff_verifier_deltas(**kwargs)
    if producer == "formal_verifier":
        return produce_formal_verifier_deltas(**kwargs)
    if producer == "code_node_return":
        return produce_code_node_deltas(**kwargs)
    raise AssertionError(f"unknown fixture producer {producer!r}")


PRODUCER_FIXTURES = (
    "tool_execution_exit_zero",
    "tool_execution_exit_nonzero",
    "tool_execution_timed_out",
    "tool_execution_fatal_failure",
    "pytest_pass",
    "pytest_fail",
    "pytest_partial",
    "pytest_parse_failed",
    "diff_clean",
    "diff_hunk_header_mismatch",
    "diff_repair_accepted",
    "formal_obligation_proved",
    "formal_obligation_refuted",
    "formal_unknown",
    "code_node_structured_return_valid",
    "code_node_structured_return_invalid",
)


class TestProducerContracts:
    @pytest.mark.parametrize("fixture_name", PRODUCER_FIXTURES)
    def test_fixture_producer_outputs_exact_runtime_delta_shape(
        self,
        fixture_name: str,
    ) -> None:
        input_payload, expected_raw = _fixture(fixture_name)
        expected = _materialize_expected(expected_raw, input_payload)

        result = _produce_from_fixture(input_payload)

        assert result.rejected_reason is None
        assert {"deltas": [_delta_to_plain(delta) for delta in result.deltas]} == expected

    def test_planner_scaffold_records_structural_decision_only(self) -> None:
        result = produce_planner_deltas(
            run_id="run-planner",
            node_run_id="node-planner",
            event_seq=3,
            source_id="planner:topology",
            decision_type="topology_selected",
            topology_id="topo-a",
            graph_digest="digest-a",
            node_count=4,
        )

        assert result.rejected_reason is None
        delta = result.deltas[0]
        assert delta.producer == "planner_decision"
        assert delta.delta_kind == "topology_selected"
        assert delta.polarity == "neutral"
        assert "message" not in delta.payload


class TestSemanticNegatives:
    def test_runtime_delta_rejects_unknown_kind(self) -> None:
        with pytest.raises(EvidenceError, match="delta_kind"):
            RuntimeDelta(
                schema_version="0",
                producer="tool_execution",
                delta_kind="not_real",
                polarity="neutral",
                confidence=1.0,
                run_id="run",
                source_id="source",
                payload={},
            )

    def test_runtime_delta_rejects_illegal_polarity_for_kind(self) -> None:
        with pytest.raises(EvidenceError, match="polarity"):
            RuntimeDelta(
                schema_version="0",
                producer="test_parser",
                delta_kind="tests_failed",
                polarity="positive",
                confidence=1.0,
                run_id="run",
                source_id="pytest",
                payload={"framework": "pytest", "parser_id": "pytest_summary_v0"},
            )

    def test_runtime_delta_rejects_oversized_payload_string(self) -> None:
        with pytest.raises(EvidenceError, match="exceeds"):
            RuntimeDelta(
                schema_version="0",
                producer="tool_execution",
                delta_kind="fatal_failure",
                polarity="negative",
                confidence=1.0,
                run_id="run",
                source_id="tool",
                payload={"tool_error_class": "x" * 300},
            )

    def test_runtime_delta_rejects_forbidden_raw_payload_keys(self) -> None:
        with pytest.raises(EvidenceError, match="forbidden"):
            RuntimeDelta(
                schema_version="0",
                producer="tool_execution",
                delta_kind="exit_zero",
                polarity="positive",
                confidence=1.0,
                run_id="run",
                source_id="tool",
                payload={"stdout": "secret"},
            )

    def test_unknown_parser_produces_no_delta_and_rejected_reason(self) -> None:
        result = produce_test_parser_deltas(
            run_id="run",
            node_run_id=None,
            event_seq=1,
            source_id="tests",
            framework="nose",
            stdout="OK",
            stderr="",
            exit_code=0,
        )

        assert result.deltas == ()
        assert result.rejected_reason is not None

    def test_missing_code_node_schema_produces_no_delta(self) -> None:
        result = produce_code_node_deltas(
            run_id="run",
            node_run_id="node",
            event_seq=1,
            source_id="code-node",
            return_value={"status": "ok"},
            return_schema=None,
        )

        assert result.deltas == ()
        assert result.rejected_reason is not None


class TestOracleConsumption:
    def test_tool_oracle_parser_pass_returns_trainable_positive(self) -> None:
        delta = produce_test_parser_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="pytest",
            framework="pytest",
            stdout="=== 3 passed in 0.10s ===",
            stderr="",
            exit_code=0,
        ).deltas[0]

        verdict = evaluate(FakeRunFrameView(runtime_deltas=(delta,)), final_output="ok")

        assert verdict.verdict_source == "tool"
        assert verdict.quality_label == "pass"
        assert verdict.score == 1.0
        assert verdict.evidence[0].evidence_hash == delta.evidence_hash

    def test_tool_oracle_parser_partial_returns_ratio(self) -> None:
        delta = produce_test_parser_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="pytest",
            framework="pytest",
            stdout="=== 8 passed, 2 failed in 0.10s ===",
            stderr="",
            exit_code=1,
        ).deltas[0]

        verdict = evaluate(FakeRunFrameView(runtime_deltas=(delta,)), final_output="partial")

        assert verdict.verdict_source == "tool"
        assert verdict.quality_label == "partial"
        assert verdict.score == pytest.approx(0.8)

    def test_tool_oracle_generic_nonzero_without_parser_abstains(self) -> None:
        delta = produce_tool_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="tool:cmd",
            exit_code=2,
            timed_out=False,
            duration_ms=12.0,
        ).deltas[0]

        verdict = evaluate(FakeRunFrameView(runtime_deltas=(delta,)), final_output="err")

        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False

    def test_formal_oracle_obligation_refuted_returns_negative(self) -> None:
        delta = produce_formal_verifier_deltas(
            run_id="run-oracle",
            node_run_id="node-f",
            event_seq=9,
            source_id="z3",
            delta_kind="obligation_refuted",
            obligation_id="obl-1",
            obligation_type="postcondition",
            verifier_id="z3",
            solver_status="sat",
            encoding="find_counterexample",
        ).deltas[0]

        verdict = evaluate(FakeRunFrameView(runtime_deltas=(delta,)), final_output="bad")

        assert verdict.verdict_source == "formal"
        assert verdict.quality_label == "fail"
        assert verdict.score == 0.0
        assert verdict.evidence[0].evidence_hash == delta.evidence_hash

    def test_formal_oracle_contradictory_same_obligation_abstains(self) -> None:
        proved = produce_formal_verifier_deltas(
            run_id="run-oracle",
            node_run_id="node-f",
            event_seq=9,
            source_id="z3",
            delta_kind="obligation_proved",
            obligation_id="obl-1",
            obligation_type="postcondition",
            verifier_id="z3",
            solver_status="unsat",
            encoding="prove_no_counterexample",
        ).deltas[0]
        refuted = produce_formal_verifier_deltas(
            run_id="run-oracle",
            node_run_id="node-f",
            event_seq=10,
            source_id="z3",
            delta_kind="counterexample_found",
            obligation_id="obl-1",
            obligation_type="postcondition",
            verifier_id="z3",
            solver_status="sat",
            encoding="find_counterexample",
        ).deltas[0]

        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(proved, refuted)),
            final_output="mixed",
        )

        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False

    def test_tool_unavailable_abstains(self) -> None:
        delta = produce_tool_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="tool:missing",
            exit_code=127,
            timed_out=False,
            duration_ms=1.0,
            tool_error_class="ToolUnavailable",
        ).deltas[0]

        verdict = evaluate(FakeRunFrameView(runtime_deltas=(delta,)), final_output="missing")

        assert verdict.verdict_source == "abstain"


class TestRawOutputSafety:
    def test_forbidden_raw_payload_names_never_enter_runtime_delta(self) -> None:
        for forbidden_key in (
            "stdout",
            "stderr",
            "raw_output",
            "raw_patch",
            "diff",
            "final_answer",
            "message",
            "traceback",
        ):
            with pytest.raises(EvidenceError, match="forbidden"):
                RuntimeDelta(
                    schema_version="0",
                    producer="tool_execution",
                    delta_kind="exit_zero",
                    polarity="positive",
                    confidence=1.0,
                    run_id="run",
                    source_id="tool",
                    payload={forbidden_key: "secret"},
                )


class TestOffModeStability:
    @pytest.mark.asyncio
    async def test_code_node_runtime_skips_tool_producer_when_oracle_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sage.topology.runner import TopologyRunner

        monkeypatch.delenv("SAGE_ORACLE", raising=False)
        monkeypatch.setenv("SAGE_UNSAFE_RAW_EXEC", "1")

        def _boom(**_kwargs: Any) -> Any:
            raise AssertionError("producer must not be called when SAGE_ORACLE is unset")

        monkeypatch.setattr(
            "sage.runtime.evidence.producers.tool.produce_tool_deltas",
            _boom,
        )

        class CodeGraph:
            def get_node(self, _idx: int) -> Any:
                return SimpleNamespace(
                    role="code",
                    code_spec='print("ok")',
                    model_id="",
                    provider_id="",
                    system=1,
                )

            def get_predecessors(self, _idx: int) -> list[int]:
                return []

        builder = _RunFrameBuilder(run_id="run-code", task_id="task", task_hash="hash")
        runner = TopologyRunner(
            graph=CodeGraph(),
            executor=SimpleNamespace(),
            llm_provider=SimpleNamespace(),
            run_frame_builder=builder,
        )

        result = await runner._execute_code_node(0, "task")

        assert "ok" in result
        assert builder.finalize().runtime_deltas == ()

    @pytest.mark.asyncio
    async def test_agent_loop_tool_path_skips_producer_when_oracle_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SAGE_ORACLE", raising=False)

        def _boom(**_kwargs: Any) -> Any:
            raise AssertionError("producer must not be called when SAGE_ORACLE is unset")

        monkeypatch.setattr(
            "sage.runtime.evidence.producers.tool.produce_tool_deltas",
            _boom,
        )

        class ToolRegistry:
            def get(self, _name: str) -> Any:
                return SimpleNamespace(
                    execute=lambda _kwargs: _async_result(SimpleNamespace(output="ok")),
                )

        builder = _RunFrameBuilder(run_id="run-tool", task_id="task", task_hash="hash")
        output = await execute_tool_call(
            SimpleNamespace(name="demo", arguments={}, id="call-1"),
            ToolRegistry(),
            lambda *_args, **_kwargs: None,
            run_frame_builder=builder,
            node_run_id="node-tool",
            source_id="tool:demo",
        )

        assert output == "ok"
        assert builder.finalize().runtime_deltas == ()


async def _async_result(value: Any) -> Any:
    return value


class TestRunFrameDeltas:
    def test_builder_record_delta_freezes_payload_and_snapshot(self) -> None:
        builder = _RunFrameBuilder(run_id="run-frame", task_id="task", task_hash="hash")
        mutable_payload = {"exit_code": 0, "timed_out": False}
        delta = RuntimeDelta(
            schema_version="0",
            producer="tool_execution",
            delta_kind="exit_zero",
            polarity="positive",
            confidence=1.0,
            run_id="run-frame",
            source_id="tool",
            payload=mutable_payload,
        )

        builder.record_delta(delta)
        mutable_payload["exit_code"] = 99
        view = builder.snapshot_view()

        assert isinstance(view.runtime_deltas, tuple)
        assert isinstance(view.runtime_deltas[0].payload, MappingProxyType)
        assert view.runtime_deltas[0].payload["exit_code"] == 0
        with pytest.raises(TypeError):
            view.runtime_deltas[0].payload["exit_code"] = 1  # type: ignore[index]

    def test_runtime_deltas_order_by_event_seq_then_insertion_order(self) -> None:
        builder = _RunFrameBuilder(run_id="run-frame", task_id="task", task_hash="hash")
        first_none = RuntimeDelta(
            schema_version="0",
            producer="tool_execution",
            delta_kind="exit_nonzero",
            polarity="neutral",
            confidence=1.0,
            run_id="run-frame",
            event_seq=None,
            source_id="tool:none-first",
            payload={"exit_code": 2},
        )
        seq_five = RuntimeDelta(
            schema_version="0",
            producer="tool_execution",
            delta_kind="exit_zero",
            polarity="positive",
            confidence=1.0,
            run_id="run-frame",
            event_seq=5,
            source_id="tool:five",
            payload={"exit_code": 0},
        )
        seq_two = RuntimeDelta(
            schema_version="0",
            producer="tool_execution",
            delta_kind="exit_zero",
            polarity="positive",
            confidence=1.0,
            run_id="run-frame",
            event_seq=2,
            source_id="tool:two",
            payload={"exit_code": 0},
        )
        second_none = RuntimeDelta(
            schema_version="0",
            producer="tool_execution",
            delta_kind="exit_nonzero",
            polarity="neutral",
            confidence=1.0,
            run_id="run-frame",
            event_seq=None,
            source_id="tool:none-second",
            payload={"exit_code": 3},
        )

        for delta in (first_none, seq_five, seq_two, second_none):
            builder.record_delta(delta)

        assert [d.source_id for d in builder.finalize().runtime_deltas] == [
            "tool:two",
            "tool:five",
            "tool:none-first",
            "tool:none-second",
        ]


class TestMultiSourcePrecedence:
    def test_exact_still_wins_over_tool_delta(self) -> None:
        tool_fail = produce_test_parser_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="pytest",
            framework="pytest",
            stdout="=== 2 failed in 0.10s ===",
            stderr="",
            exit_code=1,
        ).deltas[0]

        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(tool_fail,)),
            final_output="ok",
            bench_result={"passed": True, "score": 1.0},
        )

        assert verdict.verdict_source == "exact"
        assert verdict.quality_label == "pass"

    def test_tool_wins_over_formal_when_hierarchy_unchanged(self) -> None:
        tool_fail = produce_test_parser_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="pytest",
            framework="pytest",
            stdout="=== 2 failed in 0.10s ===",
            stderr="",
            exit_code=1,
        ).deltas[0]
        formal_pass = produce_formal_verifier_deltas(
            run_id="run-oracle",
            node_run_id="node-f",
            event_seq=6,
            source_id="z3",
            delta_kind="obligation_proved",
            obligation_id="obl-1",
            obligation_type="postcondition",
            verifier_id="z3",
            solver_status="unsat",
            encoding="prove_no_counterexample",
        ).deltas[0]

        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(tool_fail, formal_pass)),
            final_output="mixed",
        )

        assert verdict.verdict_source == "tool"
        assert verdict.quality_label == "fail"

    def test_tool_partial_wins_over_formal_pass(self) -> None:
        tool_partial = produce_test_parser_deltas(
            run_id="run-oracle",
            node_run_id="node-a",
            event_seq=5,
            source_id="pytest",
            framework="pytest",
            stdout="=== 4 passed, 1 failed in 0.10s ===",
            stderr="",
            exit_code=1,
        ).deltas[0]
        formal_pass = produce_formal_verifier_deltas(
            run_id="run-oracle",
            node_run_id="node-f",
            event_seq=6,
            source_id="z3",
            delta_kind="obligation_proved",
            obligation_id="obl-1",
            obligation_type="postcondition",
            verifier_id="z3",
            solver_status="unsat",
            encoding="prove_no_counterexample",
        ).deltas[0]

        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(tool_partial, formal_pass)),
            final_output="mixed",
        )

        assert verdict.verdict_source == "tool"
        assert verdict.quality_label == "partial"
        assert verdict.score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# cgpro 2026-04-29 R6.1a verify push-back — Gate A regression tests
# ---------------------------------------------------------------------------


class TestFormalCompletenessGate:
    """Producer + oracle defense-in-depth: trainable formal kinds require
    obligation_id + verifier_id + encoding + solver_status, and solver_status
    must match delta_kind direction.
    """

    def _common_kwargs(self) -> dict[str, Any]:
        return {
            "run_id": "run-formal-gate",
            "node_run_id": "node-fa",
            "event_seq": 11,
            "source_id": "z3",
            "obligation_id": "obl-1",
            "obligation_type": "postcondition",
        }

    def test_obligation_proved_missing_verifier_id_is_rejected(self) -> None:
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_proved",
            verifier_id="",
            solver_status="unsat",
            encoding="prove_no_counterexample",
            **self._common_kwargs(),
        )
        assert result.deltas == ()
        assert result.rejected_reason is not None
        assert "verifier_id" in result.rejected_reason

    def test_obligation_proved_missing_encoding_is_rejected(self) -> None:
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_proved",
            verifier_id="z3",
            solver_status="unsat",
            encoding="",
            **self._common_kwargs(),
        )
        assert result.deltas == ()
        assert "encoding" in (result.rejected_reason or "")

    def test_obligation_proved_missing_solver_status_is_rejected(self) -> None:
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_proved",
            verifier_id="z3",
            solver_status="",
            encoding="prove_no_counterexample",
            **self._common_kwargs(),
        )
        assert result.deltas == ()
        assert "solver_status" in (result.rejected_reason or "")

    def test_obligation_proved_with_sat_status_is_rejected(self) -> None:
        # obligation_proved must pair with UNSAT (no counterexample). SAT is
        # the refutation direction; producer rejects the inconsistency.
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_proved",
            verifier_id="z3",
            solver_status="sat",
            encoding="prove_no_counterexample",
            **self._common_kwargs(),
        )
        assert result.deltas == ()
        assert "solver_status" in (result.rejected_reason or "")

    def test_obligation_refuted_with_unsat_status_is_rejected(self) -> None:
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_refuted",
            verifier_id="z3",
            solver_status="unsat",
            encoding="find_counterexample",
            **self._common_kwargs(),
        )
        assert result.deltas == ()
        assert "solver_status" in (result.rejected_reason or "")

    def test_obligation_unknown_does_not_require_solver_status(self) -> None:
        # Non-trainable kinds (unknown / verifier_unavailable) keep the
        # original lenient producer contract — only obligation_id is needed.
        result = produce_formal_verifier_deltas(
            delta_kind="obligation_unknown",
            verifier_id="",
            solver_status="",
            encoding="",
            **self._common_kwargs(),
        )
        assert len(result.deltas) == 1
        delta = result.deltas[0]
        # _formal_oracle treats unknown as Abstain regardless of completeness.
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="undecided",
        )
        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False


class TestToolFatalScopeGate:
    """ToolOracle trains fail on `fatal_failure` ONLY when
    `payload["fatal_scope"] == "claimed_task_output"`. Generic agent-loop
    tool exceptions tagged `incidental_tool_call` must abstain.
    """

    def _fatal(self, fatal_scope: str) -> RuntimeDelta:
        result = produce_tool_deltas(
            run_id="run-tool-gate",
            node_run_id="node-tg",
            event_seq=8,
            source_id="tool:incidental",
            exit_code=-1,
            timed_out=False,
            duration_ms=2.0,
            tool_error_class="RuntimeError",
            fatal_scope=fatal_scope,
        )
        assert result.rejected_reason is None, result.rejected_reason
        return result.deltas[0]

    def test_incidental_tool_fatal_does_not_train(self) -> None:
        delta = self._fatal("incidental_tool_call")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="generic agent loop side effect",
        )
        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False

    def test_unknown_scope_fatal_does_not_train(self) -> None:
        delta = self._fatal("unknown")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="unscoped failure",
        )
        assert verdict.verdict_source == "abstain"

    def test_claimed_task_output_fatal_trains_fail(self) -> None:
        delta = self._fatal("claimed_task_output")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="harness failure invalidates artifact",
        )
        assert verdict.verdict_source == "tool"
        assert verdict.quality_label == "fail"
        assert verdict.trainable is True
        assert verdict.score == 0.0

    def test_invalid_fatal_scope_is_rejected_at_producer(self) -> None:
        result = produce_tool_deltas(
            run_id="run-tool-gate",
            node_run_id="node-tg",
            event_seq=9,
            source_id="tool:bad-scope",
            exit_code=-1,
            timed_out=False,
            duration_ms=2.0,
            tool_error_class="RuntimeError",
            fatal_scope="not-a-real-scope",
        )
        assert result.deltas == ()
        assert "fatal_scope" in (result.rejected_reason or "")


class TestFormalOracleDirectDeltaDefenseInDepth:
    """cgpro 2026-04-29 R6.1a verify round-2 push-back: the producer rejects
    formal deltas missing obligation_id, but a direct RuntimeDelta could
    bypass the producer entirely. _formal_delta_is_complete must re-check
    obligation_id presence in the oracle hot path so synthetic/direct deltas
    cannot become trainable formal verdicts without an obligation reference.
    """

    def _direct_delta(self, kind: str, **payload_overrides: Any) -> RuntimeDelta:
        """Construct a RuntimeDelta directly, bypassing the producer."""
        polarity_map: dict[str, str] = {
            "obligation_proved": "positive",
            "obligation_refuted": "negative",
            "counterexample_found": "negative",
        }
        payload: dict[str, Any] = {
            "verifier_id": "z3",
            "encoding": (
                "prove_no_counterexample"
                if kind == "obligation_proved"
                else "find_counterexample"
            ),
            "solver_status": "unsat" if kind == "obligation_proved" else "sat",
        }
        # Default includes obligation_id; tests override to drop it.
        if "obligation_id" not in payload_overrides:
            payload["obligation_id"] = "obl-direct"
        payload.update(payload_overrides)
        return RuntimeDelta(
            schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
            producer="formal_verifier",
            delta_kind=kind,
            polarity=polarity_map[kind],  # type: ignore[arg-type]
            confidence=1.0,
            run_id="run-direct",
            node_run_id="node-d",
            event_seq=12,
            source_id="z3",
            payload=payload,
        )

    def test_obligation_proved_without_obligation_id_abstains(self) -> None:
        delta = self._direct_delta("obligation_proved", obligation_id="")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="proved without obligation reference",
        )
        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False

    def test_obligation_refuted_without_obligation_id_abstains(self) -> None:
        delta = self._direct_delta("obligation_refuted", obligation_id="")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="refuted without obligation reference",
        )
        assert verdict.verdict_source == "abstain"
        assert verdict.trainable is False

    def test_obligation_proved_with_complete_evidence_trains_pass(self) -> None:
        # Sanity: when all fields are present, the oracle still trains.
        delta = self._direct_delta("obligation_proved")
        verdict = evaluate(
            FakeRunFrameView(runtime_deltas=(delta,)),
            final_output="proved",
        )
        assert verdict.verdict_source == "formal"
        assert verdict.quality_label == "pass"
        assert verdict.trainable is True
