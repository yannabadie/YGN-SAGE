from __future__ import annotations

from typing import Any, Mapping

from sage.runtime.oracle.config import OracleConfig
from sage.runtime.oracle.verdict import EvidenceRef, OracleVerdict, QualityLabel

_TRUSTED_TEST_PARSERS: frozenset[str] = frozenset(
    {"pytest_summary_v0", "junit_xml_v0", "unittest_summary_v0"}
)


def _exact_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Bench harness pass/fail from normalized structured bench_result keys."""
    del final_output, config
    if bench_result is None:
        return None
    if "passed" not in bench_result:
        return None

    passed = bool(bench_result["passed"])
    try:
        score = float(bench_result.get("score", 1.0 if passed else 0.0))
    except (TypeError, ValueError):
        return None
    score = max(0.0, min(1.0, score))
    label: QualityLabel = "pass" if passed else "fail"
    reason_codes = [f"exact_test_{label}"]
    reason = bench_result.get("reason")
    if isinstance(reason, str) and reason:
        reason_codes.append(reason)

    return OracleVerdict(
        trainable=True,
        verdict_source="exact",
        quality_label=label,
        score=score,
        confidence=1.0,
        reason_codes=tuple(reason_codes),
        evidence=(
            EvidenceRef(
                run_id=view.run_id,
                node_run_id=None,
                event_seq=getattr(view, "final_result_seq", None),
                output_sha256=bench_result.get("output_sha256"),
                tool_call_id=bench_result.get("tool_call_id"),
                verifier_id=bench_result.get("verifier_id", "bench_harness"),
            ),
        ),
    )


def _tool_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Consume deterministic test_parser/tool_execution RuntimeDelta facts."""
    del final_output, bench_result, config
    deltas = tuple(getattr(view, "runtime_deltas", ()) or ())
    parser_deltas = [
        delta
        for delta in deltas
        if getattr(delta, "producer", "") == "test_parser"
        and delta.delta_kind in {"tests_passed", "tests_failed", "tests_partial"}
        and delta.payload.get("parser_id") in _TRUSTED_TEST_PARSERS
    ]
    tool_deltas = [
        delta for delta in deltas if getattr(delta, "producer", "") == "tool_execution"
    ]
    if not parser_deltas and not tool_deltas:
        return None
    if any(delta.delta_kind == "unavailable" for delta in tool_deltas):
        return None
    if _has_contradictory_parser_facts(parser_deltas):
        return None
    fatal_deltas = [delta for delta in tool_deltas if delta.delta_kind == "fatal_failure"]

    if parser_deltas:
        if fatal_deltas and all(delta.delta_kind == "tests_passed" for delta in parser_deltas):
            return None
        negative = [
            delta
            for delta in parser_deltas
            if delta.delta_kind in {"tests_failed", "tests_partial"}
        ]
        if negative:
            passed = sum(int(delta.payload.get("passed_count", 0)) for delta in negative)
            failed = sum(int(delta.payload.get("failed_count", 0)) for delta in negative)
            errors = sum(int(delta.payload.get("error_count", 0)) for delta in negative)
            total = passed + failed + errors
            is_partial = any(delta.delta_kind == "tests_partial" for delta in negative)
            if is_partial and total > 0:
                score = passed / total
                quality_label: QualityLabel = "partial"
                reason_codes = ("tool_tests_partial",)
            else:
                score = 0.0
                quality_label = "fail"
                reason_codes = ("tool_tests_failed",)
            return OracleVerdict(
                trainable=True,
                verdict_source="tool",
                quality_label=quality_label,
                score=score,
                confidence=1.0,
                reason_codes=reason_codes,
                evidence=tuple(_evidence_ref(delta) for delta in negative),
            )
        if all(delta.delta_kind == "tests_passed" for delta in parser_deltas):
            return OracleVerdict(
                trainable=True,
                verdict_source="tool",
                quality_label="pass",
                score=1.0,
                confidence=1.0,
                reason_codes=("tool_tests_passed",),
                evidence=tuple(_evidence_ref(delta) for delta in parser_deltas),
            )

    if fatal_deltas:
        return OracleVerdict(
            trainable=True,
            verdict_source="tool",
            quality_label="fail",
            score=0.0,
            confidence=1.0,
            reason_codes=("tool_fatal_failure",),
            evidence=tuple(_evidence_ref(delta) for delta in fatal_deltas),
        )

    return None


def _formal_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Consume obligation-semantic formal_verifier RuntimeDelta facts."""
    del final_output, bench_result, config
    formal_deltas = [
        delta
        for delta in tuple(getattr(view, "runtime_deltas", ()) or ())
        if getattr(delta, "producer", "") == "formal_verifier"
    ]
    if not formal_deltas:
        return None
    if _has_contradictory_formal_facts(formal_deltas):
        return None

    trainable_deltas = [
        delta
        for delta in formal_deltas
        if delta.delta_kind
        in {
            "obligation_proved",
            "obligation_refuted",
            "counterexample_found",
            "obligation_unknown",
            "verifier_unavailable",
        }
    ]
    if not trainable_deltas:
        return None

    negatives = [
        delta
        for delta in trainable_deltas
        if delta.delta_kind in {"obligation_refuted", "counterexample_found"}
    ]
    if negatives:
        return OracleVerdict(
            trainable=True,
            verdict_source="formal",
            quality_label="fail",
            score=0.0,
            confidence=1.0,
            reason_codes=("formal_obligation_refuted",),
            evidence=tuple(_evidence_ref(delta) for delta in negatives),
        )

    if any(
        delta.delta_kind in {"obligation_unknown", "verifier_unavailable"}
        for delta in trainable_deltas
    ):
        return None

    positives = [
        delta for delta in trainable_deltas if delta.delta_kind == "obligation_proved"
    ]
    if positives and len(positives) == len(trainable_deltas):
        return OracleVerdict(
            trainable=True,
            verdict_source="formal",
            quality_label="pass",
            score=1.0,
            confidence=1.0,
            reason_codes=("formal_obligations_proved",),
            evidence=tuple(_evidence_ref(delta) for delta in positives),
        )

    return None


# Negation/invalidation phrases that, when co-occurring with an invalidated
# assumption_id, indicate the output is acknowledging the invalidation
# (mention-as-invalid) rather than reasserting the assumption as true.
# In that case _spec_oracle MUST NOT emit a trainable=True negative verdict
# (cgpro 2026-04-29 R9 verify push-back: mention ≠ contradiction).
_INVALIDATION_MARKERS: tuple[str, ...] = (
    "was wrong",
    "is wrong",
    "was incorrect",
    "is incorrect",
    "is false",
    "was false",
    "no longer holds",
    "no longer valid",
    "no longer true",
    "was invalidated",
    "was retracted",
    "doesn't hold",
    "does not hold",
    "doesn't apply",
    "does not apply",
    "doesn't work",
    "does not work",
    "ruled out",
    "found to be wrong",
    "found to be false",
    "found to be incorrect",
    "found to be invalid",
    "rejected",
    "discarded",
    "is invalid",
    "was invalid",
)


def _spec_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Detect final-output reassertions of invalidated StateFrame assumptions.

    v0 guard: if the assumption_id appears in the output AND the output also
    contains an invalidation/negation marker phrase, treat the appearance as
    a "mention as invalidated" (NOT a contradiction) and return None so the
    hierarchy falls through to Abstain. Only when the assumption_id appears
    WITHOUT any invalidation marker do we emit trainable=True negative.

    This prevents the lexical-fallback failure mode where merely discussing
    that an assumption WAS invalidated would train a negative reward.
    """
    del bench_result, config
    if not getattr(view, "state_frames", None):
        return None
    output_text = final_output or ""
    output_lower = output_text.lower()
    has_invalidation_marker = any(
        marker in output_lower for marker in _INVALIDATION_MARKERS
    )
    formal_assumption_deltas = _formal_assumption_invalidations(view)
    for node_id, frame in view.state_frames.items():
        if not frame.invalidated_assumptions:
            continue
        for assumption_id in frame.invalidated_assumptions:
            if not assumption_id or assumption_id not in output_text:
                continue
            if has_invalidation_marker:
                # Output discusses the assumption in the context of its
                # invalidation; this is acknowledgment, not contradiction.
                continue
            corroborating = formal_assumption_deltas.get(assumption_id)
            evidence = [
                EvidenceRef(
                    run_id=view.run_id,
                    node_run_id=str(node_id),
                    event_seq=None,
                    output_sha256=None,
                    verifier_id="statecore_spec",
                )
            ]
            if corroborating is not None:
                evidence.append(_evidence_ref(corroborating))
            return OracleVerdict(
                trainable=True,
                verdict_source="spec",
                quality_label="fail",
                score=0.0,
                confidence=0.9 if corroborating is not None else 0.85,
                reason_codes=(
                    "state_contradiction",
                    f"invalidated_assumption_in_output:{assumption_id}",
                ),
                evidence=tuple(evidence),
            )
    return None


def _llm_judge_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """LLMJudge stub. v0 never returns trainable evidence."""
    del view, final_output, bench_result
    if not config.enable_llm_judge:
        return None
    return None


def _evidence_ref(delta: Any) -> EvidenceRef:
    return EvidenceRef(
        run_id=delta.run_id,
        node_run_id=delta.node_run_id,
        event_seq=delta.event_seq,
        verifier_id=f"{delta.producer}:{delta.source_id}",
        evidence_hash=delta.evidence_hash,
    )


def _has_contradictory_parser_facts(deltas: list[Any]) -> bool:
    by_suite: dict[tuple[str | None, str, str], set[str]] = {}
    for delta in deltas:
        suite_key = (
            delta.node_run_id,
            str(delta.payload.get("parser_id", "")),
            str(delta.payload.get("suite_id", "")),
        )
        by_suite.setdefault(suite_key, set()).add(delta.delta_kind)
    for kinds in by_suite.values():
        if "tests_passed" in kinds and (
            "tests_failed" in kinds or "tests_partial" in kinds
        ):
            return True
        if "tests_failed" in kinds and "tests_partial" in kinds:
            return True
    return False


def _has_contradictory_formal_facts(deltas: list[Any]) -> bool:
    by_obligation: dict[str, set[str]] = {}
    for delta in deltas:
        obligation_id = str(delta.payload.get("obligation_id", ""))
        if not obligation_id:
            continue
        by_obligation.setdefault(obligation_id, set()).add(delta.delta_kind)
    positive = {"obligation_proved"}
    negative = {"obligation_refuted", "counterexample_found"}
    for kinds in by_obligation.values():
        if kinds & positive and kinds & negative:
            return True
    return False


def _formal_assumption_invalidations(view: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for delta in tuple(getattr(view, "runtime_deltas", ()) or ()):
        if (
            getattr(delta, "producer", "") == "formal_verifier"
            and delta.delta_kind == "assumption_invalidated"
        ):
            assumption_id = str(delta.payload.get("obligation_id", "") or "")
            if assumption_id:
                out.setdefault(assumption_id, delta)
    return out
