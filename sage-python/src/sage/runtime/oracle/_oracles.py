from __future__ import annotations

import hashlib
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
    """Bench harness pass/fail from normalized structured bench_result keys.

    cgpro 2026-04-29 cycle-7 flip review push-back: ``bench_result["reason"]``
    is allowed to be a raw harness fragment (unittest traceback, stderr tail,
    timeout sentinel, …). Earlier code appended that raw string directly to
    ``reason_codes``, which leaked the harness body into ``oracle_verdict``
    payloads and ``run_frame_summary.oracle_verdict`` mirrors. We now:

    - Stamp ``reason_codes`` with structured tags only (``exact_test_pass`` /
      ``exact_test_fail`` plus, when an opaque reason is present, the
      sentinel ``exact_harness_detail_present``).
    - Carry the SHA-256 of the raw reason via ``EvidenceRef.evidence_hash``
      so audit trails can correlate verdicts to harness logs without the
      payload itself appearing in the JSONL.

    Optional pre-classified ``bench_result["reason_code"]`` (e.g.
    ``"bcb_unittest_error"``, ``"bcb_timeout"``) is appended verbatim — the
    bench seam owns this field's vocabulary, no raw text.
    """
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
    reason_codes: list[str] = [f"exact_test_{label}"]

    pre_classified_code = bench_result.get("reason_code")
    if isinstance(pre_classified_code, str) and pre_classified_code:
        reason_codes.append(pre_classified_code)

    raw_reason = bench_result.get("reason")
    reason_hash: str | None = None
    if isinstance(raw_reason, str) and raw_reason:
        reason_hash = hashlib.sha256(raw_reason.encode("utf-8")).hexdigest()
        if "exact_harness_detail_present" not in reason_codes:
            reason_codes.append("exact_harness_detail_present")

    pre_supplied_hash = bench_result.get("reason_sha256")
    if isinstance(pre_supplied_hash, str) and pre_supplied_hash:
        reason_hash = pre_supplied_hash

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
                evidence_hash=reason_hash,
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
        # cgpro 2026-04-29 R6.1a verify push-back: only fatal failures whose
        # `fatal_scope == "claimed_task_output"` deterministically invalidate
        # the artifact under evaluation. Generic agent-loop tool exceptions
        # (`fatal_scope == "incidental_tool_call"`) and unscoped failures
        # must NOT become trainable fail; the hierarchy falls through to
        # the next oracle (Formal / Spec / Abstain).
        scoped_fatals = [
            delta
            for delta in fatal_deltas
            if str(getattr(delta, "payload", {}).get("fatal_scope", "")).strip().lower()
            == "claimed_task_output"
        ]
        if scoped_fatals:
            return OracleVerdict(
                trainable=True,
                verdict_source="tool",
                quality_label="fail",
                score=0.0,
                confidence=1.0,
                reason_codes=("tool_fatal_failure",),
                evidence=tuple(_evidence_ref(delta) for delta in scoped_fatals),
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

    # Defense-in-depth (cgpro 2026-04-29 R6.1a verify push-back): even though
    # the formal producer rejects incomplete trainable deltas, re-check that
    # any delta we are about to convert into a trainable verdict carries
    # complete obligation semantics (verifier_id + encoding + solver_status
    # consistent with delta_kind). Incomplete deltas force Abstain.
    if not all(_formal_delta_is_complete(delta) for delta in trainable_deltas):
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


_FORMAL_KIND_TO_SOLVER_STATUS: dict[str, frozenset[str]] = {
    "obligation_proved": frozenset({"unsat"}),
    "obligation_refuted": frozenset({"sat"}),
    "counterexample_found": frozenset({"sat"}),
}


def _formal_delta_is_complete(delta: Any) -> bool:
    """Defense-in-depth re-check for trainable formal deltas.

    A trainable formal delta (obligation_proved / obligation_refuted /
    counterexample_found) MUST carry complete obligation semantics:
    obligation_id + verifier_id + encoding + solver_status, and
    solver_status must match the kind's expected direction.
    obligation_unknown / verifier_unavailable do not produce trainable
    verdicts and are skipped here (the caller handles them as Abstain
    triggers). cgpro 2026-04-29 R6.1a verify round-2 push-back: the
    obligation_id check is here AS WELL AS in the producer so a directly-
    constructed RuntimeDelta cannot bypass the producer to become a
    trainable formal verdict without an obligation reference.
    """
    kind = getattr(delta, "delta_kind", "")
    if kind not in _FORMAL_KIND_TO_SOLVER_STATUS:
        return True
    payload = getattr(delta, "payload", {}) or {}
    obligation_id = str(payload.get("obligation_id", "")).strip()
    verifier_id = str(payload.get("verifier_id", "")).strip()
    encoding = str(payload.get("encoding", "")).strip()
    solver_status = str(payload.get("solver_status", "")).strip().lower()
    if not obligation_id or not verifier_id or not encoding or not solver_status:
        return False
    return solver_status in _FORMAL_KIND_TO_SOLVER_STATUS[kind]


def _spec_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Spec oracle: trainable=True only with structured claim-dependency evidence.

    cgpro 2026-04-29 R6.1a verify push-back: the lexical/substring scan
    (`_INVALIDATION_MARKERS` + `assumption_id in output_text`) is removed.
    For v1, we abstain unless a future cycle (cycle 7+) wires a structured
    claim-dependency channel proving the final output reasserts the
    invalidated assumption. StateFrame.invalidated_assumptions alone OR
    formal_verifier/assumption_invalidated alone are insufficient — we need
    proof that the output structurally depends on the invalidated assumption.
    Until that channel exists, spec oracle abstains and the hierarchy falls
    through to LLMJudge stub → Abstain.
    """
    del view, final_output, bench_result, config
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
