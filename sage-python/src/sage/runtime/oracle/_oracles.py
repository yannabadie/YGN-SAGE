from __future__ import annotations

from typing import Any, Mapping

from sage.runtime.oracle.config import OracleConfig
from sage.runtime.oracle.verdict import EvidenceRef, OracleVerdict, QualityLabel


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
    """Structured tool evidence only. v0 waits for structured deltas."""
    del view, final_output, bench_result, config
    return None


def _formal_oracle(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None,
    config: OracleConfig,
) -> OracleVerdict | None:
    """Z3/SMT/OxiZ formal verdict placeholder for v0."""
    del view, final_output, bench_result, config
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
            return OracleVerdict(
                trainable=True,
                verdict_source="spec",
                quality_label="fail",
                score=0.0,
                confidence=0.85,
                reason_codes=(
                    "state_contradiction",
                    f"invalidated_assumption_in_output:{assumption_id}",
                ),
                evidence=(
                    EvidenceRef(
                        run_id=view.run_id,
                        node_run_id=str(node_id),
                        event_seq=None,
                        output_sha256=None,
                        verifier_id="statecore_spec",
                    ),
                ),
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
