from __future__ import annotations

from typing import Any, Mapping

from sage.runtime.evidence.delta import (
    RUNTIME_DELTA_SCHEMA_VERSION,
    DeltaPolarity,
    DeltaProducerResult,
    RuntimeDelta,
    _DELTA_KIND_TABLE,
    _POLARITY_RULES,
)
from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import PAYLOAD_ALLOWED_KEYS


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["diff_verifier"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["diff_verifier"]
POLARITY_RULES = _POLARITY_RULES["diff_verifier"]


def produce_diff_verifier_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    verify_result: Mapping[str, Any] | Any,
    repair_result: Mapping[str, Any] | None = None,
) -> DeltaProducerResult:
    """Produce diff_verifier facts, never raw patch/diff content."""
    try:
        patch_hash = _get(verify_result, "patch_hash")
        mismatch_kinds = _mismatch_kinds(verify_result)
        mismatch_count = sum(mismatch_kinds.values())
        outcome = str(_get(verify_result, "outcome") or "")
        if not mismatch_kinds:
            mismatch_kinds = _reason_kinds(verify_result)

        patch_apply_ok = outcome == "clean"
        if outcome == "clean":
            delta_kind = "patch_applied"
            polarity: DeltaPolarity = "positive"
        elif outcome in {"malformed_hunk_header", "hunk_body_count_mismatch"}:
            delta_kind = "hunk_header_mismatch"
            polarity = "negative"
            patch_apply_ok = False
        elif mismatch_count > 0 or outcome in {
            "content_mismatch",
            "file_missing",
            "fuzzy_below_threshold",
        }:
            delta_kind = "context_mismatch"
            polarity = "negative"
            patch_apply_ok = False
        else:
            delta_kind = "patch_failed"
            polarity = "neutral"
            patch_apply_ok = False

        payload: dict[str, Any] = {
            "patch_apply_ok": patch_apply_ok,
            "mismatch_count": mismatch_count,
            "mismatch_kinds": mismatch_kinds,
        }
        if patch_hash:
            payload["patch_hash"] = str(patch_hash)

        deltas = [
            RuntimeDelta(
                schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                producer="diff_verifier",
                delta_kind=delta_kind,
                polarity=polarity,
                confidence=1.0,
                run_id=run_id,
                node_run_id=node_run_id,
                event_seq=event_seq,
                source_id=source_id,
                payload=payload,
            )
        ]

        if repair_result is not None:
            stage = str(repair_result.get("repair_stage", "") or "")
            repair_hash = str(repair_result.get("patch_hash", patch_hash or "") or "")
            if stage:
                repair_kind = "repair_rejected" if stage == "failed" else "repair_accepted"
                repair_payload: dict[str, Any] = {"repair_stage": stage}
                if repair_hash:
                    repair_payload["patch_hash"] = repair_hash
                deltas.append(
                    RuntimeDelta(
                        schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                        producer="diff_verifier",
                        delta_kind=repair_kind,
                        polarity="negative" if repair_kind == "repair_rejected" else "positive",
                        confidence=1.0,
                        run_id=run_id,
                        node_run_id=node_run_id,
                        event_seq=event_seq,
                        source_id=source_id,
                        payload=repair_payload,
                    )
                )
        return DeltaProducerResult(deltas=tuple(deltas))
    except (EvidenceError, TypeError, ValueError) as exc:
        return DeltaProducerResult(deltas=(), rejected_reason=str(exc))


def _get(value: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _mismatch_kinds(value: Mapping[str, Any] | Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for mismatch in _get(value, "mismatches") or ():
        kind = _get(mismatch, "kind")
        if not kind:
            continue
        out[str(kind)] = out.get(str(kind), 0) + 1
    return out


def _reason_kinds(value: Mapping[str, Any] | Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for reason in _get(value, "reasons") or ():
        reason_key = str(reason)
        if reason_key == "clean":
            continue
        out[reason_key] = out.get(reason_key, 0) + 1
    return out
