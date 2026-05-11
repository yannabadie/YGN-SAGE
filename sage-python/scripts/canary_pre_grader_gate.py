#!/usr/bin/env python3
"""Pre-grader gate for SWE-bench Pro canary predictions.

Block ``canary-stage-timing-budget`` (cgpro DESIGN 2026-05-11, conv
``cgpro_ygn_sage_global_analysis_20260510``) slice 2.

Validates that every canary prediction with ``patch == ""`` carries an
explicit reason code from the allowlist
``sage.bench.event_ledger.EMPTY_PATCH_REASON_CODES``. Without this gate
an empty-patch prediction can reach the Modal grader and produce a
``resolved=0`` row for the wrong reason — silent infra fail rather
than an instrumented "no patch was extracted because X" classification.

Reason-code resolution order, per instance:

1. ``patch != ""`` → pass with ``verdict="pass:non_empty_patch"``.
2. ``patch == ""`` and a matching ``predictions.jsonl`` row exposes a
   pre-recorded ``_reason_code`` member of ``EMPTY_PATCH_REASON_CODES``
   → pass with that code. (Pre-recording is a slice-4 follow-up; the
   gate accepts it once wired.)
3. ``patch == ""`` and ``_timeout=True`` in the jsonl row → load the
   per-task events file and call
   ``sage.bench.event_ledger.categorize_timeout`` to derive the code.
4. ``patch == ""`` and ``_timeout=False`` → call
   ``classify_non_timeout_empty_patch`` with ``budget_exhausted`` (only
   when explicitly recorded) and ``_diff_verifier_outcome`` from the
   jsonl row.
5. None of the above → fail with ``verdict="fail:no_allowed_reason_code"``.

CLI:

    python sage-python/scripts/canary_pre_grader_gate.py \\
        --predictions docs/benchmarks/<run>/predictions.json \\
        --events-dir docs/benchmarks/<run>/per_task

Exit 0 on overall PASS, 1 on FAIL. Writes ``gate_result.json`` alongside
``--predictions`` unless ``--output`` overrides.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sage.bench.event_ledger import (
    EMPTY_PATCH_REASON_CODES,
    categorize_timeout,
    classify_non_timeout_empty_patch,
)

log = logging.getLogger("canary_pre_grader_gate")


_RUNTIME_EVENT_TYPES = {
    "cli_progress": "progress_events",
    "model_assigned": "model_assigned_events",
    "node_started": "node_started_events",
    "routing_decision": "routing_decision_events",
}


def _load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load the grader-shaped predictions.json (list of dicts)."""
    raw = predictions_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(
            f"predictions file at {predictions_path} is not a JSON list "
            f"(got {type(data).__name__}); the grader expects a list"
        )
    return data


def _load_predictions_jsonl(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load the annotated predictions.jsonl and index by ``instance_id``.

    Missing file returns an empty dict — the gate degrades gracefully
    to events-only reason derivation.
    """
    if not jsonl_path.exists():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        iid = row.get("instance_id")
        if iid:
            indexed[iid] = row
    return indexed


def _load_events(events_path: Path) -> list[dict[str, Any]]:
    """Load a per-task RuntimeEventLog jsonl. Missing file returns []."""
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # The runner appends a non-schema runner_timeout sentinel
            # at the very end on hard cutoffs. Skip parse errors so a
            # malformed sentinel never invalidates the whole stream.
            log.warning("Skipping unparseable event line in %s", events_path)
            continue
    return events


def _bucket_events(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group events by the four categorize_timeout input keys."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "progress_events": [],
        "model_assigned_events": [],
        "node_started_events": [],
        "routing_decision_events": [],
    }
    for ev in events:
        key = _RUNTIME_EVENT_TYPES.get(ev.get("event_type") or "")
        if key:
            buckets[key].append(ev)
    return buckets


def _resolve_elapsed_total_ms(
    annotation: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> float:
    """Best-effort ``elapsed_total_ms`` for categorize_timeout.

    Prefers ``_latency_ms`` from the annotated jsonl row, falls back to
    the max ``cli_progress.payload.elapsed_ms`` seen, and finally 0 if
    no signal is available (categorize_timeout still produces a code).
    """
    if annotation and isinstance(annotation.get("_latency_ms"), (int, float)):
        return float(annotation["_latency_ms"])
    fallback = 0
    for ev in events:
        if ev.get("event_type") != "cli_progress":
            continue
        payload = ev.get("payload") or {}
        elapsed = payload.get("elapsed_ms")
        if isinstance(elapsed, (int, float)) and elapsed > fallback:
            fallback = int(elapsed)
    return float(fallback)


def classify_empty_patch(
    *,
    annotation: dict[str, Any] | None,
    events: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Return ``(reason_code, evidence)`` for an empty-patch prediction.

    ``reason_code`` is one of ``EMPTY_PATCH_REASON_CODES`` or ``None`` if
    no recognized signal can be derived from the available inputs.
    ``evidence`` documents which path produced the code (or which path
    failed) so the gate result is self-describing.
    """
    # 1. Pre-recorded reason code (slice 4+ follow-up).
    if annotation:
        recorded = annotation.get("_reason_code")
        if isinstance(recorded, str) and recorded in EMPTY_PATCH_REASON_CODES:
            return recorded, {
                "source": "pre_recorded",
                "field": "_reason_code",
            }

    is_timeout = bool(annotation and annotation.get("_timeout"))
    diff_verifier_outcome = (
        annotation.get("_diff_verifier_outcome") if annotation else None
    )

    # 2. Timeout: derive via categorize_timeout from events.
    if is_timeout:
        buckets = _bucket_events(events)
        elapsed_total_ms = _resolve_elapsed_total_ms(annotation, events)
        timeout_result = categorize_timeout(
            progress_events=buckets["progress_events"],
            model_assigned_events=buckets["model_assigned_events"],
            node_started_events=buckets["node_started_events"],
            routing_decision_events=buckets["routing_decision_events"],
            elapsed_total_ms=elapsed_total_ms,
        )
        reason = timeout_result.get("reason_code")
        if isinstance(reason, str) and reason in EMPTY_PATCH_REASON_CODES:
            return reason, {
                "source": "categorize_timeout",
                "elapsed_total_ms": elapsed_total_ms,
                "last_stage": timeout_result.get("last_stage"),
                "provider_attempted": timeout_result.get("provider_attempted"),
            }
        return None, {
            "source": "categorize_timeout",
            "rejected_reason": reason,
            "elapsed_total_ms": elapsed_total_ms,
        }

    # 3. Non-timeout: use classifier for budget / verifier / fallback.
    budget_exhausted = bool(
        annotation and annotation.get("_budget_exhausted")
    )
    reason = classify_non_timeout_empty_patch(
        budget_exhausted=budget_exhausted,
        diff_verifier_outcome=(
            diff_verifier_outcome
            if isinstance(diff_verifier_outcome, str)
            else None
        ),
    )
    return reason, {
        "source": "classify_non_timeout_empty_patch",
        "budget_exhausted": budget_exhausted,
        "diff_verifier_outcome": diff_verifier_outcome,
    }


def _events_path_for(events_dir: Path, instance_id: str) -> Path:
    return events_dir / f"{instance_id}.events.jsonl"


def run_gate(
    *,
    predictions_path: Path,
    events_dir: Path,
    predictions_jsonl_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the gate. Returns the structured result dict."""
    predictions = _load_predictions(predictions_path)

    jsonl_path = predictions_jsonl_path or predictions_path.with_suffix(".jsonl")
    annotations = _load_predictions_jsonl(jsonl_path)

    per_instance: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0

    for entry in predictions:
        iid = entry.get("instance_id") or "<missing>"
        patch = entry.get("patch")

        if isinstance(patch, str) and patch != "":
            n_pass += 1
            per_instance.append(
                {
                    "instance_id": iid,
                    "verdict": "pass:non_empty_patch",
                    "patch_present": True,
                    "reason_code": None,
                    "evidence": {"patch_chars": len(patch)},
                }
            )
            continue

        annotation = annotations.get(iid)
        events = _load_events(_events_path_for(events_dir, iid))
        reason, evidence = classify_empty_patch(
            annotation=annotation,
            events=events,
        )

        if reason is not None:
            n_pass += 1
            verdict = f"pass:{reason}"
        else:
            n_fail += 1
            verdict = "fail:no_allowed_reason_code"

        per_instance.append(
            {
                "instance_id": iid,
                "verdict": verdict,
                "patch_present": False,
                "reason_code": reason,
                "evidence": evidence,
            }
        )

    return {
        "schema_version": "canary_pre_grader_gate_v1",
        "gate_status": "PASS" if n_fail == 0 else "FAIL",
        "predictions_path": str(predictions_path),
        "predictions_jsonl_path": str(jsonl_path) if jsonl_path.exists() else None,
        "events_dir": str(events_dir),
        "n_predictions": len(predictions),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "allowed_reason_codes": sorted(EMPTY_PATCH_REASON_CODES),
        "per_instance": per_instance,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to predictions.json (canonical grader input).",
    )
    parser.add_argument(
        "--events-dir",
        required=True,
        type=Path,
        help="Directory containing per-task RuntimeEventLog .events.jsonl files.",
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help=(
            "Path to predictions.jsonl with annotations. Defaults to the "
            "sibling of --predictions; missing file is tolerated."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write gate_result.json. Defaults to "
            "<predictions.dir>/gate_result.json."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log per-instance verdicts to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    result = run_gate(
        predictions_path=args.predictions,
        events_dir=args.events_dir,
        predictions_jsonl_path=args.predictions_jsonl,
    )

    output_path = args.output or (args.predictions.parent / "gate_result.json")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.verbose:
        for entry in result["per_instance"]:
            log.info(
                "%-7s %s reason=%s",
                entry["verdict"].split(":", 1)[0],
                entry["instance_id"],
                entry["reason_code"],
            )

    status = result["gate_status"]
    print(
        f"{status} {result['n_pass']}/{result['n_predictions']} "
        f"predictions classified; result -> {output_path}",
        file=sys.stderr,
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
