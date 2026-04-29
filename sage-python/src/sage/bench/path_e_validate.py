"""Path E step 3 post-bench validator.

cgpro 2026-04-29 R6.1a verify Path E recommendation: prove the live
``verdict_source="exact", trainable=True`` contract on a synchronous-eval
bench (BigCodeBench Hard) WITHOUT shipping a calibrated Pass@1 claim.

This module is the canonical validator for the artifacts produced by::

    SAGE_ORACLE=1 SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 \
    SAGE_TRACE_JSONL_DIR=<dir> \
    python -m sage.bench --type bigcodebench --subset hard --split instruct \
                         --limit N --output <report.json>

It reads the RuntimeEventLog JSONL traces, the SAGE bench report, and the
auto-emitted predictions JSONL; cross-checks the seam Exact verdict against
the bench's final ``passed`` field per task (escalation may diverge); emits
a BCB-canonical predictions JSONL stripping the ``_trace`` field; computes
SHA-256 of every artifact; and writes a Markdown validation report.

Honest framing locks (cgpro 2026-04-29 lock + AUDIT2 2026-04-24):

- This is **NOT** a BigCodeBench leaderboard submission. The leaderboard
  reports calibrated Pass@1 with greedy decoding through the official
  ``bigcodebench.evaluate`` harness (or its e2b/gradio backends). On
  Windows that path raises ``AttributeError: os.killpg`` from
  ``bigcodebench.eval.utils.safe_environment`` and silently coerces every
  task to ``timeout``. The seam evaluator therefore documents a fall-back
  to ``BigCodeBenchBench._evaluate_solution_with_stderr`` (matplotlib-
  headless subprocess; deterministic per (solution, test_code)) and tags
  the resulting bench_result with ``verifier_id="bcb_internal_subprocess_fallback"``.
- This run is a **seam validation smoke**, not a value/regression benchmark.
- All artifacts are committed to the repo; SHA-256 manifest enables
  third-party reproduction of the validation criteria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class TaskValidation:
    task_id: str | None
    run_id: str
    seam_verdict_source: str | None
    seam_quality_label: str | None
    seam_score: float | None
    seam_trainable: bool | None
    bench_passed: bool | None
    runtime_delta_count: int | None
    final_status: str | None
    event_order_ok: bool


def _validate_run(events: list[dict[str, Any]]) -> TaskValidation:
    """Reduce a single run's events into a validation record."""
    by_type: dict[str, dict[str, Any]] = {}
    seqs: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "")
        if et not in by_type:
            by_type[et] = e
            seqs[et] = e.get("seq", -1)
    ov = by_type.get("oracle_verdict", {}).get("payload", {})
    rfs = by_type.get("run_frame_summary", {}).get("payload", {})
    fr = by_type.get("final_result", {}).get("payload", {})
    ts = by_type.get("task_started", {}).get("payload", {})

    event_order_ok = (
        "final_result" in seqs
        and "oracle_verdict" in seqs
        and "run_frame_summary" in seqs
        and seqs["final_result"] < seqs["oracle_verdict"] < seqs["run_frame_summary"]
    )
    return TaskValidation(
        task_id=ts.get("task_hash") or events[0].get("run_id", ""),
        run_id=events[0].get("run_id", ""),
        seam_verdict_source=ov.get("verdict_source"),
        seam_quality_label=ov.get("quality_label"),
        seam_score=ov.get("score"),
        seam_trainable=ov.get("trainable"),
        bench_passed=None,
        runtime_delta_count=rfs.get("runtime_delta_count"),
        final_status=fr.get("status") or rfs.get("status"),
        event_order_ok=event_order_ok,
    )


_RAW_LEAK_PHRASES: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "AssertionError",
    "  File \"",
    "Error: ",
    "FAIL: test_",
    "ERROR: test_",
)


def _scan_payload_for_raw_leaks(payload: Any, _path: str = "") -> list[str]:
    """Walk the payload tree and return paths where forbidden raw-output keys
    carry strings longer than 64 chars OR where any string anywhere contains
    a known raw-output phrase (e.g. unittest traceback fragment, harness
    error tail). The phrase scan is the cgpro 2026-04-29 cycle-7 flip review
    add-on: ``oracle_verdict.reason_codes`` is a tuple of strings, not a
    forbidden-key field, but a raw harness reason can still leak through
    that channel as a tuple element. Scan ALL strings, not just keys.
    """
    forbidden = {
        "stdout",
        "stderr",
        "raw_stdout",
        "raw_stderr",
        "output",
        "raw_output",
        "content",
        "raw_content",
        "patch",
        "raw_patch",
        "diff",
        "raw_diff",
        "final_answer",
        "message",
        "traceback",
        "reason",
    }
    leaks: list[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            sub_path = f"{_path}.{k}" if _path else k
            if k in forbidden and isinstance(v, str) and len(v) > 64:
                leaks.append(f"{sub_path}=str{len(v)}")
            leaks.extend(_scan_payload_for_raw_leaks(v, sub_path))
    elif isinstance(payload, list) or isinstance(payload, tuple):
        for i, item in enumerate(payload):
            leaks.extend(_scan_payload_for_raw_leaks(item, f"{_path}[{i}]"))
    elif isinstance(payload, str):
        for phrase in _RAW_LEAK_PHRASES:
            if phrase in payload:
                leaks.append(f"{_path}: raw-phrase {phrase!r} in str{len(payload)}")
                break
    return leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", required=True, type=Path)
    parser.add_argument("--bench-report", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out-canonical-predictions", required=True, type=Path)
    parser.add_argument("--out-manifest", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    args = parser.parse_args()

    # Load all artifacts.
    bench_report = json.loads(args.bench_report.read_text(encoding="utf-8"))
    predictions = _load_jsonl(args.predictions)
    bench_results_by_task = {r["task_id"]: r for r in bench_report.get("results", [])}

    jsonl_files = sorted(args.jsonl_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no JSONL files in {args.jsonl_dir}", file=sys.stderr)
        return 1

    # Validate each run.
    runs: list[TaskValidation] = []
    raw_leak_findings: list[str] = []
    for jf in jsonl_files:
        events = _load_jsonl(jf)
        if not events:
            continue
        v = _validate_run(events)
        # Map the run's task by predictions order if event payload doesn't
        # carry task_id (RuntimeEventLog redacts the prompt; we walk the
        # node_started events to find any model_id hints, otherwise keep
        # run-level identity only).
        runs.append(v)
        # Raw output safety scan across the full event document — not only
        # ``payload``. cgpro 2026-04-29 cycle-7 flip review: oracle_verdict
        # reason_codes / run_frame_summary nested oracle_verdict / EvidenceRef
        # fields can all carry strings outside the conventional ``payload``
        # subtree.
        for e in events:
            leaks = _scan_payload_for_raw_leaks(e)
            for leak in leaks:
                raw_leak_findings.append(
                    f"{e.get('event_type','?')}@seq{e.get('seq','?')}:{leak}"
                )

    # Order JSONL files chronologically and zip with predictions order;
    # this is best-effort because the bench writes predictions in task-id
    # order while RuntimeEventLog files are timestamped by run start. For
    # smokes where N is small and runs are sequential the ordering matches.
    for r, p in zip(runs, predictions):
        r.task_id = p.get("task_id", r.task_id)
        bench_r = bench_results_by_task.get(p.get("task_id"))
        if bench_r is not None:
            r.bench_passed = bool(bench_r.get("passed"))

    # Pass criteria checks (cgpro Path E B' minimum).
    seam_verdicts = Counter(
        (r.seam_verdict_source, r.seam_quality_label) for r in runs
    )
    has_exact_pass = any(
        r.seam_verdict_source == "exact"
        and r.seam_quality_label == "pass"
        and r.seam_trainable is True
        for r in runs
    )
    has_exact_fail = any(
        r.seam_verdict_source == "exact"
        and r.seam_quality_label == "fail"
        and r.seam_trainable is True
        for r in runs
    )
    event_order_pass = all(r.event_order_ok for r in runs if r.seam_verdict_source)
    raw_leaks_pass = len(raw_leak_findings) == 0

    # Cross-check seam vs bench (escalation may diverge).
    cross_check_rows: list[dict[str, Any]] = []
    seam_match_bench = 0
    seam_diverged_bench = 0
    seam_unknown = 0
    for r in runs:
        seam_pass = (
            r.seam_quality_label == "pass" if r.seam_verdict_source == "exact" else None
        )
        if r.bench_passed is None or seam_pass is None:
            seam_unknown += 1
            note = "bench_passed_unknown" if r.bench_passed is None else "seam_abstain"
        elif seam_pass == r.bench_passed:
            seam_match_bench += 1
            note = "agree"
        else:
            seam_diverged_bench += 1
            note = "diverged_likely_escalation"
        cross_check_rows.append(
            {
                "task_id": r.task_id,
                "run_id": r.run_id,
                "seam_verdict_source": r.seam_verdict_source,
                "seam_quality_label": r.seam_quality_label,
                "seam_score": r.seam_score,
                "seam_trainable": r.seam_trainable,
                "bench_passed_final": r.bench_passed,
                "runtime_delta_count": r.runtime_delta_count,
                "final_status": r.final_status,
                "event_order_ok": r.event_order_ok,
                "cross_check": note,
            }
        )

    # Write canonical BCB predictions (strip _trace; raw_solution NOT
    # recoverable for this run since the bench writer didn't capture
    # raw_solution before R6.1a-Path-E-step-3 — minor R6.1b ticket).
    canonical_preds = []
    for p in predictions:
        canon = {"task_id": p.get("task_id"), "solution": p.get("solution", "")}
        canonical_preds.append(canon)
    args.out_canonical_predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.out_canonical_predictions.open("w", encoding="utf-8") as fh:
        for cp in canonical_preds:
            fh.write(json.dumps(cp, ensure_ascii=False) + "\n")

    # SHA-256 manifest of all artifacts.
    manifest = {
        "validator_version": "path_e_step3_v1",
        "bench_report_sha256": _sha256(args.bench_report),
        "predictions_jsonl_sha256": _sha256(args.predictions),
        "canonical_predictions_jsonl_sha256": _sha256(args.out_canonical_predictions),
        "jsonl_traces": {
            jf.name: _sha256(jf) for jf in jsonl_files
        },
    }
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Markdown report.
    md_lines = [
        "# Path E step 3 — BigCodeBench Hard Instruct N=10 seam validation",
        "",
        "**Date**: 2026-04-29",
        "**Cycle**: 6 R6.1a verify Path E (post Gate D)",
        "**Purpose**: prove the live ``verdict_source=\"exact\", trainable=True`` "
        "contract on a synchronous-eval bench via the bench-result feedback seam.",
        "",
        "## Honest framing locks",
        "",
        "- This run is **NOT** a BigCodeBench leaderboard submission. The "
        "leaderboard reports **calibrated Pass@1** with greedy decoding through "
        "the official ``bigcodebench.evaluate`` harness (or its e2b/gradio "
        "backends). On Windows the official ``untrusted_check`` path fails on "
        "``os.killpg`` and coerces every task to ``timeout``; the seam evaluator "
        "documents a fall-back to ``BigCodeBenchBench._evaluate_solution_with_stderr``"
        " (matplotlib-headless subprocess, deterministic per (solution, test_code)) "
        "and tags ``bench_result.verifier_id`` accordingly.",
        "- This is a **seam validation smoke**, not a value/regression benchmark.",
        "- Per AUDIT2 2026-04-24 framing rule: no \"above SOTA\" or leaderboard-"
        "style claims attached to this number.",
        "",
        "## Setup",
        "",
        "- ``SAGE_ORACLE=1``, ``SAGE_RUN_FRAME=1``, ``SAGE_BENCH_ORACLE_SEAM=1``, ``SAGE_DIFF_VERIFIER_MODE=observe``",
        "- ``StateCore`` OFF (``SAGE_STATECORE`` unset).",
        "- Throwaway bandit DB: production state moved to "
        "``.tmp/path_e_backup/`` pre-bench, restored post-bench. Production "
        "posteriors not polluted.",
        "- SSL: ``SSL_CERT_FILE`` + ``REQUESTS_CA_BUNDLE`` + ``CURL_CA_BUNDLE`` + "
        "``GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`` set to "
        "``C:/Code/certs/windows-full-bundle.pem``.",
        "- Greedy decoding: SAGE pipeline default temperature settings; not "
        "the BCB CLI ``--temp 0`` enforcement (separate from the seam contract).",
        "- Single entry point: ``python -m sage.bench --type bigcodebench "
        "--subset hard --split instruct --limit 10`` — no parallel scripts.",
        "",
        "## cgpro Path E B' minimum pass criteria",
        "",
    ]
    md_lines.append("| # | Criterion | Result |")
    md_lines.append("|---|---|---|")
    md_lines.append(
        f"| 1 | ≥1 ``verdict_source='exact', quality_label='pass', trainable=True`` | "
        f"{'PASS' if has_exact_pass else 'FAIL'} |"
    )
    md_lines.append(
        f"| 2 | ≥1 ``verdict_source='exact', quality_label='fail', trainable=True`` | "
        f"{'PASS' if has_exact_fail else 'FAIL'} |"
    )
    md_lines.append(
        f"| 3 | Event order ``final_result < oracle_verdict < run_frame_summary`` "
        f"on every run | {'PASS' if event_order_pass else 'FAIL'} "
        f"({sum(1 for r in runs if r.event_order_ok)}/{len(runs)}) |"
    )
    md_lines.append(
        f"| 4 | No raw stdout/stderr/raw_output/raw_patch leaks in any payload "
        f"| {'PASS' if raw_leaks_pass else 'FAIL'} ({len(raw_leak_findings)} leaks) |"
    )

    md_lines += [
        "",
        "## Seam-vs-bench cross-check (per task)",
        "",
        "Escalation/repair may turn a first-attempt seam fail into a final bench "
        "pass; the seam captures the **first-attempt** verdict, the bench report "
        "captures the **final** outcome.",
        "",
        "| task_id | seam source | seam label | seam score | seam trainable | "
        "bench passed (final) | runtime_deltas | cross-check |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in cross_check_rows:
        md_lines.append(
            f"| {row['task_id']} | {row['seam_verdict_source']} | "
            f"{row['seam_quality_label']} | {row['seam_score']} | "
            f"{row['seam_trainable']} | {row['bench_passed_final']} | "
            f"{row['runtime_delta_count']} | {row['cross_check']} |"
        )

    md_lines += [
        "",
        f"**Cross-check totals**: agree={seam_match_bench}, "
        f"diverged_likely_escalation={seam_diverged_bench}, "
        f"unknown/abstain={seam_unknown}.",
        "",
        "## Verdict-source distribution",
        "",
    ]
    for (src, label), count in seam_verdicts.most_common():
        md_lines.append(f"- ``{src}/{label}``: {count}")

    md_lines += [
        "",
        "## Reproducibility",
        "",
        "- Repo: https://github.com/yannabadie/YGN-SAGE",
        "- All artifacts SHA-256-hashed in the manifest below.",
        "- Bench command (canonical, single entry point):",
        "",
        "```bash",
        "SAGE_ORACLE=1 SAGE_RUN_FRAME=1 SAGE_BENCH_ORACLE_SEAM=1 \\",
        "  SAGE_TRACE_JSONL_DIR=.tmp/path_e_artifacts/jsonl_n10 \\",
        "  python -m sage.bench --type bigcodebench --subset hard --split instruct \\",
        "                       --limit 10 --output report.json",
        "```",
        "",
        "## Manifest (SHA-256)",
        "",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
    ]
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(md_lines), encoding="utf-8")

    # Print summary to stdout for CI.
    overall_pass = (
        has_exact_pass and has_exact_fail and event_order_pass and raw_leaks_pass
    )
    print(f"Seam Path E step 3 verdict: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  Runs analyzed: {len(runs)}")
    print(f"  Seam Exact pass: {has_exact_pass}")
    print(f"  Seam Exact fail: {has_exact_fail}")
    print(f"  Event order pass: {event_order_pass}")
    print(f"  Raw output leaks: {len(raw_leak_findings)}")
    print(f"  Cross-check agree/diverged/unknown: "
          f"{seam_match_bench}/{seam_diverged_bench}/{seam_unknown}")
    print(f"  Manifest: {args.out_manifest}")
    print(f"  Report: {args.out_report}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
