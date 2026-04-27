#!/usr/bin/env python3
"""Aggregate diff-verifier reason/outcome telemetry from a predictions.jsonl.

A22 (commit ``133b86b5``) added two telemetry fields to SWE-bench
predictions when ``SAGE_DIFF_VERIFIER_MODE`` is ``observe`` or ``repair``:

* ``_diff_verifier_outcome``: a single roll-up reason per prediction
  (``clean`` / ``content_mismatch`` / ``malformed_hunk_header`` / ...).
* ``_diff_verifier_reasons``: the ordered stream of reason events for
  that prediction (may have duplicates when multiple hunks each produce
  a separate reason).

This script aggregates both across a JSONL file. Outcome distribution
answers the dashboard question ("how many of my patches are clean
vs flagged?"); reason distribution answers the per-event question
("which structural failures dominate?").

Usage:

    python scripts/analyze_diff_verifier_buckets.py \\
        docs/benchmarks/2026-04-23-observe.json-predictions.jsonl

    # Multiple files (concatenated): compares verifier behavior across
    # smokes
    python scripts/analyze_diff_verifier_buckets.py \\
        docs/benchmarks/2026-04-23-*-predictions.jsonl

    # JSON output (stable schema for downstream tooling):
    python scripts/analyze_diff_verifier_buckets.py --json predictions.jsonl

Input compatibility:

* Predictions written before A22 (commit ``133b86b5``) only carry
  ``_diff_verifier_mismatches`` (or none of the keys, if the run was
  in ``off`` mode). Such predictions are counted in the dedicated
  ``no_outcome_field`` / ``no_reasons_field`` buckets so an old
  predictions file still produces a meaningful summary.
* Predictions in ``off`` mode (none of the verifier keys present) are
  reported separately as ``unverified`` so dashboards can distinguish
  "verifier said clean" from "verifier never ran".

This script intentionally does not depend on the YGN-SAGE package — it
operates on plain JSONL so post-mortem analysis works against archived
predictions even when the local checkout has drifted from the version
that produced them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_KEY_OUTCOME = "_diff_verifier_outcome"
_KEY_REASONS = "_diff_verifier_reasons"
_KEY_MISMATCHES = "_diff_verifier_mismatches"


def _iter_records(paths: list[Path]) -> Any:
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield path, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"warning: skipping malformed line in {path}: {exc}",
                        file=sys.stderr,
                    )


def _aggregate(paths: list[Path]) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    total = 0
    no_outcome = 0
    no_reasons = 0
    unverified = 0  # neither outcome nor reasons nor mismatches present (off mode)
    legacy_only = 0  # mismatches present but no outcome/reasons (pre-A22)
    per_file: Counter[str] = Counter()

    for path, record in _iter_records(paths):
        total += 1
        per_file[str(path)] += 1

        has_outcome = _KEY_OUTCOME in record
        has_reasons = _KEY_REASONS in record
        has_mismatches = _KEY_MISMATCHES in record

        if not (has_outcome or has_reasons or has_mismatches):
            unverified += 1
            continue

        if has_mismatches and not (has_outcome or has_reasons):
            legacy_only += 1

        if has_outcome:
            outcome_counts[record[_KEY_OUTCOME]] += 1
        else:
            no_outcome += 1

        if has_reasons:
            for reason in record[_KEY_REASONS]:
                reason_counts[reason] += 1
        else:
            no_reasons += 1

    return {
        "total_predictions": total,
        "outcome_counts": dict(outcome_counts.most_common()),
        "reason_counts": dict(reason_counts.most_common()),
        "no_outcome_field": no_outcome,
        "no_reasons_field": no_reasons,
        "unverified_off_mode": unverified,
        "legacy_pre_a22_only_mismatches": legacy_only,
        "per_file_counts": dict(per_file.most_common()),
    }


def _format_text(stats: dict[str, Any]) -> str:
    total = stats["total_predictions"]
    if total == 0:
        return "No predictions found."

    lines: list[str] = []
    lines.append(f"Total predictions: {total}")
    if stats["unverified_off_mode"]:
        lines.append(
            f"  (of which {stats['unverified_off_mode']} ran in off-mode — no verifier keys)"
        )
    if stats["legacy_pre_a22_only_mismatches"]:
        lines.append(
            f"  (of which {stats['legacy_pre_a22_only_mismatches']} are pre-A22 — only "
            f"_diff_verifier_mismatches present)"
        )

    lines.append("")
    lines.append("Outcome distribution (single roll-up per prediction):")
    if stats["outcome_counts"]:
        for outcome, count in stats["outcome_counts"].items():
            pct = 100.0 * count / total
            lines.append(f"  {outcome:<32}  {count:>5}  ({pct:5.1f}%)")
    else:
        lines.append("  (no _diff_verifier_outcome fields found)")
    if stats["no_outcome_field"]:
        pct = 100.0 * stats["no_outcome_field"] / total
        lines.append(
            f"  {'<no outcome field>':<32}  {stats['no_outcome_field']:>5}  ({pct:5.1f}%)"
        )

    lines.append("")
    lines.append("Reason distribution (per-event, may have duplicates):")
    if stats["reason_counts"]:
        for reason, count in stats["reason_counts"].items():
            lines.append(f"  {reason:<32}  {count:>5}")
    else:
        lines.append("  (no _diff_verifier_reasons fields found)")
    if stats["no_reasons_field"]:
        lines.append(f"  {'<no reasons field>':<32}  {stats['no_reasons_field']:>5}")

    if len(stats["per_file_counts"]) > 1:
        lines.append("")
        lines.append("Predictions per file:")
        for path, count in stats["per_file_counts"].items():
            lines.append(f"  {path}: {count}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analyze_diff_verifier_buckets",
        description=(
            "Aggregate _diff_verifier_outcome (primary) and "
            "_diff_verifier_reasons (secondary) across one or more "
            "predictions.jsonl files."
        ),
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="one or more predictions.jsonl files (use a shell glob for batch)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of human-readable text",
    )
    args = parser.parse_args(argv)

    missing = [str(p) for p in args.paths if not p.is_file()]
    if missing:
        print(
            "error: the following input files do not exist: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    stats = _aggregate(args.paths)

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=False))
    else:
        print(_format_text(stats))

    return 0


if __name__ == "__main__":
    sys.exit(main())
