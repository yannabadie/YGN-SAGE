#!/usr/bin/env python3
"""Slice 10C — synth-diff-fidelity-audit.

cgpro VERIFY 2026-05-11 on conv ``cgpro_ygn_sage_global_analysis_20260510``
upheld RF#B "synth rewrap may alter diff fidelity" as MODIFY: do NOT
extract the patch from the coder/actor's raw output bypassing the synth
(``final_result.output`` stays canonical); instead, AUDIT whether the
synth/judge/output role actually alters the diff content vs what the
coder/actor emitted.

This script reads an existing canary run artefact directory (the slice 9
output at ``docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/``
or any later run) and produces a per-task fidelity report:

- coder_or_actor_raw_diff: SHA-256 + char count of the diff extracted from
  the first non-sentinel node_completed event
- final_diff: SHA-256 + char count of the diff in ``predictions.json``
- byte_identical: bool
- files_dropped / files_added: file-path set diff between extraction points
- hunks_count_pre / hunks_count_post: ``@@`` block counts
- chars_delta: int
- verdict: ``preserved`` | ``cosmetic_drift`` | ``files_altered`` | ``rewritten``

Usage:

    python sage-python/scripts/diff_fidelity_audit.py \\
        --run-dir docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/run \\
        --output  docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/synth_diff_fidelity.json

Exit codes:
- 0 — at least one task `preserved` AND zero tasks `rewritten`
- 1 — any task verdict is `rewritten` (judge / output / synth produced
  a diff with paths the coder/actor did NOT have)
- 2 — usage / invalid input
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("diff_fidelity_audit")


_SENTINEL_MARKER = "[sage: agent exited after"


def _extract_patch(text: str) -> str:
    """Mirror of ``sage.bench.swebench_bench._extract_patch`` so this
    audit reproduces what the canary actually emitted.
    """
    if not text:
        return ""
    if _SENTINEL_MARKER in text:
        return ""
    text = text.replace("\r\n", "\n").strip()

    if text.startswith("diff --git") or text.startswith("---"):
        return text + ("\n" if not text.endswith("\n") else "")

    for marker in ("```diff", "```patch", "```"):
        if marker in text:
            start = text.find(marker)
            start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                candidate = text[start:end].strip()
                if candidate and ("---" in candidate or "diff --git" in candidate):
                    return candidate + ("\n" if not candidate.endswith("\n") else "")

    lines = text.split("\n")
    diff_lines: list[str] = []
    in_diff = False
    for line in lines:
        if line.startswith("diff --git") or line.startswith("---"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
        if (
            in_diff
            and not line.strip()
            and diff_lines
            and not diff_lines[-1].startswith(("diff", "---", "+++", "@@", " ", "+", "-"))
        ):
            break

    if diff_lines:
        patch = "\n".join(diff_lines).strip()
        return patch + ("\n" if not patch.endswith("\n") else "")
    return ""


def _diff_files(diff_text: str) -> list[str]:
    """Return the ordered list of file paths a unified diff touches."""
    return re.findall(r"^diff --git a/(\S+) b/", diff_text, re.M)


def _hunk_count(diff_text: str) -> int:
    """Count ``@@ -X,Y +X,Y @@`` hunk headers."""
    return len(re.findall(r"^@@ ", diff_text, re.M))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _classify_fidelity(
    pre_text: str,
    post_text: str,
    pre_files: list[str],
    post_files: list[str],
) -> str:
    """Verdict per cgpro VERIFY decision tree.

    - ``preserved`` — pre == post byte-identical.
    - ``cosmetic_drift`` — files identical, char count differs by <5%.
    - ``files_altered`` — non-empty intersection but at least one file
      dropped or added.
    - ``rewritten`` — file set disjoint (the post stage emitted a
      diff for entirely different paths).
    """
    if pre_text == post_text:
        return "preserved"

    pre_set = set(pre_files)
    post_set = set(post_files)

    if not pre_set or not post_set:
        # one side has no parseable diff
        if not pre_set and post_set:
            return "files_altered"  # post invented files
        return "preserved"  # pre had no files; nothing to alter

    if pre_set == post_set:
        # Same files, content shifted: cosmetic if small, content if large
        if pre_text and post_text:
            delta = abs(len(post_text) - len(pre_text))
            if delta / max(len(pre_text), 1) < 0.05:
                return "cosmetic_drift"
        return "files_altered"

    intersection = pre_set & post_set
    if not intersection:
        return "rewritten"
    return "files_altered"


def _first_non_sentinel_node_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first ``node_completed`` event whose payload is a
    non-sentinel string. The canary's contract is that the first node
    (coder for sequential, actor for AVR) carries the substantive
    output.
    """
    for ev in events:
        if ev.get("event_type") != "node_completed":
            continue
        payload = ev.get("payload")
        if not isinstance(payload, str):
            continue
        if _SENTINEL_MARKER in payload:
            continue
        return ev
    return None


def _last_node_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the LAST node_completed event whose payload is a string.
    The canary writes ``predictions.json`` from ``final_result.output``
    which is the last node's output — capture it here for completeness
    (and to cross-check against predictions.json).
    """
    for ev in reversed(events):
        if ev.get("event_type") != "node_completed":
            continue
        payload = ev.get("payload")
        if isinstance(payload, str):
            return ev
    return None


def audit_task(
    events_path: Path,
    final_patch: str,
) -> dict[str, Any]:
    """Run the fidelity audit for a single per-task events file."""
    events: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    first_node = _first_non_sentinel_node_payload(events)
    last_node = _last_node_payload(events)

    if first_node is None:
        # All nodes sentinel-ed — audit not applicable.
        return {
            "events_path": str(events_path),
            "first_substantive_node": None,
            "last_node": None,
            "final_predictions_patch": {
                "chars": len(final_patch),
                "sha256": _sha256(final_patch),
            },
            "verdict": "all_sentinel",
        }

    pre_raw = first_node.get("payload", "") if isinstance(first_node.get("payload"), str) else ""
    pre_diff = _extract_patch(pre_raw)
    pre_files = _diff_files(pre_diff)
    pre_hunks = _hunk_count(pre_diff)
    pre_node = {
        "role": first_node.get("node_role"),
        "model_id": first_node.get("model_id"),
        "provider_id": first_node.get("provider_id"),
        "raw_output_chars": len(pre_raw),
        "extracted_diff_chars": len(pre_diff),
        "extracted_diff_sha256": _sha256(pre_diff),
        "files": pre_files,
        "hunks": pre_hunks,
    }

    last_raw = last_node.get("payload", "") if last_node and isinstance(last_node.get("payload"), str) else ""
    last_diff = _extract_patch(last_raw)
    last_files = _diff_files(last_diff)
    last_hunks = _hunk_count(last_diff)
    last_node_info = {
        "role": last_node.get("node_role") if last_node else None,
        "model_id": last_node.get("model_id") if last_node else None,
        "raw_output_chars": len(last_raw),
        "extracted_diff_chars": len(last_diff),
        "extracted_diff_sha256": _sha256(last_diff),
        "files": last_files,
        "hunks": last_hunks,
    }

    verdict = _classify_fidelity(pre_diff, last_diff, pre_files, last_files)

    files_dropped = sorted(set(pre_files) - set(last_files))
    files_added = sorted(set(last_files) - set(pre_files))

    return {
        "events_path": str(events_path),
        "first_substantive_node": pre_node,
        "last_node": last_node_info,
        "final_predictions_patch": {
            "chars": len(final_patch),
            "sha256": _sha256(final_patch),
        },
        "files_dropped_pre_to_post": files_dropped,
        "files_added_pre_to_post": files_added,
        "chars_delta": len(last_diff) - len(pre_diff),
        "byte_identical": pre_diff == last_diff,
        "verdict": verdict,
    }


def run_audit(run_dir: Path) -> dict[str, Any]:
    """Audit all per-task events in ``run_dir``. ``run_dir`` is
    expected to be the canary's ``run/`` directory, containing
    ``predictions.json`` and ``per_task/<iid>.events.jsonl``.
    """
    per_task_dir = run_dir / "per_task"
    predictions_path = run_dir / "predictions.json"
    if not per_task_dir.is_dir():
        raise SystemExit(2)
    if not predictions_path.is_file():
        raise SystemExit(2)

    predictions = {
        rec["instance_id"]: rec.get("patch", "")
        for rec in json.loads(predictions_path.read_text(encoding="utf-8"))
    }

    results: list[dict[str, Any]] = []
    for ev_path in sorted(per_task_dir.glob("*.events.jsonl")):
        instance_id = ev_path.name.replace(".events.jsonl", "")
        final_patch = predictions.get(instance_id, "")
        per_task = audit_task(ev_path, final_patch)
        per_task["instance_id"] = instance_id
        results.append(per_task)

    # Verdict tally
    tally: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1

    return {
        "schema_version": "diff_fidelity_audit_v1",
        "run_dir": str(run_dir),
        "n_tasks": len(results),
        "verdict_tally": tally,
        "any_rewritten": tally.get("rewritten", 0) > 0,
        "all_preserved_or_cosmetic": all(
            r["verdict"] in {"preserved", "cosmetic_drift"} for r in results
        ),
        "per_task": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="canary `run/` dir containing predictions.json + per_task/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the JSON report (default: <run-dir>/../synth_diff_fidelity.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="log per-task verdicts to stderr",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    audit = run_audit(args.run_dir)

    output_path = args.output or (args.run_dir.parent / "synth_diff_fidelity.json")
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.verbose:
        for r in audit["per_task"]:
            log.info(
                "  %s verdict=%s pre_role=%s post_role=%s files_dropped=%s files_added=%s",
                r["instance_id"][:50],
                r["verdict"],
                (r.get("first_substantive_node") or {}).get("role"),
                (r.get("last_node") or {}).get("role"),
                r.get("files_dropped_pre_to_post"),
                r.get("files_added_pre_to_post"),
            )

    print(
        f"Audit complete: {audit['n_tasks']} tasks, verdicts={audit['verdict_tally']} -> {output_path}",
        file=sys.stderr,
    )

    if audit["any_rewritten"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
