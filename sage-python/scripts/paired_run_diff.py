#!/usr/bin/env python3
"""Slice 10A — paired run differ (cgpro VERIFY RF#C MODIFY).

Given two canary run artefacts on the same instance set, surface the
differences in `topology_audit` so attribution between prompt/profile
changes vs bandit-Thompson noise is observable. Per cgpro VERIFY:

> "Pour les benchs, il faut rendre la distribution auditée: seed si
>  disponible, posterior epoch, selected template, node count,
>  per-node role/model, et control surface complet. Pour attribution,
>  faire des paired reruns, pas un override déterministe."

This script consumes ``summary.json`` from each run dir (the slice
10A topology_audit block must be present, which means both runs must
be on commit ≥ slice 10A's HEAD).

Usage::

    python sage-python/scripts/paired_run_diff.py \\
        --run-a docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/run \\
        --run-b docs/benchmarks/2026-05-12-canary-paired-rerun/run \\
        --output docs/benchmarks/2026-05-12-paired-diff.json

Exit codes:
- 0  — paired runs cover the same instances; report written
- 2  — instance-set mismatch (one run is missing tasks the other has)
- 3  — at least one run's summary.json lacks ``topology_audit``
       (was generated pre-slice-10A); upgrade required
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("paired_run_diff")


def _load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing summary.json in {run_dir}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _per_task_dict(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["instance_id"]: t for t in summary.get("task_summaries", [])}


def _diff_topology(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare two topology_audit blocks. Returns a structured diff."""
    if a is None and b is None:
        return {"both_missing_audit": True}
    if a is None or b is None:
        return {"one_side_missing_audit": True, "a_present": a is not None, "b_present": b is not None}

    diff: dict[str, Any] = {}

    # Flat fields to compare
    flat_keys = (
        "topology_template",
        "topology_id",
        "node_count",
        "edge_count",
        "routing_model_id",
        "routing_source",
        "routing_system",
        "routing_domain",
    )
    for k in flat_keys:
        va, vb = a.get(k), b.get(k)
        if va != vb:
            diff[k] = {"a": va, "b": vb}

    # routing_confidence: tolerate float drift up to 0.001
    ca, cb = a.get("routing_confidence"), b.get("routing_confidence")
    if isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
        if abs(ca - cb) > 1e-3:
            diff["routing_confidence"] = {"a": ca, "b": cb, "delta": cb - ca}
    elif ca != cb:
        diff["routing_confidence"] = {"a": ca, "b": cb}

    # Per-node comparison: by node_id position
    nodes_a = {str(n.get("node_id")): n for n in (a.get("nodes") or [])}
    nodes_b = {str(n.get("node_id")): n for n in (b.get("nodes") or [])}
    node_keys = sorted(set(nodes_a) | set(nodes_b), key=lambda x: int(x or "0"))
    per_node_diff: dict[str, Any] = {}
    for nk in node_keys:
        na, nb = nodes_a.get(nk), nodes_b.get(nk)
        node_d: dict[str, Any] = {}
        if na is None or nb is None:
            node_d["existence"] = {
                "a": na is not None,
                "b": nb is not None,
            }
        else:
            for nfield in (
                "assigned_role",
                "assigned_model_id",
                "assigned_provider_id",
                "completed_model_id",
                "completed_provider_id",
                "is_sentinel",
            ):
                if na.get(nfield) != nb.get(nfield):
                    node_d[nfield] = {"a": na.get(nfield), "b": nb.get(nfield)}
        if node_d:
            per_node_diff[nk] = node_d
    if per_node_diff:
        diff["nodes"] = per_node_diff

    # Oracle
    oa, ob = a.get("oracle"), b.get("oracle")
    if oa != ob:
        diff["oracle"] = {"a": oa, "b": ob}

    # Control surface
    csa = a.get("control_surface") or {}
    csb = b.get("control_surface") or {}
    cs_diff = {}
    for k in set(csa) | set(csb):
        if csa.get(k) != csb.get(k):
            cs_diff[k] = {"a": csa.get(k), "b": csb.get(k)}
    if cs_diff:
        diff["control_surface"] = cs_diff

    # Substitution detection
    if a.get("provider_policy_substitution_detected") != b.get("provider_policy_substitution_detected"):
        diff["provider_policy_substitution_detected"] = {
            "a": a.get("provider_policy_substitution_detected"),
            "b": b.get("provider_policy_substitution_detected"),
        }

    return diff


def _summarize_diffs(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Tally what kinds of differences appeared across tasks."""
    counts: dict[str, int] = {}
    for _iid, d in per_task.items():
        if "topology_template" in d:
            counts["template_changed"] = counts.get("template_changed", 0) + 1
        if "topology_id" in d:
            counts["topology_id_changed"] = counts.get("topology_id_changed", 0) + 1
        if "node_count" in d:
            counts["node_count_changed"] = counts.get("node_count_changed", 0) + 1
        if "routing_model_id" in d:
            counts["routing_model_changed"] = counts.get("routing_model_changed", 0) + 1
        if "nodes" in d:
            counts["per_node_differences"] = counts.get("per_node_differences", 0) + 1
        if "control_surface" in d:
            counts["control_surface_drift"] = counts.get("control_surface_drift", 0) + 1
        if not d:
            counts["identical"] = counts.get("identical", 0) + 1
    return counts


def run_paired_diff(run_a: Path, run_b: Path) -> dict[str, Any]:
    summary_a = _load_summary(run_a)
    summary_b = _load_summary(run_b)

    tasks_a = _per_task_dict(summary_a)
    tasks_b = _per_task_dict(summary_b)

    # Instance set must match
    set_a = set(tasks_a)
    set_b = set(tasks_b)
    if set_a != set_b:
        return {
            "schema_version": "paired_run_diff_v1",
            "run_a": str(run_a),
            "run_b": str(run_b),
            "instance_set_mismatch": True,
            "only_in_a": sorted(set_a - set_b),
            "only_in_b": sorted(set_b - set_a),
        }

    # Verify topology_audit present on both sides
    missing_audit_a = [iid for iid, t in tasks_a.items() if "topology_audit" not in t]
    missing_audit_b = [iid for iid, t in tasks_b.items() if "topology_audit" not in t]
    if missing_audit_a or missing_audit_b:
        return {
            "schema_version": "paired_run_diff_v1",
            "run_a": str(run_a),
            "run_b": str(run_b),
            "topology_audit_missing_in_a": missing_audit_a,
            "topology_audit_missing_in_b": missing_audit_b,
            "hint": "regenerate the missing-side run on a HEAD >= slice 10A",
        }

    per_task: dict[str, dict[str, Any]] = {}
    for iid in sorted(set_a):
        a_audit = tasks_a[iid].get("topology_audit")
        b_audit = tasks_b[iid].get("topology_audit")
        per_task[iid] = _diff_topology(a_audit, b_audit)

    return {
        "schema_version": "paired_run_diff_v1",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "n_tasks": len(set_a),
        "summary_counts": _summarize_diffs(per_task),
        "per_task": per_task,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    result = run_paired_diff(args.run_a, args.run_b)
    output_path = args.output or (Path(".") / "paired-run-diff.json")
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.get("instance_set_mismatch"):
        print(
            f"FAIL: instance set mismatch (a-only={result['only_in_a']} b-only={result['only_in_b']})",
            file=sys.stderr,
        )
        return 2
    if result.get("topology_audit_missing_in_a") or result.get("topology_audit_missing_in_b"):
        print(
            f"FAIL: topology_audit missing on one side (a={len(result.get('topology_audit_missing_in_a', []))}, b={len(result.get('topology_audit_missing_in_b', []))})",
            file=sys.stderr,
        )
        return 3

    counts = result.get("summary_counts", {})
    print(
        f"Paired diff complete: {result['n_tasks']} tasks, drift counts={counts} -> {output_path}",
        file=sys.stderr,
    )
    if args.verbose:
        for iid, d in result.get("per_task", {}).items():
            log.info(f"  {iid[:50]}: {sorted(d.keys()) if d else '(identical)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
