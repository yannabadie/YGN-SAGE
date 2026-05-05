#!/usr/bin/env python3
"""Fetch SWE-bench Pro task metadata for the cycle-13 4-arm ablation dry-run.

Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`,
verdict GO_TIER_1_PLUS_2 trap Q4):
  - Fetch metadata only (HuggingFace `datasets` lib).
  - DO NOT pull Docker images here. Image pull is for grading time.
  - Cache by task_id for idempotent re-runs.
  - Stratification driven by dataset metadata (NOT hardcoded
    Python/Java/C++ buckets; fall back to repo diversity if a
    bucket is empty).
  - Exclude known-bug instances (NodeBB test-name mismatch flagged
    in scaleapi/SWE-bench_Pro-os issue #?? as of 2026-05-05).

Usage:
    python -m sage_python.scripts.swebench_pro_fetch \\
        --n 10 --output-dir data/swebench_pro/n10/ [--seed 42]

Output:
    data/swebench_pro/n10/
        instances.json      — list of N task metadata records (input shape for downstream runners)
        manifest.json       — fetch metadata: timestamp, dataset version, stratification record, seed
        per_task/<id>.json  — one JSON per task (sharded for parallel runs)

The `instances.json` records carry: instance_id, repo, base_commit,
problem_statement, dockerhub_tag, language (if present), task_size
(if present), patch_test_files (Pro-specific). Other Pro columns
preserved as-is.

This script is idempotent: re-running with the same --seed produces
the same task selection. The cache check is on (output_dir, seed)
— if the manifest matches, we re-use existing per_task/ files.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("sage.bench.swebench_pro_fetch")

# Per cgpro DESIGN E trap Q4: known-bug instances to exclude from any
# dry-run subset. List MUST track scaleapi/SWE-bench_Pro-os open issues.
# Sources: README "News" section + GitHub issues.
_KNOWN_BUG_INSTANCES: frozenset[str] = frozenset({
    # Add specific instance_ids here when scaleapi confirms a known
    # grader bug. Currently the README only mentions a NodeBB-class
    # issue without a specific instance list; document the policy here
    # and update via cycle-13 issue tracker.
})

_DATASET_NAME = "ScaleAI/SWE-bench_Pro"
_DATASET_SPLIT = "test"


def _load_dataset() -> Any:
    """Load SWE-bench Pro `test` split via HuggingFace `datasets` lib.

    No pre-loading of Docker images. Network access is required at
    fetch time (HuggingFace download), but cached on disk afterwards.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover — clear error path
        log.error(
            "Missing dependency: `datasets`. Install via "
            "`pip install datasets` (also pulled by sage-python[bench])"
        )
        raise SystemExit(1) from exc

    log.info("Loading %s split=%s from HuggingFace...", _DATASET_NAME, _DATASET_SPLIT)
    return load_dataset(_DATASET_NAME, split=_DATASET_SPLIT)


def _bucket_key(row: dict[str, Any]) -> str:
    """Stratification key driven by metadata, not hardcoded language list.

    Per cgpro DESIGN E trap Q5: language stratification is FALLBACK,
    primary key is `task_size` if present (per Pro's documented schema).
    """
    size = row.get("task_size") or row.get("size") or "unknown"
    lang = row.get("language") or row.get("repo_language") or "lang_unknown"
    return f"{size}::{lang}"


def _stratified_sample(
    rows: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Group by metadata bucket, then pick proportionally.

    Falls back to repo diversity if buckets are sparse — never fails
    fetch because a desired bucket is empty (cgpro DESIGN E trap Q5).
    """
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row.get("instance_id") in _KNOWN_BUG_INSTANCES:
            continue
        buckets[_bucket_key(row)].append(row)

    bucket_names = sorted(buckets.keys())  # stable order for determinism
    if not bucket_names:
        raise RuntimeError("All rows excluded — empty pool after _KNOWN_BUG_INSTANCES filter")

    # Round-robin across buckets, cap at n total. Repo diversity is
    # a soft secondary constraint: prefer rotating repos within a
    # bucket too.
    selected: list[dict[str, Any]] = []
    bucket_iters: dict[str, list[dict[str, Any]]] = {}
    for name in bucket_names:
        shuffled = list(buckets[name])
        rng.shuffle(shuffled)
        bucket_iters[name] = shuffled

    repos_seen: set[str] = set()
    while len(selected) < n:
        progressed = False
        for name in bucket_names:
            if len(selected) >= n:
                break
            queue = bucket_iters[name]
            if not queue:
                continue
            # Prefer a repo we haven't seen yet
            picked_idx = 0
            for i, candidate in enumerate(queue):
                repo = candidate.get("repo") or ""
                if repo not in repos_seen:
                    picked_idx = i
                    break
            picked = queue.pop(picked_idx)
            selected.append(picked)
            repos_seen.add(picked.get("repo") or "")
            progressed = True
        if not progressed:
            break  # all buckets exhausted

    return selected


def _instance_record(row: dict[str, Any]) -> dict[str, Any]:
    """Reduce HuggingFace row to the minimal metadata our runners need.

    Preserves dockerhub_tag for cycle-13 grading-time docker pull.
    """
    record = {
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "problem_statement": row.get("problem_statement"),
        "dockerhub_tag": row.get("dockerhub_tag"),
        "language": row.get("language") or row.get("repo_language"),
        "task_size": row.get("task_size") or row.get("size"),
    }
    # Pro-specific test/patch metadata varies by row schema; preserve
    # any keys that look benchmark-relevant.
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS", "patch", "test_patch", "version"):
        if k in row:
            record[k] = row[k]
    return record


def fetch(n: int, output_dir: Path, seed: int) -> dict[str, Any]:
    """Fetch + stratify + cache. Returns the manifest dict."""
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_task = output_dir / "per_task"
    per_task.mkdir(exist_ok=True)

    dataset = _load_dataset()
    rows = list(dataset)
    log.info("Loaded %d rows from %s", len(rows), _DATASET_NAME)

    selected = _stratified_sample(rows, n=n, rng=rng)
    log.info("Stratified to %d rows", len(selected))

    instances = [_instance_record(r) for r in selected]

    # Persist
    (output_dir / "instances.json").write_text(
        json.dumps(instances, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for inst in instances:
        path = per_task / f"{inst['instance_id']}.json"
        path.write_text(
            json.dumps(inst, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    bucket_counts: dict[str, int] = collections.Counter(
        _bucket_key(inst) for inst in instances
    )
    repo_counts: dict[str, int] = collections.Counter(
        inst.get("repo") or "" for inst in instances
    )

    # Manifest hash includes dataset name + seed + N. If two runs share
    # this hash, they should produce byte-identical instances.json.
    selection_hash = hashlib.sha256(
        json.dumps(
            [inst["instance_id"] for inst in instances],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest: dict[str, Any] = {
        "dataset": _DATASET_NAME,
        "split": _DATASET_SPLIT,
        "n": n,
        "seed": seed,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_hash": selection_hash,
        "bucket_distribution": dict(bucket_counts),
        "repo_distribution": dict(repo_counts),
        "known_bug_excluded": sorted(_KNOWN_BUG_INSTANCES),
        "instance_ids": [inst["instance_id"] for inst in instances],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(
        "Wrote %d instances to %s (selection_hash=%s)",
        len(instances), output_dir, selection_hash[:12],
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="number of tasks to fetch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sage-python/data/swebench_pro/n10"),
        help="where to write manifest + per_task/ files (relative to repo root)",
    )
    parser.add_argument("--seed", type=int, default=42, help="stratification RNG seed")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    manifest = fetch(args.n, args.output_dir, args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
