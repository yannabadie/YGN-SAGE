#!/usr/bin/env python3
"""Build SWE-bench Pro grader's raw_sample CSV for a given instance subset.

Per `swe_bench_pro_eval.py` in scaleapi/SWE-bench_Pro-os: the grader
expects a CSV with columns:
    instance_id, repo, before_repo_set_cmd, selected_test_files_to_run,
    base_commit, fail_to_pass, pass_to_pass, FAIL_TO_PASS, PASS_TO_PASS

The HuggingFace dataset (`ScaleAI/SWE-bench_Pro` test split) provides
the test sets under lowercase names (`fail_to_pass`, `pass_to_pass`).
The grader code is inconsistent: comments mention uppercase columns,
while the scoring path reads lowercase. This script writes both.

`base_dockerfile` and `instance_dockerfile` columns are NOT required
in the CSV — the grader loads them from
`dockerfiles/{base,instance}_dockerfile/{iid}/Dockerfile` on disk.

Usage:
    python -m sage_python.scripts.swebench_pro_build_grader_csv \\
        --instance-ids instance_future-architect__vuls-139f3a... \\
        --output sage-python/data/swebench_pro/grader_n1.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

log = logging.getLogger("sage.bench.swebench_pro_build_grader_csv")

_GRADER_COLUMNS = (
    "instance_id",
    "repo",
    "before_repo_set_cmd",
    "selected_test_files_to_run",
    "base_commit",
    "fail_to_pass",
    "pass_to_pass",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def build_csv(instance_ids: Iterable[str], output: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        log.error("Missing `datasets` library; pip install datasets")
        raise SystemExit(1) from exc

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    target_ids = set(instance_ids)
    found: dict[str, dict] = {}

    for row in ds:
        iid = row["instance_id"]
        if iid in target_ids:
            found[iid] = {
                "instance_id": iid,
                # Required by helper_code/image_uri.py to resolve
                # jefzda/sweap-images:<repo-derived-tag>.
                "repo": row["repo"],
                "before_repo_set_cmd": row["before_repo_set_cmd"],
                "selected_test_files_to_run": row["selected_test_files_to_run"],
                "base_commit": row["base_commit"],
                "fail_to_pass": row["fail_to_pass"],
                "pass_to_pass": row["pass_to_pass"],
                # Compatibility with grader comments / auxiliary tooling.
                "FAIL_TO_PASS": row["fail_to_pass"],
                "PASS_TO_PASS": row["pass_to_pass"],
            }
        if len(found) == len(target_ids):
            break

    missing = target_ids - found.keys()
    if missing:
        log.error("Missing %d instance IDs in dataset: %s", len(missing), sorted(missing))
        raise SystemExit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_GRADER_COLUMNS)
        writer.writeheader()
        for iid in instance_ids:
            writer.writerow(found[iid])
    log.info("Wrote %d rows to %s", len(found), output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        required=True,
        help="One or more SWE-bench Pro instance_ids to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output path for grader CSV",
    )
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

    build_csv(args.instance_ids, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
