#!/usr/bin/env python3
"""Convert YGN-SAGE agent output to SWE-bench Pro grader patch format.

Per cgpro DESIGN E trap Q5 (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`):

  SWE-bench Pro grader (`swe_bench_pro_eval.py` in scaleapi/SWE-bench_Pro-os)
  expects a JSON list of `{instance_id, patch, prefix?}` dicts. This is
  DIFFERENT from SWE-bench Lite's `{instance_id, model_name_or_path,
  model_patch}` shape that our existing `swebench_bench.py` writes.

  An explicit shape adapter prevents the "discover the grader rejects
  the shape after the API run" failure mode.

Usage:
    python -m sage_python.scripts.swebench_pro_format_patch \\
        --instance-id <task_id> \\
        --patch-file <unified_diff.patch> \\
        --prefix <run_label> \\
        --output predictions.json

Or via library API (used by the Tier 2.1 arm D runner):

    from swebench_pro_format_patch import format_patch, write_predictions

    record = format_patch(instance_id, patch_text, prefix)
    write_predictions([record], output_path)

Validation: the writer rejects records that don't match the Pro shape
exactly. Use `validate_record(record)` for shape-only checks.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TypedDict

log = logging.getLogger("sage.bench.swebench_pro_format_patch")


class ProPatchRecord(TypedDict, total=False):
    """Shape required by `swe_bench_pro_eval.py`.

    The grader (scaleapi/SWE-bench_Pro-os/swe_bench_pro_eval.py) loads
    a JSON list and reads `instance_id` + `patch` per record. `prefix`
    is optional but useful for tagging runs in batched evaluations.
    """

    instance_id: str
    patch: str
    prefix: str  # optional


_REQUIRED_KEYS: frozenset[str] = frozenset({"instance_id", "patch"})
_ALLOWED_KEYS: frozenset[str] = frozenset({"instance_id", "patch", "prefix"})


def validate_record(record: dict) -> None:
    """Shape-only validation. Raises ValueError on failure.

    Checks per the grader's expected schema:
      - instance_id is non-empty str.
      - patch is str (may be empty for "no-op patch" / canary cases).
      - prefix, if present, is str.
      - No extra keys beyond {instance_id, patch, prefix}.

    Does NOT validate the patch is a parseable unified diff -- that is
    the diff verifier's job (sage.bench.swebench_diff_verifier).
    """
    if not isinstance(record, dict):
        raise ValueError(f"Record must be dict, got {type(record).__name__}")

    # Check unexpected keys FIRST so migrators from SWE-bench Lite
    # ({instance_id, model_name_or_path, model_patch}) see the
    # diagnostic "unexpected keys" error rather than "missing patch".
    extra = set(record.keys()) - _ALLOWED_KEYS
    if extra:
        raise ValueError(f"Record has unexpected keys: {sorted(extra)}")

    missing = _REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"Record missing required keys: {sorted(missing)}")

    iid = record["instance_id"]
    if not isinstance(iid, str) or not iid.strip():
        raise ValueError(f"instance_id must be non-empty str, got {iid!r}")

    patch = record["patch"]
    if not isinstance(patch, str):
        raise ValueError(f"patch must be str, got {type(patch).__name__}")

    if "prefix" in record:
        prefix = record["prefix"]
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be str, got {type(prefix).__name__}")


def format_patch(
    instance_id: str,
    patch: str,
    prefix: str | None = None,
) -> ProPatchRecord:
    """Build one SWE-bench Pro patch record from an agent's output.

    Args:
        instance_id: The Pro task identifier (matches dataset's
            `instance_id` column).
        patch: Unified diff produced by the agent. Empty string is
            valid (represents "agent gave up / could not produce
            patch") and the grader treats it as a non-resolution.
        prefix: Optional run label (e.g. "ygn-sage-arm-d-smoke-001").

    Returns:
        A dict matching the Pro grader's expected schema.
    """
    record: ProPatchRecord = {
        "instance_id": instance_id,
        "patch": patch,
    }
    if prefix is not None:
        record["prefix"] = prefix

    validate_record(record)  # type: ignore[arg-type]
    return record


def write_predictions(
    records: list[ProPatchRecord],
    output_path: Path,
) -> None:
    """Write `records` as a JSON list at `output_path`.

    Per cgpro DESIGN E trap Q6 (LF-only framing): we write standard
    JSON (NOT JSONL), but the file is utf-8 with LF line endings.
    """
    if not isinstance(records, list):
        raise ValueError(f"records must be list, got {type(records).__name__}")
    for record in records:
        validate_record(record)  # type: ignore[arg-type]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(records, indent=2, ensure_ascii=False)
    # Force LF line endings even on Windows; the Pro grader runs in
    # Modal/Docker where LF is canonical.
    output_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    log.info("Wrote %d Pro records to %s", len(records), output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True, help="task identifier")
    parser.add_argument(
        "--patch-file",
        type=Path,
        required=True,
        help="path to unified diff (or empty file for canary)",
    )
    parser.add_argument("--prefix", default=None, help="optional run label")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output predictions.json (Pro grader input)",
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

    if not args.patch_file.is_file():
        raise SystemExit(f"--patch-file not found: {args.patch_file}")

    patch = args.patch_file.read_text(encoding="utf-8")
    record = format_patch(args.instance_id, patch, args.prefix)
    write_predictions([record], args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
