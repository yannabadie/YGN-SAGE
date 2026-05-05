#!/usr/bin/env python3
"""Convert sage.bench --type swebench output (Lite shape JSONL) to SWE-bench Pro shape JSON.

`sage.bench --type swebench --dataset pro` writes `predictions.jsonl`
in SWE-bench Lite/Verified shape:
    {instance_id, model_name_or_path, model_patch, ...metadata}

The SWE-bench Pro grader expects:
    [{instance_id, patch, prefix?}, ...]

This is the bridge. Renames `model_patch` -> `patch`, drops Lite-only
metadata, optionally injects a `prefix` label. Output validates via
`swebench_pro_format_patch.validate_record`.

Usage:
    python -m sage_python.scripts.swebench_pro_lite_to_pro_predictions \\
        --predictions-jsonl <bench_out_dir>/predictions.jsonl \\
        --output predictions.json \\
        --prefix ygn-sage-arm-d-reasoner-n1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("sage.bench.swebench_pro_lite_to_pro_predictions")

_FORMAT_PATCH_PATH = (
    Path(__file__).parent / "swebench_pro_format_patch.py"
).resolve()


def _load_format_patch_module():
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_format_patch", _FORMAT_PATCH_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("swebench_pro_format_patch", mod)
    spec.loader.exec_module(mod)
    return mod


def convert(predictions_jsonl: Path, output: Path, prefix: str | None) -> None:
    fmt = _load_format_patch_module()
    records = []

    if not predictions_jsonl.is_file():
        raise SystemExit(f"--predictions-jsonl not found: {predictions_jsonl}")

    with predictions_jsonl.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                lite = json.loads(line)
            except json.JSONDecodeError as exc:
                log.error("malformed JSONL at line %d: %s", line_no, exc)
                raise SystemExit(1) from exc
            iid = lite.get("instance_id")
            patch = lite.get("model_patch") or ""
            if not iid:
                log.warning("line %d: missing instance_id, skipping", line_no)
                continue
            record = fmt.format_patch(iid, patch, prefix=prefix)
            records.append(record)

    fmt.write_predictions(records, output)
    log.info(
        "Converted %d Lite-shape JSONL records -> %d Pro-shape JSON records "
        "(non-empty patches: %d)",
        line_no, len(records),
        sum(1 for r in records if r.get("patch")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        required=True,
        help="path to sage.bench's predictions.jsonl (Lite shape)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output Pro-shape predictions.json",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="optional prefix label to inject into every Pro record",
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

    convert(args.predictions_jsonl, args.output, args.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
