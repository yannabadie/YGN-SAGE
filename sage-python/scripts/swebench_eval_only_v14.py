"""V14 eval-only re-run on v13 predictions.

Purpose: prove whether the CRLF fix committed in efb8afd resolves the
v13 0/10 result by re-evaluating the exact same 10 predictions under
the new swebench_ca_patch.py. This isolates the CRLF fix — no new
generation, no minimax variance, same patches.

Usage:
    python sage-python/scripts/swebench_eval_only_v14.py

Writes the graded report JSON to docs/benchmarks/<timestamp>-swebench-v14-eval-report.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Module-level imports happen BEFORE any sage.bench import so the resource
# stub in swebench_bench.py is installed first (swebench's harness imports
# `resource` which is Unix-only).
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# NB: SAGE_SWEBENCH_ALLOW_INSECURE is explicitly NOT set here — the v14
# smoke runs under Directive #3 defaults (no SSL bypass in Dockerfile).
# The existing env images from v13 (which were built WITH the bypass at
# the time) are still on disk and will be reused; we aren't rebuilding
# base images, just running eval on cached ones.

# Must import swebench_bench BEFORE swebench.harness.* so the stub is in place.
from sage.bench.swebench_bench import SWEBenchBench  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("swebench-v14")

V13_PREDICTIONS = Path(
    "C:/Users/yann.abadie/AppData/Local/Temp/sage_swebench_kfhxbz7i/predictions.jsonl"
)


def main() -> int:
    if not V13_PREDICTIONS.is_file():
        log.error("V13 predictions not found at %s", V13_PREDICTIONS)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"sage-v14-crlf-{timestamp}"
    log.info("V14 eval-only run: run_id=%s", run_id)
    log.info("Using v13 predictions: %s", V13_PREDICTIONS)

    bench = SWEBenchBench(
        system=None,  # eval-only path doesn't touch self.system
        dataset="lite",
        eval_timeout=600,  # 10 min per instance (some astropy suites are long)
        max_workers=4,
        run_id=run_id,
    )

    results = bench.evaluate_with_harness(V13_PREDICTIONS)

    print("\n" + "=" * 70)
    print(f"V14 eval-only results (run_id={run_id}):")
    print("=" * 70)
    print(f"  Total: {results.get('total', 0)}")
    print(f"  Resolved: {results.get('resolved', 0)}")
    print(f"  Resolved rate: {results.get('resolved_rate', 0):.1%}")
    print(f"  Completed IDs: {results.get('completed_ids', [])}")
    print(f"  Resolved IDs: {results.get('resolved_ids', [])}")
    print(f"  Error IDs: {results.get('error_ids', [])}")
    if results.get("error"):
        print(f"  ERROR: {results['error']}")
    if results.get("report_path"):
        print(f"  Report: {results['report_path']}")

    # Persist to docs/benchmarks
    out_dir = Path(__file__).resolve().parents[2] / "docs" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}-swebench-v14-eval-report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Saved to: {out_path}")
    print("=" * 70)

    # Exit code mirrors success signal — non-zero if 0/10 persists.
    return 0 if results.get("resolved", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
