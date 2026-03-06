"""CLI entry point: ``python -m sage.bench``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from this file to find the repository root (contains .git/)."""
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / ".git").is_dir():
            return parent
    # Fallback: 4 levels up from sage-python/src/sage/bench/__main__.py
    return here.parents[3]


async def _run_routing(output: str | None) -> None:
    from sage.strategy.metacognition import MetacognitiveController
    from sage.bench.routing import RoutingAccuracyBench

    mc = MetacognitiveController()
    bench = RoutingAccuracyBench(metacognition=mc)
    report = await bench.run()

    # --- Print summary to stdout ---
    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {report.benchmark}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"{'=' * 60}")
    print(f"  Total tasks : {report.total}")
    print(f"  Passed      : {report.passed}")
    print(f"  Failed      : {report.failed}")
    print(f"  Errors      : {report.errors}")
    print(f"  Pass rate   : {report.pass_rate:.1%}")
    print(f"  Avg latency : {report.avg_latency_ms:.3f} ms")
    print(f"  Avg cost    : ${report.avg_cost_usd:.6f}")
    print(f"  Routing     : {report.routing_breakdown}")
    print(f"{'=' * 60}")

    # Show failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f.task_id}: {f.error}")
    else:
        print("\n  All tasks passed.")
    print()

    # --- Save JSON report ---
    if output is None:
        repo = _repo_root()
        bench_dir = repo / "docs" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = str(bench_dir / f"{date_str}-routing.json")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(report)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Report saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.bench",
        description="YGN-SAGE benchmark pipeline",
    )
    parser.add_argument(
        "--type",
        choices=["routing"],
        default="routing",
        help="Benchmark type to run (default: routing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for the JSON report",
    )
    args = parser.parse_args()

    if args.type == "routing":
        asyncio.run(_run_routing(args.output))
    else:
        print(f"Unknown benchmark type: {args.type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
