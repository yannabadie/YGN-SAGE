"""CLI entry point: ``python -m sage.bench``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
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


def _load_env() -> None:
    """Load .env file for API keys (same logic as boot.py)."""
    try:
        from dotenv import load_dotenv
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass


async def _run_humaneval(output: str | None, limit: int | None) -> None:
    from sage.bench.humaneval import HumanEvalBench

    # For real benchmark, need real LLM
    if os.environ.get("GOOGLE_API_KEY"):
        from sage.boot import boot_agent_system
        from sage.events.bus import EventBus

        bus = EventBus()
        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
        bench = HumanEvalBench(system=system, event_bus=bus)
    else:
        bench = HumanEvalBench()  # No system = direct mode

    report = await bench.run(limit=limit)

    # --- Print summary to stdout ---
    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {report.benchmark}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"{'=' * 60}")
    print(f"  HumanEval pass@1: {report.pass_rate:.1%} ({report.passed}/{report.total})")
    print(f"  Avg Latency     : {report.avg_latency_ms:.1f}ms")
    print(f"  Avg Cost        : ${report.avg_cost_usd:.6f}/task")
    print(f"  Routing         : {report.routing_breakdown}")
    print(f"{'=' * 60}")

    # Show failures summary (limit to 10 to avoid spam)
    failures = [r for r in report.results if not r.passed]
    if failures:
        shown = failures[:10]
        print(f"\n  Failures ({len(failures)} total, showing first {len(shown)}):")
        for f in shown:
            err_short = f.error[:80] if f.error else "no output"
            print(f"    {f.task_id}: {err_short}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    else:
        print("\n  All tasks passed.")
    print()

    # --- Save JSON report ---
    if output is None:
        repo = _repo_root()
        bench_dir = repo / "docs" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = str(bench_dir / f"{date_str}-humaneval.json")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(report)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Report saved to: {out_path}")

    # Save truth pack (JSONL per-task traces + summary)
    if bench.manifest and bench.manifest.traces:
        jsonl_path = out_path.with_suffix(".jsonl")
        jsonl_path.write_text(bench.manifest.to_jsonl(), encoding="utf-8")
        print(f"  Truth pack (JSONL): {jsonl_path}")

        summary_path = out_path.with_name(out_path.stem + "-summary.json")
        summary_path.write_text(
            json.dumps(bench.manifest.summary(), indent=2), encoding="utf-8"
        )
        print(f"  Truth pack (summary): {summary_path}")


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

    # Save truth pack (JSONL per-task traces + summary)
    if hasattr(bench, 'manifest') and bench.manifest and bench.manifest.traces:
        jsonl_path = out_path.with_suffix(".jsonl")
        jsonl_path.write_text(bench.manifest.to_jsonl(), encoding="utf-8")
        print(f"  Truth pack (JSONL): {jsonl_path}")

        summary_path = out_path.with_name(out_path.stem + "-summary.json")
        summary_path.write_text(
            json.dumps(bench.manifest.summary(), indent=2), encoding="utf-8"
        )
        print(f"  Truth pack (summary): {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.bench",
        description="YGN-SAGE benchmark pipeline",
    )
    parser.add_argument(
        "--type",
        choices=["routing", "humaneval", "all"],
        default="routing",
        help="Benchmark type to run (default: routing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for the JSON report",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of problems (humaneval only)",
    )
    args = parser.parse_args()

    _load_env()

    if args.type == "routing" or args.type == "all":
        asyncio.run(_run_routing(args.output))

    if args.type in ("humaneval", "all"):
        asyncio.run(_run_humaneval(args.output, args.limit))


if __name__ == "__main__":
    main()
