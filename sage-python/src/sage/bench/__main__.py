"""CLI entry point: ``python -m sage.bench``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from datetime import datetime, timezone
from typing import Any
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


def _save_report(report, bench, output: str | None, name: str) -> None:
    """Save JSON report + truth pack."""
    if output is None:
        repo = _repo_root()
        bench_dir = repo / "docs" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = str(bench_dir / f"{date_str}-{name}.json")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(report)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Report saved to: {out_path}")

    if hasattr(bench, "manifest") and bench.manifest and bench.manifest.traces:
        jsonl_path = out_path.with_suffix(".jsonl")
        jsonl_path.write_text(bench.manifest.to_jsonl(), encoding="utf-8")
        print(f"  Truth pack (JSONL): {jsonl_path}")

        summary_path = out_path.with_name(out_path.stem + "-summary.json")
        summary_path.write_text(
            json.dumps(bench.manifest.summary(), indent=2), encoding="utf-8"
        )
        print(f"  Truth pack (summary): {summary_path}")


def _print_report(report) -> None:
    """Print benchmark summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {report.benchmark}")
    print(f"  Timestamp: {report.timestamp}")
    print(f"{'=' * 60}")
    print(f"  Pass rate : {report.pass_rate:.1%} ({report.passed}/{report.total})")
    print(f"  Avg Latency: {report.avg_latency_ms:.1f}ms")
    print(f"  Avg Cost   : ${report.avg_cost_usd:.6f}/task")
    print(f"  Routing    : {report.routing_breakdown}")
    print(f"{'=' * 60}")

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


_BOOT_TIER = "fast"  # Set by main() from --tier flag


def _boot_system(tier: str | None = None):
    """Boot AgentSystem with real LLM (requires GOOGLE_API_KEY)."""
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier=tier or _BOOT_TIER, event_bus=bus)
    return system, bus


async def _run_humaneval(output: str | None, limit: int | None) -> None:
    from sage.bench.humaneval import HumanEvalBench

    if os.environ.get("GOOGLE_API_KEY"):
        system, bus = _boot_system()
        bench = HumanEvalBench(system=system, event_bus=bus)
    else:
        bench = HumanEvalBench()

    report = await bench.run(limit=limit)
    _print_report(report)
    _save_report(report, bench, output, "humaneval")


async def _run_routing(output: str | None) -> None:
    # Legacy self-consistency benchmark removed (measured heuristic vs itself).
    # Use routing_gt instead for real accuracy measurement.
    print("  routing benchmark removed — use --type routing_gt instead")
    return


async def _run_evalplus(
    output: str | None, limit: int | None, dataset: str, official: bool = False,
) -> None:
    from sage.bench.evalplus_bench import EvalPlusBench

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  ERROR: GOOGLE_API_KEY required for EvalPlus benchmark")
        return

    system, bus = _boot_system()
    bench = EvalPlusBench(system=system, event_bus=bus, dataset=dataset, official_mode=official)

    if official:
        results = await bench.run_official(limit=limit)
        print(f"\n  Official EvalPlus Results:")
        print(f"    Base pass@1: {results.get('base', 0):.1%}")
        print(f"    Plus pass@1: {results.get('plus', 0):.1%}")
    else:
        report = await bench.run(limit=limit)
        _print_report(report)
        _save_report(report, bench, output, f"evalplus-{dataset}")


async def _run_ablation(output: str | None, limit: int | None) -> None:
    from sage.bench.ablation import ABLATION_CONFIGS
    from sage.bench.bigcodebench_bench import BigCodeBenchBench

    all_results: dict[str, dict] = {}

    for config in ABLATION_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"  ABLATION: {config.label}")
        print(f"  memory={config.memory} avr={config.avr} "
              f"routing={config.routing} guardrails={config.guardrails}")
        print(f"{'#' * 60}")

        system, bus = _boot_system()

        if config.label == "baseline":
            # Disable pipeline entirely — bare LLM call via legacy path
            system.pipeline = None
        else:
            # Apply skip flags to agent_loop AND disable pipeline components
            config.apply(system)
            if not config.routing and system.pipeline:
                system.pipeline.router = None  # Skip classify stage
            if not config.memory:
                system.agent_loop.consolidator = None

        bench = BigCodeBenchBench(
            system=system, event_bus=bus, subset="hard", split="instruct",
        )

        report = await bench.run(limit=limit)
        _print_report(report)

        all_results[config.label] = {
            "config": dataclasses.asdict(config),
            "pass_rate": report.pass_rate,
            "passed": report.passed,
            "total": report.total,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_cost_usd": report.avg_cost_usd,
            "per_task": [r.passed for r in report.results],  # binary outcomes for stats
        }

    # Print ablation comparison table
    print(f"\n{'=' * 60}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Config':<16} {'Pass Rate':>10} {'Passed':>8} {'Total':>8} {'Delta':>8}")
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    full_rate = all_results.get("full", {}).get("pass_rate", 0.0)
    for label, data in all_results.items():
        rate = data["pass_rate"]
        delta = rate - full_rate
        delta_str = f"{delta:+.1%}" if label != "full" else "ref"
        print(f"  {label:<16} {rate:>9.1%} {data['passed']:>8} {data['total']:>8} {delta_str:>8}")
    print()

    # Statistical tests (McNemar + Cohen's d + Bootstrap CI)
    from sage.bench.ablation import compute_ablation_stats
    binary_results = {label: data["per_task"] for label, data in all_results.items() if data.get("per_task")}
    if len(binary_results) >= 2 and all(len(v) == len(next(iter(binary_results.values()))) for v in binary_results.values()):
        stats = compute_ablation_stats(binary_results)
        all_results["_statistics"] = stats
        print("  STATISTICAL TESTS (McNemar + Cohen's d)")
        print(f"  {'-'*50}")
        for pair, s in stats.get("pairwise", {}).items():
            sig = "***" if s["mcnemar_p"] < 0.05 else "n.s."
            print(f"  {pair:<30} p={s['mcnemar_p']:.4f} {sig}  d={s['cohens_d']:+.3f}  CI={s['bootstrap_ci_95']}")
        print()

    # Save combined results
    if output is None:
        repo = _repo_root()
        bench_dir = repo / "docs" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = str(bench_dir / f"{date_str}-ablation-study.json")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"  Ablation report saved to: {out_path}")


async def _run_swebench(args) -> None:
    from sage.bench.swebench_bench import SWEBenchBench, evaluate_predictions, dataset_info

    # Dataset selection: default to "lite" if humaneval/mbpp selected
    swe_dataset = args.dataset if args.dataset in ("lite", "verified") else "lite"

    # Info mode
    if args.swebench_info:
        info = dataset_info(swe_dataset)
        print(f"\n{'=' * 60}")
        print(f"  SWE-Bench Dataset: {info['hf_name']}")
        print(f"{'=' * 60}")
        print(f"  Total instances: {info['total_instances']}")
        print(f"  Repositories: {info['repo_count']}")
        if info.get("difficulties"):
            print(f"  Difficulties: {info['difficulties']}")
        print(f"\n  Top repos:")
        for repo, count in list(info['repos'].items())[:15]:
            print(f"    {repo}: {count}")
        print()
        return

    # Evaluate pre-generated predictions
    if args.eval_predictions:
        print(f"\n  Evaluating pre-generated predictions: {args.eval_predictions}")
        results = evaluate_predictions(
            predictions_path=args.eval_predictions,
            dataset=swe_dataset,
            timeout=args.eval_timeout,
            max_workers=args.max_workers,
        )
        print(f"\n{'=' * 60}")
        print(f"  SWE-Bench Evaluation Results")
        print(f"{'=' * 60}")
        print(f"  Resolved: {results.get('resolved', 0)}/{results.get('total', 0)} "
              f"({results.get('resolved_rate', 0):.1%})")
        if results.get("error"):
            print(f"  Error: {results['error']}")
        if results.get("report_path"):
            print(f"  Report: {results['report_path']}")
        print()
        return

    try:
        system, bus = _boot_system()
    except RuntimeError as exc:
        print(f"  ERROR: {exc}")
        return
    bench = SWEBenchBench(
        system=system,
        event_bus=bus,
        dataset=swe_dataset,
        eval_timeout=args.eval_timeout,
        max_workers=args.max_workers,
    )

    # offset defaults to 0 and is optional on argparse, but callers
    # constructing args via SimpleNamespace may omit it — handle both.
    _offset = int(getattr(args, "offset", 0) or 0)
    if args.generate_only:
        # Generate patches only (no Docker evaluation)
        preds_path = await bench.run_generate_only(limit=args.limit, offset=_offset)
        print(f"  Predictions saved to: {preds_path}")
    else:
        # Full pipeline: generate + evaluate
        report = await bench.run(limit=args.limit, offset=_offset)
        _print_report(report)
        _save_report(report, bench, args.output, f"swebench-{swe_dataset}")


async def _run_bigcodebench(output: str | None, limit: int | None, subset: str, split: str) -> None:
    from sage.bench.bigcodebench_bench import BigCodeBenchBench

    if os.environ.get("GOOGLE_API_KEY"):
        system, bus = _boot_system()
        bench = BigCodeBenchBench(system=system, event_bus=bus, subset=subset, split=split)
    else:
        bench = BigCodeBenchBench(subset=subset, split=split)

    report = await bench.run(limit=limit)
    _print_report(report)
    _save_report(report, bench, output, f"bigcodebench-{subset}-{split}")

    # Write predictions JSONL for official submission
    from datetime import datetime
    pred_path = f"docs/benchmarks/{datetime.now():%Y-%m-%d}-predictions-{subset}-{split}.jsonl"
    bench.write_predictions(pred_path)
    print(f"  Predictions: {pred_path}")


async def _run_apps(
    output: str | None, limit: int | None, difficulty: str | None,
) -> None:
    from sage.bench.apps_bench import APPSBench

    if os.environ.get("GOOGLE_API_KEY"):
        system, bus = _boot_system()
        bench = APPSBench(system=system, event_bus=bus, difficulty=difficulty)
    else:
        bench = APPSBench(difficulty=difficulty)

    report = await bench.run(limit=limit)
    _print_report(report)
    diff_label = difficulty or "all"
    _save_report(report, bench, output, f"apps-{diff_label}")


async def _run_livecodebench(output: str | None, limit: int | None) -> None:
    from sage.bench.livecodebench_bench import LiveCodeBenchBench

    if os.environ.get("GOOGLE_API_KEY"):
        system, bus = _boot_system()
        bench = LiveCodeBenchBench(system=system, event_bus=bus)
    else:
        bench = LiveCodeBenchBench()

    report = await bench.run(limit=limit)
    _print_report(report)
    _save_report(report, bench, output, "livecodebench")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.bench",
        description="YGN-SAGE benchmark pipeline",
    )
    parser.add_argument(
        "--type",
        choices=["routing", "humaneval", "evalplus", "ablation", "routing_gt", "memory_ablation", "memory_coherence", "evolution_ablation", "swebench", "heterogeneous", "gaia", "bigcodebench", "apps", "livecodebench", "all"],
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
        help="Limit the number of problems",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N problems (SWE-Bench only). Useful to avoid "
             "tasks the model has memorized (e.g. first 3 astropy tasks).",
    )
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp", "lite", "verified"],
        default="humaneval",
        help="Dataset: humaneval/mbpp for EvalPlus, lite/verified for SWE-Bench",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        default=False,
        help="Use official EvalPlus CLI evaluation (comparable to leaderboard)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        default=False,
        help="SWE-Bench: generate patches only, skip Docker evaluation",
    )
    parser.add_argument(
        "--eval-predictions",
        type=str,
        default=None,
        help="SWE-Bench: evaluate a pre-generated predictions JSONL file",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="SWE-Bench: parallel Docker evaluation workers (default: 4)",
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=300,
        help="SWE-Bench: timeout per Docker evaluation in seconds (default: 300)",
    )
    parser.add_argument(
        "--swebench-info",
        action="store_true",
        default=False,
        help="SWE-Bench: print dataset info and exit",
    )
    parser.add_argument(
        "--subset",
        choices=["full", "hard"],
        default="full",
        help="BigCodeBench subset: full (1140) or hard (~150)",
    )
    parser.add_argument(
        "--split",
        choices=["instruct", "complete"],
        default="instruct",
        help="BigCodeBench split: instruct (NL) or complete (docstring)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["introductory", "interview", "competition"],
        default=None,
        help="APPS difficulty filter (default: all levels)",
    )
    parser.add_argument(
        "--tier",
        choices=["fast", "mutator", "reasoner", "codex", "codex_max", "budget", "fallback", "auto"],
        default="fast",
        help="LLM tier for default provider (default: fast). 'reasoner' uses gemini-3.1-pro-preview.",
    )
    args = parser.parse_args()

    _load_env()

    # Configure logging so bench progress is visible
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=__import__("sys").stderr,
    )

    # Set boot tier for all benchmark runs
    global _BOOT_TIER
    _BOOT_TIER = args.tier

    if args.type in ("routing", "all"):
        asyncio.run(_run_routing(args.output))

    if args.type in ("humaneval", "all"):
        asyncio.run(_run_humaneval(args.output, args.limit))

    if args.type == "evalplus":
        asyncio.run(_run_evalplus(args.output, args.limit, args.dataset, args.official))

    if args.type == "ablation":
        asyncio.run(_run_ablation(args.output, args.limit))

    if args.type == "routing_gt":
        from sage.bench.routing_ground_truth import run_routing_gt

        def _run_gt_with(name, router):
            print(f"\n{'=' * 60}")
            print(f"  Routing GT: {name}")
            print(f"{'=' * 60}")
            result = run_routing_gt(router, verbose=True)
            print(f"\n  Accuracy: {result.accuracy:.1%} ({result.correct}/{result.total})")
            print(f"  Elapsed: {result.elapsed_ms:.0f}ms")
            for sys, stats in sorted(result.per_system.items()):
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                print(f"    S{sys}: {acc:.0%} ({stats['correct']}/{stats['total']})")
            if result.misroutes:
                print(f"\n  Misroutes ({len(result.misroutes)}):")
                for m in result.misroutes:
                    print(f"    [{m['id']}] expected=S{m['expected']} got=S{m['actual']}: {m['task']}")
            return result

        # AdaptiveRouter + kNN (primary — learned routing)
        try:
            from sage.strategy.adaptive_router import AdaptiveRouter
            from sage.strategy.knn_router import KnnRouter
            knn = KnnRouter()
            if not knn.is_ready:
                knn.build_from_ground_truth()
            if knn.is_ready:
                ar = AdaptiveRouter(knn_router=knn)
                _run_gt_with(f"AdaptiveRouter + kNN ({knn.exemplar_count} exemplars, {knn.embedder_backend})", ar)
            else:
                print("\n  kNN router: not available (no semantic embedder)")
        except Exception as e:
            print(f"\n  kNN router failed: {e}")

    if args.type == "swebench":
        asyncio.run(_run_swebench(args))

    if args.type == "heterogeneous":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("  ERROR: GOOGLE_API_KEY required for heterogeneous benchmark")
        else:
            from sage.bench.heterogeneous_bench import HeterogeneousBench
            system, bus = _boot_system()
            bench = HeterogeneousBench(system=system)
            report = asyncio.run(bench.run(limit=args.limit))
            _print_report(report)
            _save_report(report, bench, args.output, "heterogeneous")

    if args.type == "gaia":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("  ERROR: GOOGLE_API_KEY required for GAIA benchmark")
        else:
            from sage.bench.gaia_bench import GaiaBench
            system, bus = _boot_system()
            bench: Any = GaiaBench(system=system)
            report = asyncio.run(bench.run(limit=args.limit))
            _print_report(report)
            _save_report(report, bench, args.output, "gaia")

    if args.type == "bigcodebench":
        asyncio.run(_run_bigcodebench(args.output, args.limit, args.subset, args.split))

    if args.type == "apps":
        asyncio.run(_run_apps(args.output, args.limit, args.difficulty))

    if args.type == "livecodebench":
        asyncio.run(_run_livecodebench(args.output, args.limit))

    if args.type == "memory_ablation":
        print("Memory Ablation requires full boot. Run: python -m sage.bench.memory_ablation")

    if args.type == "memory_coherence":
        from sage.bench.memory_coherence import run_memory_coherence, _default_boot
        from datetime import datetime, timezone

        report = asyncio.run(run_memory_coherence(_default_boot, limit=args.limit))
        print(f"\n{'=' * 60}")
        print("  Benchmark: memory_coherence")
        print(f"{'=' * 60}")
        print(f"  Pairs run          : {report.total}")
        print(f"  Cold   pass@0.7    : {report.cold_pass}/{report.total}")
        print(f"  Primed pass@0.7    : {report.primed_pass}/{report.total}")
        print(
            f"  Avg quality        : cold={report.avg_cold_quality:.3f}  "
            f"primed={report.avg_primed_quality:.3f}  Δ={report.quality_gain:+.3f}"
        )
        print(
            f"  Avg latency (ms)   : cold={report.avg_cold_latency_ms:.0f}  "
            f"primed={report.avg_primed_latency_ms:.0f}  Δ={report.latency_gain_ms:+.0f}"
        )
        print(f"{'=' * 60}\n")
        if args.output is None:
            repo = _repo_root()
            bench_dir = repo / "docs" / "benchmarks"
            bench_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            args.output = str(bench_dir / f"{date_str}-memory_coherence.json")
        data = dataclasses.asdict(report)
        Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Report saved to: {args.output}")

    if args.type == "evolution_ablation":
        print("Evolution Ablation requires full boot. Run: python -m sage.bench.evolution_ablation")


if __name__ == "__main__":
    main()
