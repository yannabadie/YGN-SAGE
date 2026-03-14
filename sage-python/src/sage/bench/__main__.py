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


def _boot_system():
    """Boot AgentSystem with real LLM (requires GOOGLE_API_KEY)."""
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
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
    from sage.strategy.metacognition import MetacognitiveController
    from sage.bench.routing import RoutingAccuracyBench

    mc = MetacognitiveController()
    bench = RoutingAccuracyBench(metacognition=mc)
    report = await bench.run()
    _print_report(report)
    _save_report(report, bench, output, "routing")


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
    from sage.bench.evalplus_bench import EvalPlusBench

    if not os.environ.get("GOOGLE_API_KEY"):
        print("  ERROR: GOOGLE_API_KEY required for ablation study")
        return

    all_results: dict[str, dict] = {}

    for config in ABLATION_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"  ABLATION: {config.label}")
        print(f"  memory={config.memory} avr={config.avr} "
              f"routing={config.routing} guardrails={config.guardrails}")
        print(f"{'#' * 60}")

        system, bus = _boot_system()

        if config.label == "baseline":
            bench = EvalPlusBench(
                system=system, event_bus=bus, dataset="humaneval",
                baseline_mode=True,
            )
        else:
            config.apply(system)
            bench = EvalPlusBench(
                system=system, event_bus=bus, dataset="humaneval",
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

    # Full run requires API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("  ERROR: GOOGLE_API_KEY required for SWE-Bench benchmark")
        return

    system, bus = _boot_system()
    bench = SWEBenchBench(
        system=system,
        event_bus=bus,
        dataset=swe_dataset,
        eval_timeout=args.eval_timeout,
        max_workers=args.max_workers,
    )

    if args.generate_only:
        # Generate patches only (no Docker evaluation)
        preds_path = await bench.run_generate_only(limit=args.limit)
        print(f"  Predictions saved to: {preds_path}")
    else:
        # Full pipeline: generate + evaluate
        report = await bench.run(limit=args.limit)
        _print_report(report)
        _save_report(report, bench, args.output, f"swebench-{swe_dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sage.bench",
        description="YGN-SAGE benchmark pipeline",
    )
    parser.add_argument(
        "--type",
        choices=["routing", "humaneval", "evalplus", "ablation", "routing_gt", "memory_ablation", "evolution_ablation", "swebench", "heterogeneous", "all"],
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
    args = parser.parse_args()

    _load_env()

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

        # 1. ComplexityRouter (heuristic baseline)
        from sage.strategy.metacognition import ComplexityRouter
        _run_gt_with("ComplexityRouter (heuristic)", ComplexityRouter())

        # 2. AdaptiveRouter + kNN (if embedder available)
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

    if args.type == "memory_ablation":
        print("Memory Ablation requires full boot. Run: python -m sage.bench.memory_ablation")

    if args.type == "evolution_ablation":
        print("Evolution Ablation requires full boot. Run: python -m sage.bench.evolution_ablation")


if __name__ == "__main__":
    main()
