#!/usr/bin/env python3
"""Collect (task, response, quality_score) training triples for DistilBERT QualityEstimator.

Runs EvalPlus benchmarks (HumanEval+ / MBPP+) through the full SAGE framework,
captures model responses, scores them with both the heuristic QualityEstimator
and EvalPlus ground-truth evaluation, and writes JSONL training data suitable
for fine-tuning a learned quality estimator (DistilBERT).

Ground-truth quality scoring:
    1.0  — plus_passed  (solution passes all 80x-harder EvalPlus+ tests)
    0.5  — base_passed  (passes original tests only)
    0.0  — failed       (does not pass base tests)

Usage::

    # Default: HumanEval+ (164 tasks), output to data/quality_triples.jsonl
    python scripts/collect_quality_triples.py

    # MBPP+ dataset, limited to 20 tasks
    python scripts/collect_quality_triples.py --dataset mbpp --limit 20

    # Custom output path
    python scripts/collect_quality_triples.py --output my_triples.jsonl

Requires:
    - GOOGLE_API_KEY (or Codex CLI) for LLM generation
    - evalplus package (pip install evalplus) for dataset loading
    - sage-python installed in dev mode (pip install -e ".[all,dev]")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# SSL bypass for corporate proxy (must be set before any imports that do HTTPS)
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("quality_triples")


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


def _load_dataset(dataset: str) -> dict:
    """Load an EvalPlus dataset by name.

    Returns dict keyed by task_id, e.g. {"HumanEval/0": {...}, ...}.
    """
    loaders = {
        "humaneval": "evalplus.data:get_human_eval_plus",
        "mbpp": "evalplus.data:get_mbpp_plus",
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {list(loaders.keys())}")

    import importlib
    module_path, func_name = loaders[dataset].split(":")
    mod = importlib.import_module(module_path)
    loader = getattr(mod, func_name)
    return loader()


async def collect_triples(
    dataset: str,
    limit: int | None,
    output_path: str,
) -> None:
    """Main collection loop: generate, score, evaluate, save."""

    # 1. Boot the SAGE system
    log.info("Booting SAGE agent system...")
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus

    bus = EventBus()
    try:
        system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
    except Exception as exc:
        log.error("Failed to boot agent system: %s", exc)
        log.error("Ensure GOOGLE_API_KEY is set or Codex CLI is installed.")
        sys.exit(1)

    log.info("Agent system ready.")

    # 2. Load EvalPlus dataset
    log.info("Loading EvalPlus dataset: %s", dataset)
    try:
        problems = _load_dataset(dataset)
    except Exception as exc:
        log.error("Failed to load dataset: %s", exc)
        log.error("Install evalplus: pip install evalplus")
        sys.exit(1)

    task_ids = list(problems.keys())
    if limit is not None:
        task_ids = task_ids[:limit]

    log.info("Loaded %d tasks (limit=%s)", len(task_ids), limit or "none")

    # 3. Create evaluator for ground-truth scoring
    from sage.bench.evalplus_bench import EvalPlusBench
    from sage.bench.humaneval import extract_code
    from sage.quality_estimator import QualityEstimator

    evaluator = EvalPlusBench(system=system, event_bus=bus, dataset=dataset)

    # 4. Prepare output file
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_path.resolve())

    # Counters for summary
    total_tasks = len(task_ids)
    completed = 0
    skipped = 0
    base_passed_count = 0
    plus_passed_count = 0
    failed_count = 0
    heuristic_scores: list[float] = []
    gt_scores: list[float] = []

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, task_id in enumerate(task_ids):
            problem = problems[task_id]
            prompt = problem["prompt"]
            entry_point = problem["entry_point"]

            # Build task prompt (same as EvalPlusBench.generate_solutions)
            task_prompt = (
                "Complete this Python function. "
                "Return ONLY the complete function, no explanation.\n\n"
                f"```python\n{prompt}\n```"
            )

            t0 = time.perf_counter()
            response = ""
            system_used = 0
            had_errors = False
            avr_iterations = 0

            # --- Step A: Generate response via SAGE ---
            try:
                response = await asyncio.wait_for(
                    system.run(task_prompt),
                    timeout=60.0,
                )
                system_used = (
                    getattr(system.agent_loop, "_last_routing_system", 0) or 2
                )
                had_errors = bool(getattr(system.agent_loop, "_last_error", None))
                avr_iterations = getattr(system.agent_loop, "_last_avr_iterations", 0)
            except asyncio.TimeoutError:
                log.warning("[%d/%d] %s: generation timed out (60s)", i + 1, total_tasks, task_id)
                had_errors = True
            except Exception as exc:
                log.warning("[%d/%d] %s: generation failed: %s", i + 1, total_tasks, task_id, str(exc)[:200])
                had_errors = True

            latency_ms = (time.perf_counter() - t0) * 1000

            # --- Step B: Heuristic quality score ---
            heuristic_score = QualityEstimator.estimate(
                task=task_prompt,
                result=response,
                latency_ms=latency_ms,
                had_errors=had_errors,
                avr_iterations=avr_iterations,
            )

            # --- Step C: Ground-truth evaluation via EvalPlus ---
            base_passed = False
            plus_passed = False
            eval_error = ""

            if response.strip():
                try:
                    solution_code = extract_code(response, entry_point)
                    eval_result = evaluator.evaluate_task(
                        solution_code, problem, timeout=15.0,
                    )
                    base_passed = eval_result["base_passed"]
                    plus_passed = eval_result["plus_passed"]
                    eval_error = eval_result.get("error", "")
                except Exception as exc:
                    eval_error = str(exc)[:200]
                    log.warning("[%d/%d] %s: evaluation failed: %s", i + 1, total_tasks, task_id, eval_error)
            else:
                eval_error = "empty_response"

            # --- Step D: Compute ground-truth quality ---
            if plus_passed:
                ground_truth_score = 1.0
                plus_passed_count += 1
            elif base_passed:
                ground_truth_score = 0.5
                base_passed_count += 1
            else:
                ground_truth_score = 0.0
                failed_count += 1

            # --- Step E: Write JSONL triple ---
            triple = {
                "task_id": task_id,
                "task": task_prompt,
                "response": response,
                "heuristic_score": round(heuristic_score, 4),
                "ground_truth_score": ground_truth_score,
                "base_passed": base_passed,
                "plus_passed": plus_passed,
                "latency_ms": round(latency_ms, 1),
                "system_used": system_used,
            }
            fout.write(json.dumps(triple, ensure_ascii=False) + "\n")
            fout.flush()

            completed += 1
            heuristic_scores.append(heuristic_score)
            gt_scores.append(ground_truth_score)

            # Status indicator
            if plus_passed:
                status = "PLUS_PASS"
            elif base_passed:
                status = "BASE_ONLY"
            else:
                status = "FAIL"

            # Progress every 10 tasks or on first/last
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == total_tasks:
                current_plus_rate = plus_passed_count / completed if completed > 0 else 0
                current_base_rate = (plus_passed_count + base_passed_count) / completed if completed > 0 else 0
                log.info(
                    "[%d/%d] %s: %s (heuristic=%.2f, gt=%.1f) | "
                    "Running: plus=%.1f%% base=%.1f%% (%.0fms)",
                    i + 1, total_tasks, task_id, status,
                    heuristic_score, ground_truth_score,
                    current_plus_rate * 100, current_base_rate * 100,
                    latency_ms,
                )
            else:
                print(
                    f"  [{i + 1}/{total_tasks}] {task_id}: {status} "
                    f"(h={heuristic_score:.2f}, gt={ground_truth_score:.1f}, "
                    f"{latency_ms:.0f}ms)",
                    flush=True,
                )

    # 5. Final summary
    _print_summary(
        total_tasks=total_tasks,
        completed=completed,
        skipped=skipped,
        plus_passed=plus_passed_count,
        base_only=base_passed_count,
        failed=failed_count,
        heuristic_scores=heuristic_scores,
        gt_scores=gt_scores,
        output_path=out_path,
    )


def _print_summary(
    total_tasks: int,
    completed: int,
    skipped: int,
    plus_passed: int,
    base_only: int,
    failed: int,
    heuristic_scores: list[float],
    gt_scores: list[float],
    output_path: Path,
) -> None:
    """Print collection summary with score correlation."""
    print(f"\n{'=' * 60}")
    print("  Quality Triple Collection Summary")
    print(f"{'=' * 60}")
    print(f"  Total tasks:     {total_tasks}")
    print(f"  Completed:       {completed}")
    if skipped > 0:
        print(f"  Skipped:         {skipped}")
    print()
    print(f"  Plus passed:     {plus_passed} ({plus_passed / completed * 100:.1f}%)" if completed > 0 else "  Plus passed: 0")
    print(f"  Base only:       {base_only} ({base_only / completed * 100:.1f}%)" if completed > 0 else "  Base only: 0")
    print(f"  Failed:          {failed} ({failed / completed * 100:.1f}%)" if completed > 0 else "  Failed: 0")
    print()

    if len(heuristic_scores) >= 2 and len(gt_scores) >= 2:
        # Pearson correlation between heuristic and ground truth
        h_mean = sum(heuristic_scores) / len(heuristic_scores)
        g_mean = sum(gt_scores) / len(gt_scores)

        cov = sum(
            (h - h_mean) * (g - g_mean)
            for h, g in zip(heuristic_scores, gt_scores)
        ) / len(heuristic_scores)

        h_std = (sum((h - h_mean) ** 2 for h in heuristic_scores) / len(heuristic_scores)) ** 0.5
        g_std = (sum((g - g_mean) ** 2 for g in gt_scores) / len(gt_scores)) ** 0.5

        if h_std > 0 and g_std > 0:
            correlation = cov / (h_std * g_std)
            print(f"  Heuristic-GT correlation: {correlation:.3f} (Pearson r)")
        else:
            print("  Heuristic-GT correlation: N/A (zero variance)")

        print(f"  Heuristic mean:  {h_mean:.3f}")
        print(f"  Ground truth mean: {g_mean:.3f}")
    else:
        print("  Heuristic-GT correlation: N/A (too few samples)")

    print()
    print(f"  Output: {output_path.resolve()}")
    print(f"  Lines:  {completed} JSONL triples")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect (task, response, quality_score) triples for DistilBERT "
            "QualityEstimator training. Runs EvalPlus benchmarks through SAGE "
            "and evaluates correctness as ground truth."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp"],
        default="humaneval",
        help="EvalPlus dataset to use (default: humaneval)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/quality_triples.jsonl",
        help="Output JSONL path (default: data/quality_triples.jsonl)",
    )
    args = parser.parse_args()

    _load_env()

    if not os.environ.get("GOOGLE_API_KEY") and not _has_codex():
        print("ERROR: GOOGLE_API_KEY or Codex CLI required for LLM generation.")
        print("Set GOOGLE_API_KEY in your environment or .env file.")
        sys.exit(1)

    asyncio.run(collect_triples(
        dataset=args.dataset,
        limit=args.limit,
        output_path=args.output,
    ))


def _has_codex() -> bool:
    """Check if Codex CLI is installed."""
    import shutil
    return shutil.which("codex") is not None


if __name__ == "__main__":
    main()
