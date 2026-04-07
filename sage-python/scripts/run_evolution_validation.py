#!/usr/bin/env python3
"""Run evolution validation: compare baseline (template-only) vs evolved (6-path) topologies.

Uses BigCodeBench Hard tasks with executable tests for binary scoring (pass=1, fail=0).
Applies Wilcoxon signed-rank test + Cohen's d to determine if evolution helps.

Usage:
    python scripts/run_evolution_validation.py [--n-runs 20] [--tier budget]

Requirements:
    - API keys set (GOOGLE_API_KEY, DEEPSEEK_API_KEY, etc.)
    - bigcodebench package installed
    - scipy, numpy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
log = logging.getLogger("evolution_validation")


async def _run_task(system, task_text: str, entry_point: str, test_code: str,
                    timeout: float = 120.0) -> tuple[float, str]:
    """Run a single task through the pipeline and evaluate.

    Returns (score, solution) where score is 1.0 (pass) or 0.0 (fail).
    """
    from sage.bench.humaneval import extract_code
    from sage.bench.bigcodebench_bench import BigCodeBenchBench

    try:
        raw = await asyncio.wait_for(system.run(task_text), timeout=timeout)
        solution = extract_code(raw, entry_point)
    except Exception as exc:
        log.debug("  Generation error: %s", str(exc)[:100])
        return 0.0, ""

    if not solution.strip():
        return 0.0, ""

    passed, _ = BigCodeBenchBench._evaluate_solution_with_stderr(
        solution=solution,
        test_code=test_code,
        entry_point=entry_point,
        task_id="validation",
        timeout=30.0,
    )
    return 1.0 if passed else 0.0, solution


async def run_validation(n_runs: int = 20, tier: str = "budget") -> dict:
    """Run the full evolution validation on BigCodeBench Hard tasks.

    Phase 1: Baseline — template-only (topology_engine=None)
    Phase 2: Evolved — full 6-path topology engine
    Phase 3: Wilcoxon + Cohen's d gate
    """
    from sage.evolution.evaluator import validate_evolution

    # Load BigCodeBench Hard
    log.info("Loading BigCodeBench Hard dataset...")
    from bigcodebench.data import get_bigcodebench
    problems = get_bigcodebench(subset="hard")
    task_ids = list(problems.keys())[:n_runs]
    log.info("Selected %d tasks for paired comparison", len(task_ids))

    # Prepare prompts
    tasks = []
    for tid in task_ids:
        t = problems[tid]
        prompt = t.get("instruct_prompt", "")
        code_prompt = t.get("code_prompt", "")
        if code_prompt:
            prompt = f"Use this function signature and imports:\n```python\n{code_prompt}\n```\n\n{prompt}"
        tasks.append({
            "id": tid,
            "prompt": prompt,
            "entry_point": t["entry_point"],
            "test": t["test"],
        })

    # ── Phase 1: Baseline (template-only) ──────────────────────────────
    log.info("=" * 60)
    log.info("Phase 1: BASELINE (template-only, no topology engine)")
    log.info("=" * 60)

    from sage.boot import boot_agent_system
    system_baseline = boot_agent_system(use_mock_llm=False, llm_tier=tier)
    system_baseline.topology_engine = None  # Disable 6-path generation

    baseline_scores: list[float] = []
    for i, task in enumerate(tasks):
        t0 = time.monotonic()
        score, _ = await _run_task(
            system_baseline, task["prompt"], task["entry_point"], task["test"],
        )
        elapsed = time.monotonic() - t0
        baseline_scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        log.info("  [%d/%d] %s %s (%.1fs)", i + 1, len(tasks), status, task["id"], elapsed)

    log.info("  Baseline pass rate: %.1f%% (%d/%d)",
             100 * sum(baseline_scores) / len(baseline_scores),
             int(sum(baseline_scores)), len(baseline_scores))

    # ── Phase 2: Evolved (6-path topology engine) ──────────────────────
    log.info("=" * 60)
    log.info("Phase 2: EVOLVED (6-path topology engine active)")
    log.info("=" * 60)

    system_evolved = boot_agent_system(use_mock_llm=False, llm_tier=tier)
    # topology_engine is active by default when Rust is available

    evolved_scores: list[float] = []
    for i, task in enumerate(tasks):
        t0 = time.monotonic()
        score, _ = await _run_task(
            system_evolved, task["prompt"], task["entry_point"], task["test"],
        )
        elapsed = time.monotonic() - t0
        evolved_scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        log.info("  [%d/%d] %s %s (%.1fs)", i + 1, len(tasks), status, task["id"], elapsed)

    log.info("  Evolved pass rate: %.1f%% (%d/%d)",
             100 * sum(evolved_scores) / len(evolved_scores),
             int(sum(evolved_scores)), len(evolved_scores))

    # ── Phase 3: Statistical validation ────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 3: STATISTICAL VALIDATION")
    log.info("=" * 60)

    result = validate_evolution(baseline_scores, evolved_scores)

    if "error" in result:
        log.error("  Error: %s", result["error"])
    else:
        log.info("  N runs:           %d", result["n_runs"])
        log.info("  p-value:          %.6f", result["p_value"])
        log.info("  Effect size (d):  %.4f", result["effect_size"])
        log.info("  Significant:      %s", result["significant"])
        log.info("  Mean improvement: %.4f", result["mean_improvement"])
        log.info("  GATE PASSED:      %s", result["gate_passed"])

    # Save results
    output = {
        "task_ids": task_ids,
        "baseline_scores": baseline_scores,
        "evolved_scores": evolved_scores,
        "baseline_pass_rate": sum(baseline_scores) / len(baseline_scores),
        "evolved_pass_rate": sum(evolved_scores) / len(evolved_scores),
        "validation": result,
        "config": {"n_runs": n_runs, "tier": tier},
    }
    output_path = Path("docs/benchmarks") / f"evolution-validation-{time.strftime('%Y-%m-%d')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolution validation: template-only vs 6-path on BigCodeBench Hard"
    )
    parser.add_argument("--n-runs", type=int, default=20, help="Paired task count (default: 20, min: 10)")
    parser.add_argument("--tier", default="budget", help="LLM tier (default: budget)")
    args = parser.parse_args()

    if args.n_runs < 10:
        log.error("Need at least 10 runs for statistical validity (got %d)", args.n_runs)
        sys.exit(1)

    # Load .env
    try:
        from dotenv import load_dotenv
        for parent in [Path.cwd()] + list(Path.cwd().parents):
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                break
    except ImportError:
        pass

    result = asyncio.run(run_validation(n_runs=args.n_runs, tier=args.tier))
    sys.exit(0 if result.get("gate_passed", False) else 1)


if __name__ == "__main__":
    main()
