"""Statistical proof: does SAGE's evolution system improve task outcomes?

Addresses Audit3 F-09: "No quantitative evidence that evolution improves
task outcomes yet."

Methodology:
  - 2 configs: SAGE-full (evolution ON) vs SAGE-no-evolution (evolution OFF)
  - N independent runs per config on the same HumanEval+ task set
  - Paired statistical tests (Wilcoxon signed-rank, McNemar's)
  - Effect size (Cohen's d) and bootstrap 95% CI

Evolution system under test:
  - TopologyEngine: MAP-Elites + CMA-ME + MCTS topology generation with
    outcome recording (learning loop feeds back into archive)
  - ContextualBandit: Thompson sampling for model/template selection

"Evolution OFF" means: topology_engine=None, bandit=None. The system still
routes tasks and calls LLMs, but without evolutionary topology search or
bandit-guided model selection.

Usage:
  python scripts/evolution_statistical_proof.py --runs 10 --limit 20
  python scripts/evolution_statistical_proof.py --runs 5 --limit 10 --seed 42
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evo_proof")

# Silence noisy sub-loggers during benchmark runs
for _quiet in (
    "sage.boot", "sage.agent_loop", "sage.llm", "sage.orchestrator",
    "sage.providers", "sage.memory", "sage.routing", "sage.topology",
    "sage.strategy", "sage.guardrails", "sage.evolution", "sage.events",
    "httpx", "httpcore", "urllib3", "google",
    "sentence_transformers", "transformers",
):
    logging.getLogger(_quiet).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Outcome of a single benchmark run (all tasks in one config)."""
    config: str
    run_id: int
    seed: int
    pass_rate: float
    passed: int
    total: int
    avg_latency_ms: float
    total_cost_usd: float
    per_task: list[dict]  # [{task_id, passed, latency_ms, cost_usd}, ...]
    wall_time_s: float = 0.0


@dataclass
class StatResult:
    """Statistical test results."""
    # Wilcoxon signed-rank test
    wilcoxon_stat: float | None = None
    wilcoxon_p: float | None = None
    wilcoxon_interpretation: str = ""

    # Cohen's d effect size
    cohens_d: float | None = None
    effect_size_label: str = ""

    # Bootstrap 95% CI for delta (full - no_evo)
    bootstrap_mean_delta: float | None = None
    bootstrap_ci_lower: float | None = None
    bootstrap_ci_upper: float | None = None

    # McNemar's test (per-task pass/fail)
    mcnemar_stat: float | None = None
    mcnemar_p: float | None = None

    # Raw data
    full_rates: list[float] = field(default_factory=list)
    no_evo_rates: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core: boot system with/without evolution
# ---------------------------------------------------------------------------
def _boot_system(evolution_enabled: bool):
    """Boot an AgentSystem with evolution ON or OFF."""
    # Ensure CWD is the repo root so boot.py finds sage-core/config/cards.toml.
    # boot.py searches Path.cwd() / "sage-core" / "config" / "cards.toml".
    _repo_root = Path(__file__).resolve().parent.parent.parent
    _prev_cwd = Path.cwd()
    os.chdir(_repo_root)

    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=False, llm_tier="auto")

    os.chdir(_prev_cwd)

    if not evolution_enabled:
        # Disable the evolutionary components
        system.topology_engine = None
        system.bandit = None
        # Also clear topology engine ref on agent loop
        system.agent_loop.topology_engine = None
        system.agent_loop._current_topology = None
        log.info("Evolution DISABLED: topology_engine=None, bandit=None")
    else:
        has_topo = system.topology_engine is not None
        has_bandit = system.bandit is not None
        log.info(
            "Evolution ENABLED: topology_engine=%s, bandit=%s",
            "active" if has_topo else "MISSING",
            "active" if has_bandit else "MISSING",
        )
        if not has_topo and not has_bandit:
            log.warning(
                "Evolution requested ON but neither TopologyEngine nor "
                "ContextualBandit are available. This run will be equivalent "
                "to evolution-OFF. Results may be meaningless."
            )

    return system


# ---------------------------------------------------------------------------
# Single run: generate + evaluate on HumanEval+ tasks
# ---------------------------------------------------------------------------
async def _run_once(
    config_label: str,
    evolution_enabled: bool,
    limit: int,
    run_id: int,
    seed: int,
) -> RunResult:
    """Execute one full benchmark run and return results."""
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    log.info(
        "=== Run %d [%s] (seed=%d, limit=%d) ===",
        run_id, config_label, seed, limit,
    )

    t0 = time.perf_counter()

    # Boot a fresh system for each run (clean state)
    system = _boot_system(evolution_enabled)

    # Use EvalPlus bench to generate + evaluate
    from sage.bench.evalplus_bench import EvalPlusBench
    bench = EvalPlusBench(
        system=system,
        event_bus=system.event_bus,
        dataset="humaneval",
    )

    report = await bench.run(limit=limit)

    wall_time = time.perf_counter() - t0

    per_task = [
        {
            "task_id": r.task_id,
            "passed": r.passed,
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "error": r.error,
        }
        for r in report.results
    ]

    result = RunResult(
        config=config_label,
        run_id=run_id,
        seed=seed,
        pass_rate=report.pass_rate,
        passed=report.passed,
        total=report.total,
        avg_latency_ms=report.avg_latency_ms,
        total_cost_usd=sum(r.cost_usd for r in report.results),
        per_task=per_task,
        wall_time_s=round(wall_time, 1),
    )

    log.info(
        "  -> %s run %d: pass_rate=%.1f%% (%d/%d), "
        "latency=%.0fms, cost=$%.4f, wall=%.0fs",
        config_label, run_id,
        result.pass_rate * 100, result.passed, result.total,
        result.avg_latency_ms, result.total_cost_usd, result.wall_time_s,
    )

    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _compute_statistics(
    full_runs: list[RunResult],
    no_evo_runs: list[RunResult],
) -> StatResult:
    """Compute paired statistical tests on the two sets of runs."""
    from scipy.stats import wilcoxon

    full_rates = [r.pass_rate for r in full_runs]
    no_evo_rates = [r.pass_rate for r in no_evo_runs]

    stat = StatResult(
        full_rates=full_rates,
        no_evo_rates=no_evo_rates,
    )

    n = len(full_rates)
    diffs = [f - n for f, n in zip(full_rates, no_evo_rates)]

    # --- Wilcoxon signed-rank test ---
    # H0: median difference = 0
    # H1: full > no_evo (one-sided "greater")
    try:
        # Wilcoxon requires non-zero differences
        nonzero_diffs = [d for d in diffs if d != 0.0]
        if len(nonzero_diffs) < 2:
            stat.wilcoxon_stat = None
            stat.wilcoxon_p = 1.0
            stat.wilcoxon_interpretation = (
                "INCONCLUSIVE: fewer than 2 non-zero paired differences. "
                "The two configs produced identical results on most runs."
            )
        else:
            w_stat, w_p = wilcoxon(
                full_rates, no_evo_rates, alternative="greater",
            )
            stat.wilcoxon_stat = float(w_stat)
            stat.wilcoxon_p = float(w_p)

            if w_p < 0.01:
                stat.wilcoxon_interpretation = (
                    f"SIGNIFICANT (p={w_p:.4f} < 0.01): strong evidence "
                    "that evolution improves pass rate."
                )
            elif w_p < 0.05:
                stat.wilcoxon_interpretation = (
                    f"SIGNIFICANT (p={w_p:.4f} < 0.05): moderate evidence "
                    "that evolution improves pass rate."
                )
            else:
                stat.wilcoxon_interpretation = (
                    f"NOT SIGNIFICANT (p={w_p:.4f} >= 0.05): no evidence "
                    "that evolution improves pass rate."
                )
    except Exception as e:
        stat.wilcoxon_interpretation = f"TEST FAILED: {e}"

    # --- Cohen's d (paired) ---
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    if std_diff > 0:
        stat.cohens_d = mean_diff / std_diff
    else:
        # All differences are identical (including all-zero)
        stat.cohens_d = 0.0 if mean_diff == 0 else float("inf")

    d = abs(stat.cohens_d) if stat.cohens_d is not None else 0.0
    if d < 0.2:
        stat.effect_size_label = "negligible"
    elif d < 0.5:
        stat.effect_size_label = "small"
    elif d < 0.8:
        stat.effect_size_label = "medium"
    else:
        stat.effect_size_label = "large"

    # --- Bootstrap 95% CI for mean delta ---
    try:
        rng = np.random.default_rng(seed=12345)
        n_boot = 10000
        boot_deltas = np.empty(n_boot)
        full_arr = np.array(full_rates)
        no_evo_arr = np.array(no_evo_rates)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_deltas[i] = np.mean(full_arr[idx]) - np.mean(no_evo_arr[idx])

        stat.bootstrap_mean_delta = float(np.mean(boot_deltas))
        stat.bootstrap_ci_lower = float(np.percentile(boot_deltas, 2.5))
        stat.bootstrap_ci_upper = float(np.percentile(boot_deltas, 97.5))
    except Exception as e:
        log.warning("Bootstrap CI failed: %s", e)

    # --- McNemar's test (per-task pass/fail contingency) ---
    try:
        _compute_mcnemar(stat, full_runs, no_evo_runs)
    except Exception as e:
        log.warning("McNemar's test failed: %s", e)

    return stat


def _compute_mcnemar(
    stat: StatResult,
    full_runs: list[RunResult],
    no_evo_runs: list[RunResult],
) -> None:
    """McNemar's test on pooled per-task pass/fail across all runs.

    Contingency table:
                    no_evo PASS   no_evo FAIL
    full PASS         a             b
    full FAIL         c             d

    McNemar tests whether b != c (discordant pairs).
    """
    # Build pass/fail lookup: (run_id, task_id) -> bool
    full_pass: dict[tuple[int, str], bool] = {}
    no_evo_pass: dict[tuple[int, str], bool] = {}

    for r in full_runs:
        for t in r.per_task:
            full_pass[(r.run_id, t["task_id"])] = t["passed"]
    for r in no_evo_runs:
        for t in r.per_task:
            no_evo_pass[(r.run_id, t["task_id"])] = t["passed"]

    # Count discordant pairs
    b = 0  # full PASS, no_evo FAIL
    c = 0  # full FAIL, no_evo PASS

    common_keys = set(full_pass.keys()) & set(no_evo_pass.keys())
    for key in common_keys:
        fp = full_pass[key]
        np_ = no_evo_pass[key]
        if fp and not np_:
            b += 1
        elif not fp and np_:
            c += 1

    # McNemar's chi-squared (with continuity correction)
    if b + c == 0:
        stat.mcnemar_stat = 0.0
        stat.mcnemar_p = 1.0
        return

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    stat.mcnemar_stat = float(chi2)
    stat.mcnemar_p = float(1.0 - chi2_dist.cdf(chi2, df=1))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _generate_report(
    full_runs: list[RunResult],
    no_evo_runs: list[RunResult],
    stats: StatResult,
    args: argparse.Namespace,
    total_wall_s: float,
) -> dict[str, Any]:
    """Build the final JSON report."""
    full_mean = float(np.mean(stats.full_rates))
    no_evo_mean = float(np.mean(stats.no_evo_rates))
    delta = full_mean - no_evo_mean

    # Interpretation
    if stats.wilcoxon_p is not None and stats.wilcoxon_p < 0.05:
        if delta > 0:
            if stats.cohens_d is not None and abs(stats.cohens_d) >= 0.5:
                verdict = "EVOLUTION HELPS (medium+ effect, statistically significant)"
            elif stats.cohens_d is not None and abs(stats.cohens_d) >= 0.2:
                verdict = "EVOLUTION HELPS (small effect, statistically significant)"
            else:
                verdict = "EVOLUTION HELPS (negligible effect size despite statistical significance)"
        else:
            verdict = "EVOLUTION HURTS (statistically significant negative delta)"
    elif delta < -0.05:
        verdict = "EVOLUTION MAY HURT (not statistically significant, but negative trend)"
    elif abs(delta) < 0.02:
        verdict = "NO EFFECT DETECTED (pass rates nearly identical)"
    else:
        verdict = "EFFECT NOT PROVEN (may help, may not -- insufficient evidence)"

    report = {
        "experiment": "evolution_statistical_proof",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "runs_per_config": args.runs,
            "tasks_per_run": args.limit,
            "total_invocations": args.runs * args.limit * 2,
            "base_seed": args.seed,
            "dataset": "humaneval+",
        },
        "summary": {
            "verdict": verdict,
            "full_mean_pass_rate": round(full_mean, 4),
            "no_evo_mean_pass_rate": round(no_evo_mean, 4),
            "delta_pass_rate": round(delta, 4),
            "delta_pp": round(delta * 100, 2),
        },
        "statistics": {
            "wilcoxon": {
                "statistic": stats.wilcoxon_stat,
                "p_value": stats.wilcoxon_p,
                "interpretation": stats.wilcoxon_interpretation,
                "test": "Wilcoxon signed-rank (paired, one-sided H1: full > no_evo)",
            },
            "effect_size": {
                "cohens_d": round(stats.cohens_d, 4) if stats.cohens_d is not None else None,
                "label": stats.effect_size_label,
                "interpretation": (
                    f"|d| = {abs(stats.cohens_d):.4f} -> {stats.effect_size_label}"
                    if stats.cohens_d is not None else "N/A"
                ),
            },
            "bootstrap_ci": {
                "mean_delta": round(stats.bootstrap_mean_delta, 4) if stats.bootstrap_mean_delta is not None else None,
                "ci_95_lower": round(stats.bootstrap_ci_lower, 4) if stats.bootstrap_ci_lower is not None else None,
                "ci_95_upper": round(stats.bootstrap_ci_upper, 4) if stats.bootstrap_ci_upper is not None else None,
                "method": "BCa-free percentile bootstrap, 10000 resamples",
            },
            "mcnemar": {
                "statistic": round(stats.mcnemar_stat, 4) if stats.mcnemar_stat is not None else None,
                "p_value": round(stats.mcnemar_p, 4) if stats.mcnemar_p is not None else None,
                "test": "McNemar's chi-squared (continuity-corrected) on pooled per-task pass/fail",
            },
        },
        "per_run": {
            "full": [
                {
                    "run_id": r.run_id,
                    "seed": r.seed,
                    "pass_rate": round(r.pass_rate, 4),
                    "passed": r.passed,
                    "total": r.total,
                    "avg_latency_ms": round(r.avg_latency_ms, 1),
                    "total_cost_usd": round(r.total_cost_usd, 4),
                    "wall_time_s": r.wall_time_s,
                }
                for r in full_runs
            ],
            "no_evolution": [
                {
                    "run_id": r.run_id,
                    "seed": r.seed,
                    "pass_rate": round(r.pass_rate, 4),
                    "passed": r.passed,
                    "total": r.total,
                    "avg_latency_ms": round(r.avg_latency_ms, 1),
                    "total_cost_usd": round(r.total_cost_usd, 4),
                    "wall_time_s": r.wall_time_s,
                }
                for r in no_evo_runs
            ],
        },
        "raw_pass_rates": {
            "full": [round(r, 4) for r in stats.full_rates],
            "no_evolution": [round(r, 4) for r in stats.no_evo_rates],
        },
        "cost": {
            "total_usd": round(
                sum(r.total_cost_usd for r in full_runs)
                + sum(r.total_cost_usd for r in no_evo_runs),
                4,
            ),
            "full_total_usd": round(sum(r.total_cost_usd for r in full_runs), 4),
            "no_evo_total_usd": round(sum(r.total_cost_usd for r in no_evo_runs), 4),
        },
        "timing": {
            "total_wall_time_s": round(total_wall_s, 1),
            "avg_run_time_s": round(
                total_wall_s / (len(full_runs) + len(no_evo_runs)), 1
            ) if (full_runs or no_evo_runs) else 0,
        },
        "methodology": {
            "evolution_components": [
                "TopologyEngine (MAP-Elites + CMA-ME + MCTS + S-MMU learning loop)",
                "ContextualBandit (Thompson sampling for model/template selection)",
            ],
            "evolution_off_means": (
                "topology_engine=None, bandit=None on AgentSystem. "
                "Tasks still routed and solved, but without evolutionary "
                "topology search or bandit-guided model selection."
            ),
            "benchmark": "EvalPlus HumanEval+ (80x harder tests than original HumanEval)",
            "evaluation": "Subprocess sandbox, solution vs canonical comparison",
            "pairing": (
                "Runs are paired by run_id (same seed, same task order). "
                "Wilcoxon signed-rank is appropriate for paired non-parametric data."
            ),
        },
    }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _main(args: argparse.Namespace) -> None:
    log.info("Evolution Statistical Proof")
    log.info("  runs=%d, limit=%d, seed=%d", args.runs, args.limit, args.seed)
    log.info(
        "  Expected invocations: %d (20 tasks x %d runs x 2 configs)",
        args.limit * args.runs * 2, args.runs,
    )

    total_t0 = time.perf_counter()

    # Generate per-run seeds (deterministic from base seed)
    rng = np.random.default_rng(seed=args.seed)
    run_seeds = [int(rng.integers(0, 2**31)) for _ in range(args.runs)]

    full_runs: list[RunResult] = []
    no_evo_runs: list[RunResult] = []

    # Interleave configs to reduce temporal confounds (e.g., API throttling).
    # Pattern: full_0, no_evo_0, full_1, no_evo_1, ...
    for run_id in range(args.runs):
        seed = run_seeds[run_id]

        # --- FULL (evolution ON) ---
        full_result = await _run_once(
            config_label="full",
            evolution_enabled=True,
            limit=args.limit,
            run_id=run_id,
            seed=seed,
        )
        full_runs.append(full_result)

        # --- NO EVOLUTION ---
        no_evo_result = await _run_once(
            config_label="no-evolution",
            evolution_enabled=False,
            limit=args.limit,
            run_id=run_id,
            seed=seed,
        )
        no_evo_runs.append(no_evo_result)

        # Progress summary
        elapsed = time.perf_counter() - total_t0
        done = (run_id + 1) * 2
        total = args.runs * 2
        eta = elapsed / done * (total - done) if done > 0 else 0
        log.info(
            "Progress: %d/%d runs complete (%.0fs elapsed, ~%.0fs remaining)",
            run_id + 1, args.runs, elapsed, eta,
        )

    total_wall = time.perf_counter() - total_t0

    # --- Compute statistics ---
    log.info("Computing statistics...")
    stats = _compute_statistics(full_runs, no_evo_runs)

    # --- Generate report ---
    report = _generate_report(full_runs, no_evo_runs, stats, args, total_wall)

    # --- Save ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evolution_statistical_proof.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Report saved to %s", out_path)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("EVOLUTION STATISTICAL PROOF — RESULTS")
    print("=" * 70)
    print(f"  Runs per config:     {args.runs}")
    print(f"  Tasks per run:       {args.limit}")
    print(f"  Total invocations:   {args.runs * args.limit * 2}")
    print(f"  Total wall time:     {total_wall:.0f}s ({total_wall / 60:.1f} min)")
    print()
    print(f"  Full (evolution ON):     mean pass rate = {np.mean(stats.full_rates):.1%}")
    print(f"  No-evolution (evo OFF):  mean pass rate = {np.mean(stats.no_evo_rates):.1%}")
    print(f"  Delta:                   {report['summary']['delta_pp']:+.2f} pp")
    print()
    print(f"  Wilcoxon p-value:  {stats.wilcoxon_p}")
    print(f"  Cohen's d:         {stats.cohens_d}")
    print(f"  Effect size:       {stats.effect_size_label}")
    if stats.bootstrap_ci_lower is not None:
        print(
            f"  Bootstrap 95% CI:  [{stats.bootstrap_ci_lower:.4f}, "
            f"{stats.bootstrap_ci_upper:.4f}]"
        )
    if stats.mcnemar_p is not None:
        print(f"  McNemar p-value:   {stats.mcnemar_p:.4f}")
    print()
    print(f"  VERDICT: {report['summary']['verdict']}")
    print("=" * 70)
    print(f"\nFull report: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical proof: does evolution improve SAGE task outcomes?",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of independent runs per config (default: 10)",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of HumanEval+ tasks per run (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Base random seed for reproducibility (default: 2026)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for report JSON (default: data/)",
    )

    args = parser.parse_args()

    # Validate
    if args.runs < 2:
        parser.error("--runs must be >= 2 for statistical tests")
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
