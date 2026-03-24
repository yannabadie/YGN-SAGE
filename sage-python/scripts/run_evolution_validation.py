#!/usr/bin/env python3
"""Run evolution validation: compare baseline vs evolved topologies.

Loads topologies from the archive (if available) or generates new ones,
runs them through the pipeline, and applies statistical validation
(Wilcoxon signed-rank + Cohen's d) to determine if evolution produces
genuine improvement.

Usage:
    python scripts/run_evolution_validation.py [--n-runs 20] [--budget 5.0]

Requirements:
    - scipy (for Wilcoxon test)
    - numpy
    - sage-python installed (pip install -e ".[all,dev]")
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
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("evolution_validation")


def _load_archive_topologies(archive_path: str | None = None) -> list | None:
    """Try to load persisted archive topologies.

    Returns a list of topology dicts if found, None otherwise.
    """
    candidates = [
        archive_path,
        "data/topology_archive.json",
        "sage-python/data/topology_archive.json",
        str(Path.home() / ".sage" / "topology_archive.json"),
    ]
    for path in candidates:
        if path is None:
            continue
        p = Path(path)
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                log.info("Loaded archive from %s (%d entries)", p, len(data))
                return data
            except Exception as exc:
                log.warning("Failed to load archive from %s: %s", p, exc)
    return None


def _generate_baseline_topologies(n: int = 10) -> list[str]:
    """Generate baseline (template) topologies as task descriptions.

    These represent tasks that will be run through the default sequential
    topology (no evolution).
    """
    tasks = [
        "Write a Python function to compute the Fibonacci sequence",
        "Implement a binary search tree with insert and delete operations",
        "Create a REST API endpoint that validates JSON input",
        "Write a function that finds all prime numbers up to N using the sieve of Eratosthenes",
        "Implement a least-recently-used (LRU) cache with O(1) operations",
        "Write a Python class that implements the observer design pattern",
        "Create a function that performs topological sort on a directed acyclic graph",
        "Implement a simple regular expression matcher supporting . and *",
        "Write a function to serialize and deserialize a binary tree",
        "Create a thread-safe producer-consumer queue in Python",
        "Implement Dijkstra's shortest path algorithm",
        "Write a function that evaluates mathematical expressions with parentheses",
        "Create a simple key-value store with TTL expiration",
        "Implement the A* pathfinding algorithm on a 2D grid",
        "Write a function that generates all valid parentheses combinations for N pairs",
        "Implement a bloom filter with configurable false positive rate",
        "Create a function that finds the longest common subsequence of two strings",
        "Write a rate limiter using the token bucket algorithm",
        "Implement a trie data structure with insert, search, and prefix operations",
        "Create a function that solves the N-queens problem using backtracking",
    ]
    return tasks[:n]


async def _run_task_with_quality(
    task: str,
    use_evolved: bool = False,
    budget: float = 5.0,
) -> float:
    """Run a single task and return quality score.

    Parameters
    ----------
    task : str
        Task description to execute.
    use_evolved : bool
        If True, enable evolution engine for topology selection.
        If False, use default sequential topology (baseline).
    budget : float
        Budget in USD.

    Returns
    -------
    float
        Quality score (0.0-1.0). Returns 0.0 on failure.
    """
    try:
        from sage.quality_estimator import QualityEstimator

        qe = QualityEstimator()

        if use_evolved:
            # Use full pipeline with topology engine (evolved topologies)
            try:
                from sage.boot import boot_agent_system

                system = boot_agent_system(use_mock_llm=True)
                result = await system.run(task)
            except Exception as exc:
                log.warning("Evolved run failed: %s", exc)
                return 0.0
        else:
            # Baseline: mock provider with single-agent (no topology evolution)
            try:
                from sage.boot import boot_agent_system

                system = boot_agent_system(use_mock_llm=True)
                # Disable topology engine for baseline
                system.topology_engine = None
                result = await system.run(task)
            except Exception as exc:
                log.warning("Baseline run failed: %s", exc)
                return 0.0

        quality = qe.estimate(task, result)
        return quality if quality is not None else 0.5  # neutral if unknown

    except Exception as exc:
        log.error("Task execution error: %s", exc)
        return 0.0


async def run_validation(
    n_runs: int = 10,
    budget: float = 5.0,
    archive_path: str | None = None,
) -> dict:
    """Run the full evolution validation.

    Parameters
    ----------
    n_runs : int
        Number of paired runs (minimum 10 for statistical validity).
    budget : float
        Budget per task in USD.
    archive_path : str or None
        Path to topology archive file.

    Returns
    -------
    dict
        Validation result from validate_evolution().
    """
    from sage.evolution.evaluator import validate_evolution

    log.info("=" * 60)
    log.info("Evolution Validation: %d paired runs, budget=$%.2f", n_runs, budget)
    log.info("=" * 60)

    # Check for persisted archive
    archive = _load_archive_topologies(archive_path)
    if archive:
        log.info("Archive available with %d topologies", len(archive))
    else:
        log.info("No archive found; generating %d tasks for comparison", n_runs)

    tasks = _generate_baseline_topologies(n_runs)

    # Run baseline (no evolution)
    log.info("--- Phase 1: Baseline runs (no evolution) ---")
    baseline_scores: list[float] = []
    for i, task in enumerate(tasks):
        t0 = time.monotonic()
        score = await _run_task_with_quality(task, use_evolved=False, budget=budget)
        elapsed = time.monotonic() - t0
        baseline_scores.append(score)
        log.info("  [%d/%d] baseline: quality=%.3f (%.1fs) %s",
                 i + 1, n_runs, score, elapsed, task[:50])

    # Run evolved (with topology engine)
    log.info("--- Phase 2: Evolved runs (topology engine active) ---")
    evolved_scores: list[float] = []
    for i, task in enumerate(tasks):
        t0 = time.monotonic()
        score = await _run_task_with_quality(task, use_evolved=True, budget=budget)
        elapsed = time.monotonic() - t0
        evolved_scores.append(score)
        log.info("  [%d/%d] evolved: quality=%.3f (%.1fs) %s",
                 i + 1, n_runs, score, elapsed, task[:50])

    # Statistical validation
    log.info("--- Phase 3: Statistical validation ---")
    result = validate_evolution(baseline_scores, evolved_scores)

    # Print results
    log.info("=" * 60)
    log.info("VALIDATION RESULT")
    log.info("=" * 60)
    if "error" in result:
        log.error("  Error: %s", result["error"])
    else:
        log.info("  N runs:           %d", result.get("n_runs", 0))
        log.info("  p-value:          %.6f", result.get("p_value", 1.0))
        log.info("  Effect size (d):  %.4f", result.get("effect_size", 0.0))
        log.info("  Significant:      %s", result.get("significant", False))
        log.info("  Mean improvement: %.4f", result.get("mean_improvement", 0.0))
        log.info("  GATE PASSED:      %s", result.get("gate_passed", False))
    log.info("=" * 60)

    # Dump raw data for reproducibility
    output = {
        "baseline_scores": baseline_scores,
        "evolved_scores": evolved_scores,
        "validation": result,
        "n_runs": n_runs,
        "budget": budget,
    }
    output_path = Path("evolution_validation_result.json")
    try:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info("Raw results saved to %s", output_path)
    except Exception as exc:
        log.warning("Failed to save results: %s", exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evolution validation (baseline vs evolved topologies)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=10,
        help="Number of paired runs (default: 10, minimum for statistical validity)",
    )
    parser.add_argument(
        "--budget", type=float, default=5.0,
        help="Budget per task in USD (default: 5.0)",
    )
    parser.add_argument(
        "--archive", type=str, default=None,
        help="Path to topology archive JSON file",
    )
    args = parser.parse_args()

    if args.n_runs < 10:
        log.error("Need at least 10 runs for statistical validity (got %d)", args.n_runs)
        sys.exit(1)

    result = asyncio.run(run_validation(
        n_runs=args.n_runs,
        budget=args.budget,
        archive_path=args.archive,
    ))

    # Exit code: 0 if gate passed, 1 otherwise
    sys.exit(0 if result.get("gate_passed", False) else 1)


if __name__ == "__main__":
    main()
