"""MASBENCH Official 5-Axes Ablation Benchmark.

Methodological fixes (April 2 2026):
- Fresh system boot per axis (no state leakage between axes)
- Pinned provider for bare vs SAGE (same model, topology is the variable)
- Incremental JSON save after each axis (crash-resilient)
"""
import asyncio
import json
import logging
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.environ.get(
    "MASBENCH_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "docs", "benchmarks", "masbench-runs"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "masbench_official.log")
JSON_FILE = os.path.join(OUTPUT_DIR, "masbench_official_results.json")

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(name)-25s] %(levelname)-5s %(message)s"))

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(fh)
logging.root.addHandler(ch)

for mod in [
    "sage.pipeline", "sage.pipeline_stages", "sage.topology.runner",
    "sage.topology_controller", "sage.strategy.knn_router",
    "sage.llm.provider_pool", "sage.bench.masbench", "sage.boot",
]:
    logging.getLogger(mod).setLevel(logging.DEBUG)

log = logging.getLogger("masbench_official")


def _save_incremental(all_results: dict, path: str) -> None:
    """Save results after each axis — crash-resilient."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


async def main():
    log.info("=" * 70)
    log.info("MASBENCH OFFICIAL 5-AXES ABLATION — %s", time.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 70)

    from sage.bench.masbench import MASBenchAblation

    axes = ["breadth", "depth", "horizon", "parallel", "robustness"]
    limit = 50
    all_results = {}

    # Load previous partial results if resuming
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, encoding="utf-8") as f:
                all_results = json.load(f)
            log.info("Resumed from %d completed axes", len(all_results))
        except (json.JSONDecodeError, IOError):
            pass

    for axis in axes:
        if axis in all_results:
            log.info("SKIP %s (already completed: %.1f%%)", axis, all_results[axis]["sage_pct"])
            continue

        log.info("=" * 70)
        log.info("AXIS: %s (%d tasks)", axis, limit)
        log.info("=" * 70)

        # FRESH system per axis — no state leakage (bandit, memory, MAP-Elites)
        from sage.boot import boot_agent_system
        system = boot_agent_system()
        log.info("Fresh system booted for axis=%s: pipeline=%s", axis, system.pipeline is not None)

        ablation = MASBenchAblation(system=system, axis=axis)
        t0 = time.perf_counter()
        reports = await ablation.run(limit=limit)
        elapsed = time.perf_counter() - t0

        bare = reports.get("bare")
        sage = reports.get("sage_full")
        bare_pct = bare.pass_rate * 100 if bare else 0
        sage_pct = sage.pass_rate * 100 if sage else 0
        delta = sage_pct - bare_pct

        log.info(
            "RESULT %s: bare=%.1f%% sage=%.1f%% delta=%+.1fpp (%.0fs)",
            axis, bare_pct, sage_pct, delta, elapsed,
        )

        all_results[axis] = {
            "bare_pct": round(bare_pct, 1),
            "sage_pct": round(sage_pct, 1),
            "delta_pp": round(delta, 1),
            "elapsed_s": round(elapsed, 1),
            "bare_results": [
                {"task_id": r.task_id, "passed": r.passed, "latency_ms": round(r.latency_ms, 1)}
                for r in (bare.results if bare else [])
            ],
            "sage_results": [
                {"task_id": r.task_id, "passed": r.passed, "latency_ms": round(r.latency_ms, 1)}
                for r in (sage.results if sage else [])
            ],
        }

        # Incremental save — crash-resilient
        _save_incremental(all_results, JSON_FILE)
        log.info("Saved incremental results (%d/%d axes)", len(all_results), len(axes))

    # Summary
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS")
    log.info("=" * 70)
    log.info("%-12s %8s %8s %8s", "Axis", "Bare", "SAGE", "Delta")
    log.info("-" * 40)
    for axis in axes:
        if axis in all_results:
            r = all_results[axis]
            log.info("%-12s %7.1f%% %7.1f%% %+7.1fpp", axis, r["bare_pct"], r["sage_pct"], r["delta_pp"])

    completed = [r for r in all_results.values() if "bare_pct" in r]
    if completed:
        avg_bare = sum(r["bare_pct"] for r in completed) / len(completed)
        avg_sage = sum(r["sage_pct"] for r in completed) / len(completed)
        log.info("-" * 40)
        log.info("%-12s %7.1f%% %7.1f%% %+7.1fpp", "AVERAGE", avg_bare, avg_sage, avg_sage - avg_bare)

    log.info("Results: %s", JSON_FILE)
    log.info("Full log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
