#!/usr/bin/env python3
"""Collect shadow routing traces for Phase 5 gate validation.

Runs the 50 ground-truth routing tasks through both Rust and Python routers
in shadow mode, logging divergences to ~/.sage/shadow_traces.jsonl.

Usage::

    # Requires sage-core built with maturin develop
    python scripts/collect_shadow_traces.py [--rounds N]

    # Run 20 rounds (1000 traces) to hit the hard gate
    python scripts/collect_shadow_traces.py --rounds 20

Phase 5 gates:
    - Soft:  500 traces, <10% divergence → enable Rust-primary routing
    - Hard: 1000 traces, <5%  divergence → safe to delete Python routing
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("shadow_traces")


def load_ground_truth() -> list[dict]:
    """Load routing ground truth tasks from config."""
    gt_path = Path(__file__).parent.parent / "config" / "routing_ground_truth.json"
    if not gt_path.exists():
        log.error("Ground truth file not found: %s", gt_path)
        sys.exit(1)
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["tasks"]


def create_routers():
    """Create Rust and Python routers for shadow comparison."""
    rust_router = None
    python_router = None

    # Try Rust router (SystemRouter)
    try:
        from sage_core import SystemRouter, ModelRegistry
        registry = ModelRegistry()
        # Load models from TOML if available
        models_toml = Path(__file__).parent.parent / "config" / "models.toml"
        if models_toml.exists():
            registry = ModelRegistry.from_toml(str(models_toml))
        rust_router = SystemRouter(registry)
        log.info("Rust SystemRouter: available")
    except ImportError:
        log.warning("sage_core not installed — Rust router unavailable")
    except Exception as e:
        log.warning("Rust router init failed: %s", e)

    # Python router (AdaptiveRouter or ComplexityRouter)
    try:
        from sage.strategy.adaptive_router import AdaptiveRouter
        python_router = AdaptiveRouter()
        log.info("Python AdaptiveRouter: available")
    except ImportError:
        try:
            from sage.strategy.metacognition import ComplexityRouter
            python_router = ComplexityRouter()
            log.info("Python ComplexityRouter: available (fallback)")
        except ImportError:
            log.warning("No Python router available")

    return rust_router, python_router


async def collect_traces(rounds: int = 1) -> None:
    """Run ground-truth tasks through shadow router for N rounds."""
    tasks = load_ground_truth()
    rust_router, python_router = create_routers()

    if rust_router is None and python_router is None:
        log.error("No routers available. Build sage-core and install sage-python first.")
        sys.exit(1)

    if rust_router is None or python_router is None:
        log.error(
            "Shadow comparison requires BOTH routers. "
            "Rust=%s Python=%s",
            "available" if rust_router else "MISSING",
            "available" if python_router else "MISSING",
        )
        sys.exit(1)

    from sage.routing.shadow import ShadowRouter
    shadow = ShadowRouter(
        rust_router=rust_router,
        python_metacognition=python_router,
    )
    # Load existing traces for continuity
    shadow.load_existing_traces()
    log.info(
        "Starting with %d existing traces (%.1f%% divergence)",
        shadow.total, shadow.divergence_rate() * 100,
    )

    total_tasks = len(tasks) * rounds
    log.info("Running %d tasks (%d GT x %d rounds)", total_tasks, len(tasks), rounds)

    for round_num in range(1, rounds + 1):
        for i, gt_task in enumerate(tasks, 1):
            task_str = gt_task["task"]
            try:
                await shadow.route(task_str, budget=10.0)
            except Exception as e:
                log.warning("Task %d failed: %s", gt_task["id"], e)

            # Progress every 50 tasks
            done = (round_num - 1) * len(tasks) + i
            if done % 50 == 0 or done == total_tasks:
                log.info(
                    "Progress: %d/%d traces | divergence=%.1f%% | "
                    "soft_gate=%s | hard_gate=%s",
                    shadow.total, total_tasks + shadow.total - done,
                    shadow.divergence_rate() * 100,
                    "PASS" if shadow.is_phase5_soft_ready() else "not yet",
                    "PASS" if shadow.is_phase5_hard_ready() else "not yet",
                )

    # Final report
    log.info("=" * 60)
    log.info("Shadow trace collection complete")
    log.info("  Total comparisons: %d", shadow.total)
    log.info("  System mismatches: %d", shadow.stats["system_mismatches"])
    log.info("  Divergence rate:   %.2f%%", shadow.divergence_rate() * 100)
    log.info("  Soft gate (500, <10%%):  %s", "PASS" if shadow.is_phase5_soft_ready() else "FAIL")
    log.info("  Hard gate (1000, <5%%): %s", "PASS" if shadow.is_phase5_hard_ready() else "FAIL")
    log.info("  Traces: ~/.sage/shadow_traces.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Collect shadow routing traces")
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of rounds over the 50 GT tasks (default: 1 = 50 traces)",
    )
    args = parser.parse_args()
    asyncio.run(collect_traces(rounds=args.rounds))


if __name__ == "__main__":
    main()
