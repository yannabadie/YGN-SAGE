"""Collect quality labels using Rust Z3 QualityLabeler.

Generates (task, response, quality_score) triples from benchmark tasks
by running them through SAGE and labeling with formal verification.

Usage:
    python scripts/collect_quality_labels.py --dataset humaneval --limit 20
    python scripts/collect_quality_labels.py --dataset bigcodebench --subset hard --limit 50
    python scripts/collect_quality_labels.py --list  # show existing labels count
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(levelname)s: %(message)s",
)
log = logging.getLogger("collect_labels")


def _load_tasks(dataset: str, subset: str, limit: int | None):
    """Load tasks from specified dataset."""
    if dataset == "humaneval":
        try:
            from evalplus.data import get_human_eval_plus
            problems = get_human_eval_plus()
        except ImportError:
            log.error("evalplus not installed: pip install evalplus")
            sys.exit(1)
    elif dataset == "bigcodebench":
        try:
            from bigcodebench.data import get_bigcodebench
            problems = get_bigcodebench(subset=subset)
        except ImportError:
            log.error("bigcodebench not installed: pip install bigcodebench")
            sys.exit(1)
    else:
        log.error("Unknown dataset: %s", dataset)
        sys.exit(1)

    task_ids = list(problems.keys())
    if limit:
        task_ids = task_ids[:limit]
    return [(tid, problems[tid]) for tid in task_ids]


def main():
    parser = argparse.ArgumentParser(description="Collect Z3 quality labels for training")
    parser.add_argument("--dataset", choices=["humaneval", "bigcodebench"], default="humaneval")
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/quality_labels.jsonl")
    parser.add_argument("--list", action="store_true", help="Show existing label count and exit")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.list:
        if output_path.exists():
            count = sum(1 for _ in open(output_path, encoding="utf-8"))
            assessable = sum(
                1 for line in open(output_path, encoding="utf-8")
                if json.loads(line).get("assessable", False)
            )
            log.info("Labels: %d total, %d assessable in %s", count, assessable, output_path)
        else:
            log.info("No labels file found at %s", output_path)
        return

    # Load Z3 labeler (Rust)
    try:
        from sage_core import QualityLabeler  # type: ignore[import-not-found]
    except ImportError:
        log.error(
            "sage_core not built with smt+tool-executor features.\n"
            "Run: cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor"
        )
        sys.exit(1)

    labeler = QualityLabeler()
    log.info("Z3 QualityLabeler ready")

    # Boot SAGE system
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus

    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=EventBus())
    log.info("System booted")

    # Load tasks
    tasks = _load_tasks(args.dataset, args.subset, args.limit)
    log.info("Loaded %d tasks from %s", len(tasks), args.dataset)

    # Collect labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    count = 0
    assessable = 0
    t_start = time.time()

    with open(output_path, mode, encoding="utf-8") as f:
        for i, (tid, task) in enumerate(tasks):
            # Get prompt
            prompt = task.get("instruct_prompt", task.get("prompt", ""))
            if not prompt:
                continue

            # Generate response via SAGE
            t0 = time.time()
            try:
                response = asyncio.run(
                    asyncio.wait_for(system.run(prompt), timeout=120)
                )
            except asyncio.TimeoutError:
                log.warning("[%d/%d] %s TIMEOUT", i + 1, len(tasks), tid)
                continue
            except Exception as exc:
                log.warning("[%d/%d] %s GEN_ERROR: %s", i + 1, len(tasks), tid, str(exc)[:100])
                continue
            gen_ms = (time.time() - t0) * 1000

            # Label with Z3
            label = labeler.label(prompt, response)

            entry = {
                "task_id": tid,
                "task": prompt[:500],
                "response": response[:2000],
                "score": label.score if label else None,
                "assessable": label.assessable if label else False,
                "checks_passed": label.checks_passed if label else 0,
                "checks_total": label.checks_total if label else 0,
                "details": label.details if label else "{}",
                "gen_latency_ms": round(gen_ms, 1),
                "dataset": args.dataset,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            count += 1
            if label and label.assessable:
                assessable += 1

            score_str = f"{label.score:.2f}" if label else "N/A"
            checks_str = f"{label.checks_passed}/{label.checks_total}" if label else "0/0"
            log.info(
                "[%d/%d] %s score=%s checks=%s (%.0fms)",
                i + 1, len(tasks), tid, score_str, checks_str, gen_ms,
            )

    elapsed = time.time() - t_start
    log.info(
        "Done: %d labels (%d assessable) in %.0fs, saved to %s",
        count, assessable, elapsed, output_path,
    )


if __name__ == "__main__":
    main()
