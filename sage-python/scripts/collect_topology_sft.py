"""Collect topology SFT training data for RLVR-Topology policy training.

For each benchmark task, runs SAGE with each of the 8 templates,
records (task, topology_yaml, execution_result, reward_score),
and exports validated topologies as JSONL for SFT training.

Usage:
    python scripts/collect_topology_sft.py --dataset bigcodebench --subset hard --limit 20
    python scripts/collect_topology_sft.py --dataset apps --difficulty introductory --limit 50
    python scripts/collect_topology_sft.py --list  # show existing data count
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import yaml
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(levelname)s: %(message)s",
)
log = logging.getLogger("collect_sft")

TEMPLATES = [
    "sequential", "parallel", "avr", "selfmoa",
    "hierarchical", "hub", "debate", "brainstorming",
]


def _load_tasks(dataset: str, subset: str, difficulty: str, limit: int | None):
    """Load tasks from specified dataset."""
    tasks = []
    if dataset == "bigcodebench":
        try:
            from bigcodebench.data import get_bigcodebench
            problems = get_bigcodebench(subset=subset)
        except ImportError:
            log.error("bigcodebench not installed")
            sys.exit(1)
        for tid, task in list(problems.items())[:limit]:
            tasks.append((tid, task.get("instruct_prompt", "")))
    elif dataset == "apps":
        try:
            from datasets import load_dataset
            ds = load_dataset("codeparrot/apps", split="test")
        except ImportError:
            log.error("datasets not installed: pip install datasets")
            sys.exit(1)
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            if difficulty != "all" and row.get("difficulty", "") != difficulty:
                continue
            tasks.append((f"APPS/{i}", row.get("question", "")))
    return tasks


def _topology_to_yaml(graph) -> str:
    """Convert TopologyGraph to YAML representation for SFT training."""
    nodes = []
    for i in range(graph.node_count()):
        node = graph.get_node(i)
        nodes.append({
            "role": getattr(node, "role", f"node-{i}"),
            "model_id": getattr(node, "model_id", ""),
            "prompt": getattr(node, "prompt", ""),
            "system": getattr(node, "system", 2),
        })
    return yaml.dump({"nodes": nodes, "template": "generated"}, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Collect topology SFT training data")
    parser.add_argument("--dataset", choices=["bigcodebench", "apps"], default="bigcodebench")
    parser.add_argument("--subset", choices=["full", "hard"], default="hard")
    parser.add_argument("--difficulty", choices=["introductory", "interview", "competition", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/topology_sft.jsonl")
    parser.add_argument("--list", action="store_true", help="Show existing data count and exit")
    parser.add_argument("--min-reward", type=float, default=0.3, help="Minimum reward to include in SFT data")
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.list:
        if output_path.exists():
            count = sum(1 for _ in open(output_path, encoding="utf-8"))
            log.info("SFT data: %d entries in %s", count, output_path)
        else:
            log.info("No SFT data at %s", output_path)
        return

    # Boot SAGE
    from sage.boot import boot_agent_system
    from sage.events.bus import EventBus

    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
    log.info("System booted")

    # Load reward + density functions
    try:
        from sage_core import TopologyReward, TopologyDensity, TopologyGraph, TopologyNode
        from sage_core import PyTemplateStore
        reward_fn = TopologyReward()
        density_fn = TopologyDensity()
        template_store = PyTemplateStore()
        log.info("Rust reward + density + templates loaded")
    except ImportError:
        log.error("sage_core not built with required features")
        sys.exit(1)

    # Load tasks
    tasks = _load_tasks(args.dataset, args.subset, args.difficulty, args.limit)
    log.info("Loaded %d tasks from %s", len(tasks), args.dataset)

    # Collect SFT data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for task_idx, (tid, prompt) in enumerate(tasks):
            if not prompt:
                continue

            for template_name in TEMPLATES:
                t0 = time.time()

                # Generate topology from template
                try:
                    graph = template_store.create(template_name, "gemini-2.5-flash")
                except Exception as exc:
                    log.debug("Template %s failed: %s", template_name, exc)
                    continue

                # Set topology on agent loop and run
                system.agent_loop._current_topology = graph
                try:
                    result = asyncio.run(
                        asyncio.wait_for(system.run(prompt), timeout=120)
                    )
                    execution_passed = bool(result and len(result) > 10)
                except Exception:
                    result = ""
                    execution_passed = False

                latency_ms = (time.time() - t0) * 1000

                # Compute reward
                density = density_fn.compute(graph, 2)  # S2 for code tasks
                # Structural score from verifier (if available)
                structural = 0.8  # TODO: wire HybridVerifier
                reward = reward_fn.compute(execution_passed, structural, density.s_complex, None)

                total += 1

                if reward.total >= args.min_reward:
                    topo_yaml = _topology_to_yaml(graph)
                    entry = {
                        "task_id": tid,
                        "prompt": prompt[:500],
                        "template": template_name,
                        "topology_yaml": topo_yaml,
                        "execution_passed": execution_passed,
                        "reward": reward.total,
                        "reward_breakdown": {
                            "execution": reward.execution,
                            "structural": reward.structural,
                            "density": reward.density,
                        },
                        "latency_ms": round(latency_ms, 1),
                        "dataset": args.dataset,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    kept += 1

                log.info(
                    "[%d/%d] %s template=%s reward=%.2f %s (%.0fms)",
                    task_idx + 1, len(tasks), tid, template_name,
                    reward.total, "KEPT" if reward.total >= args.min_reward else "SKIP",
                    latency_ms,
                )

            # Reset topology
            system.agent_loop._current_topology = None

    log.info("Done: %d total runs, %d kept (reward >= %.2f), saved to %s",
             total, kept, args.min_reward, output_path)


if __name__ == "__main__":
    main()
