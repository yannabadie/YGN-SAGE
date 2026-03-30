#!/usr/bin/env python3
"""Autoresearch Loop: Autonomous experiment iteration.

Reads the experiment journal, runs training with a config,
evaluates on 3 levels, records results.

Inspired by karpathy/autoresearch: fixed-budget experiments,
structured journal, reproducible configs.

Usage:
    python scripts/autoresearch_loop.py --config experiments/configs/my_config.json --budget 10
    python scripts/autoresearch_loop.py --eval-only --adapter models/local_qwen3_4b_grpo/sft_checkpoint
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("autoresearch")

JOURNAL_PATH = "experiments/journal.jsonl"
CONFIGS_DIR = "experiments/configs"


def read_journal() -> list[dict]:
    entries = []
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def next_experiment_id(journal: list[dict]) -> str:
    max_n = 0
    for entry in journal:
        eid = entry.get("id", "exp-000")
        try:
            n = int(eid.split("-")[1])
            max_n = max(max_n, n)
        except (IndexError, ValueError):
            pass
    return f"exp-{max_n + 1:03d}"


def get_best_n1(journal: list[dict]) -> float:
    best = 0.0
    for entry in journal:
        m = entry.get("metrics", {})
        val = m.get("n1_reward_avg", 0)
        if val > best:
            best = val
    return best


def run_training(config_path: str, budget_min: int, sft_only: bool = False) -> str | None:
    cmd = [sys.executable, "scripts/train_local_qwen3_4b.py",
           "--config", config_path]
    if sft_only:
        cmd.append("--sft-only")

    log.info("Training: %s (budget %d min)", " ".join(cmd), budget_min)
    timeout = budget_min * 60 + 120

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        log.error("Training failed: %s", result.stderr[-500:] if result.stderr else "")
        return None

    config = json.load(open(config_path))
    output_dir = config.get("output", "models/local_qwen3_4b_grpo")
    for sub in ["grpo_checkpoint", "sft_checkpoint"]:
        path = os.path.join(output_dir, sub)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            return path

    return output_dir


def run_n1(adapter_path: str) -> dict | None:
    output = f"experiments/n1_{os.path.basename(adapter_path)}.json"
    cmd = [sys.executable, "scripts/eval_reward_holdout.py",
           "--adapter", adapter_path, "--output", output]

    log.info("N1 eval: %s", adapter_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error("N1 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def run_n2(adapter_path: str, limit: int = 20) -> dict | None:
    output = "experiments/n2_latest.json"
    cmd = [sys.executable, "scripts/eval_masbench_local.py",
           "--adapter", adapter_path, "--limit", str(limit), "--output", output]

    log.info("N2 eval: MASBENCH depth (limit=%d)", limit)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        log.error("N2 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def run_n3(adapter_path: str, limit: int = 20) -> dict | None:
    output = "experiments/n3_latest.json"
    cmd = [sys.executable, "scripts/eval_bigcodebench_local.py",
           "--adapter", adapter_path, "--limit", str(limit), "--output", output]

    log.info("N3 eval: BigCodeBench Hard (limit=%d)", limit)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        log.error("N3 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def record_experiment(entry: dict):
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("Recorded %s -> %s", entry["id"], JOURNAL_PATH)


def run_experiment(args):
    journal = read_journal()
    exp_id = next_experiment_id(journal)
    best_n1 = get_best_n1(journal)
    t0 = time.time()

    log.info("=== Experiment %s ===", exp_id)
    log.info("Hypothesis: %s", args.hypothesis)
    log.info("Config: %s", args.config)
    log.info("Best N1 so far: %.4f", best_n1)

    entry = {
        "id": exp_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": args.phase,
        "hypothesis": args.hypothesis,
        "config": args.config,
        "base_checkpoint": args.adapter,
        "train_budget_min": args.budget,
        "metrics": {},
    }

    adapter_path = args.adapter
    if not args.eval_only:
        adapter_path = run_training(args.config, args.budget, sft_only=args.sft_only)
        if not adapter_path:
            entry["conclusion"] = "FAILED: Training crashed."
            record_experiment(entry)
            return

    n1 = run_n1(adapter_path)
    if n1:
        entry["metrics"]["n1_reward_avg"] = n1["n1_reward_avg"]
        entry["metrics"]["n1_reward_max"] = n1["n1_reward_max"]
        entry["metrics"]["n1_above_03"] = n1["n1_above_03"]
        entry["metrics"]["n1_clipped_ratio"] = n1["n1_clipped_ratio"]
        log.info("N1: avg=%.4f max=%.4f P(>0.3)=%.0f%%",
                 n1["n1_reward_avg"], n1["n1_reward_max"], n1["n1_above_03"] * 100)

    if n1 and n1["n1_reward_avg"] > best_n1:
        log.info("N1 improved (%.4f > %.4f) -> running N2", n1["n1_reward_avg"], best_n1)
        n2 = run_n2(adapter_path)
        if n2:
            entry["metrics"]["n2_masbench_depth"] = n2
    else:
        log.info("N1 did not improve (%.4f <= %.4f) -> skipping N2/N3",
                 n1["n1_reward_avg"] if n1 else 0, best_n1)

    if n1 and n1["n1_reward_avg"] > best_n1 * 1.1:
        log.info("Significant N1 improvement -> running N3")
        n3 = run_n3(adapter_path)
        if n3:
            entry["metrics"]["n3_bigcodebench_hard"] = n3

    elapsed = (time.time() - t0) / 60
    entry["duration_min"] = round(elapsed, 1)

    if not args.eval_only:
        entry["conclusion"] = args.hypothesis
    else:
        entry["conclusion"] = f"Eval-only on {adapter_path}"

    record_experiment(entry)

    log.info("=== Experiment %s complete (%.1f min) ===", exp_id, elapsed)
    log.info("Journal: %d entries total", len(journal) + 1)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch Loop")
    parser.add_argument("--config", default=None, help="Training config JSON")
    parser.add_argument("--hypothesis", default="Baseline evaluation",
                        help="What are we testing?")
    parser.add_argument("--phase", default="0-baseline", help="Roadmap phase")
    parser.add_argument("--budget", type=int, default=60, help="Training budget in minutes")
    parser.add_argument("--adapter", default=None, help="Existing adapter to evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--sft-only", action="store_true", help="SFT phase only")
    args = parser.parse_args()

    if not args.eval_only and not args.config:
        parser.error("--config required unless --eval-only")

    os.makedirs("experiments/configs", exist_ok=True)
    run_experiment(args)


if __name__ == "__main__":
    main()
