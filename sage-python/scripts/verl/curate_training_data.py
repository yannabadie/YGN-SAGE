"""Curate ~500 diverse prompts from the full 1965 dataset for execution-mode training.

Removes repetitive GSM8K entries and keeps the most diverse/valuable samples.
Prioritizes: GPT-5.4 Pro data > RAFT (exec-verified) > BigCodeBench > CodeContests.
GSM8K is capped at 50 entries (was 376 = 20% of dataset).

Usage:
    python scripts/verl/curate_training_data.py \
        --input data/verl_topology_train.parquet \
        --output data/verl_topology_curated.parquet \
        --target 500
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("curate")

# Priority order for source selection (higher = kept first)
SOURCE_PRIORITY = {
    "gpt54_correction": 10,      # Error→correction pairs (most valuable)
    "gpt54_audit": 10,           # Improved topologies
    "topology_gpt54_codeforces_gcj": 9,  # Real competition tasks
    "gpt54_deep_reasoning": 9,   # Deep reasoning
    "topology_sft_gpt54_complex": 8,     # Complex topologies
    "topology_sft_gpt54_pro": 8,
    "gpt54_simple_calibrated": 7,
    "topology_raft_phase2": 7,   # Execution-verified
    "sft_v2_combined": 5,        # Base dataset
}

GSM8K_CAP = 50  # Max GSM8K entries (was 376)


def curate(input_path: str, output_path: str, target: int = 500):
    df = pd.read_parquet(input_path)
    log.info("Input: %d entries", len(df))

    # Extract source and task_id from extra_info
    df["_source"] = df["extra_info"].apply(lambda x: x.get("source", "unknown"))
    df["_task_id"] = df["extra_info"].apply(lambda x: x.get("task_id", ""))
    df["_difficulty"] = df["ability"]
    df["_is_gsm8k"] = df["_task_id"].str.startswith("GSM8K")
    df["_priority"] = df["_source"].map(SOURCE_PRIORITY).fillna(3)

    # Step 1: Cap GSM8K
    gsm8k = df[df["_is_gsm8k"]]
    non_gsm8k = df[~df["_is_gsm8k"]]
    if len(gsm8k) > GSM8K_CAP:
        gsm8k = gsm8k.sample(n=GSM8K_CAP, random_state=42)
        log.info("GSM8K capped: %d -> %d", len(df[df["_is_gsm8k"]]), GSM8K_CAP)

    combined = pd.concat([non_gsm8k, gsm8k])

    # Step 2: Sort by priority (highest first), then sample
    combined = combined.sort_values("_priority", ascending=False)

    if len(combined) > target:
        # Keep ALL high-priority sources, sample from low-priority
        high_prio = combined[combined["_priority"] >= 7]
        low_prio = combined[combined["_priority"] < 7]

        remaining = target - len(high_prio)
        if remaining > 0 and len(low_prio) > remaining:
            # Sample low-priority with difficulty stratification
            low_prio = low_prio.groupby("_difficulty", group_keys=False).apply(
                lambda g: g.sample(
                    n=min(len(g), max(1, int(remaining * len(g) / len(low_prio)))),
                    random_state=42,
                )
            )
        combined = pd.concat([high_prio, low_prio.head(remaining)])

    # Drop helper columns
    result = combined.drop(columns=["_source", "_task_id", "_difficulty", "_is_gsm8k", "_priority"])
    result.to_parquet(output_path, index=False)

    log.info("Output: %d entries -> %s", len(result), output_path)

    # Stats
    sources = {}
    for _, row in result.iterrows():
        src = row["extra_info"].get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    log.info("Source breakdown: %s", sources)

    difficulties = result["ability"].value_counts()
    log.info("Difficulty:\n%s", difficulties.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/verl_topology_train.parquet")
    parser.add_argument("--output", default="data/verl_topology_curated.parquet")
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    curate(args.input, args.output, args.target)
