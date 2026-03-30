#!/usr/bin/env python3
"""Create a stratified holdout set of 50 prompts from SFT data.

Split: 15 simple, 20 moderate, 15 complex.
These prompts are NEVER used in training — evaluation only.
"""
import json
import random
import sys

random.seed(42)

TARGET = {"simple": 15, "moderate": 20, "complex": 15}
OUTPUT = "experiments/holdout_50.json"
SFT_DATA = "data/topology_sft_v2_combined.jsonl"


def main():
    by_difficulty: dict[str, list] = {"simple": [], "moderate": [], "complex": []}

    with open(SFT_DATA, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            diff = entry.get("difficulty", "simple")
            if diff not in by_difficulty:
                diff = "moderate"
            by_difficulty[diff].append({
                "task_id": entry.get("task_id", ""),
                "prompt": entry["prompt"],
                "difficulty": diff,
                "reference_yaml": entry.get("topology_yaml", ""),
            })

    holdout = []
    for diff, count in TARGET.items():
        pool = by_difficulty[diff]
        if len(pool) < count:
            print(f"WARNING: only {len(pool)} {diff} prompts, need {count}", file=sys.stderr)
            count = len(pool)
        holdout.extend(random.sample(pool, count))

    random.shuffle(holdout)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "count": len(holdout), "prompts": holdout}, f, indent=2)

    print(f"Created {OUTPUT}: {len(holdout)} prompts")
    for diff in TARGET:
        n = sum(1 for h in holdout if h["difficulty"] == diff)
        print(f"  {diff}: {n}")


if __name__ == "__main__":
    main()
