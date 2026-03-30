#!/usr/bin/env python3
"""Filter/resample SFT data by source or difficulty.

Creates data mix variants for ablation experiments.

Usage:
    python scripts/filter_sft_data.py --source BigCodeBench --output experiments/data/sft_code_only.jsonl
    python scripts/filter_sft_data.py --upsample-complex --output experiments/data/sft_complex_heavy.jsonl
"""
import argparse
import json
import random
import os

random.seed(42)

SFT_DATA = "data/topology_sft_v2_combined.jsonl"


def load_all(path: str) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def filter_by_source(entries: list[dict], source: str) -> list[dict]:
    return [e for e in entries if e.get("task_id", "").startswith(source)]


def upsample_complex(entries: list[dict]) -> list[dict]:
    """Upsample: complex 5x, moderate 2x, simple 1x."""
    result = []
    for e in entries:
        diff = e.get("difficulty", "simple")
        if diff == "complex":
            result.extend([e] * 5)
        elif diff == "moderate":
            result.extend([e] * 2)
        else:
            result.append(e)
    random.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter/resample SFT data")
    parser.add_argument("--input", default=SFT_DATA)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=None, help="Filter by task_id prefix")
    parser.add_argument("--upsample-complex", action="store_true",
                        help="Upsample complex 5x, moderate 2x")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    entries = load_all(args.input)
    print(f"Loaded {len(entries)} entries")

    if args.source:
        entries = filter_by_source(entries, args.source)
        print(f"Filtered to {len(entries)} ({args.source})")

    if args.upsample_complex:
        entries = upsample_complex(entries)
        print(f"Upsampled to {len(entries)}")

    if args.max_samples > 0:
        entries = entries[:args.max_samples]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_diff = {}
    for e in entries:
        d = e.get("difficulty", "unknown")
        by_diff[d] = by_diff.get(d, 0) + 1

    print(f"Wrote {len(entries)} entries to {args.output}")
    for k, v in sorted(by_diff.items()):
        print(f"  {k}: {v} ({100*v/len(entries):.0f}%)")


if __name__ == "__main__":
    main()
