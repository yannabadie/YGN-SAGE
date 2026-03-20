#!/usr/bin/env python3
"""Build combined SFT v2 dataset for topology policy training.

Sources:
  1. data/topology_sft_v2.jsonl          — 2624 entries (P0 fix: synthesizer node)
  2. data/topology_sft_gpt54_complex.jsonl — 144 entries (complex 6-7 node topologies)
  3. data/topology_raft_phase2.jsonl       — 199 entries (RAFT: execution-verified)

Steps:
  1. Load all three files
  2. Deduplicate by task_id (keep highest node_count version)
  3. Downsample GSM8K from ~50% to 20% of final dataset
  4. Print statistics
  5. Save to data/topology_sft_v2_combined.jsonl
"""

import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SOURCES = [
    ("topology_sft_v2.jsonl", "sft_v2"),
    ("topology_sft_gpt54_complex.jsonl", "gpt54_complex"),
    ("topology_raft_phase2.jsonl", "raft_phase2"),
]

OUTPUT = DATA_DIR / "topology_sft_v2_combined.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_node_count(entry: dict) -> int:
    """Get node count — prefer top-level field, fall back to len(nodes)."""
    if "node_count" in entry and entry["node_count"] is not None:
        return int(entry["node_count"])
    return len(entry.get("topology", {}).get("nodes", []))


def get_difficulty(entry: dict) -> str:
    return entry.get("difficulty") or entry.get("topology", {}).get("difficulty", "unknown")


def get_prefix(task_id: str) -> str:
    return task_id.split("/")[0]


def main():
    random.seed(SEED)

    # ── Step 1: Load all sources ──────────────────────────────────────
    all_entries = []
    for fname, source_tag in SOURCES:
        path = DATA_DIR / fname
        entries = load_jsonl(path)
        for e in entries:
            e["_source"] = source_tag  # track provenance
        print(f"Loaded {len(entries):>5} entries from {fname}")
        all_entries.extend(entries)

    print(f"\nTotal loaded (pre-dedup): {len(all_entries)}")

    # ── Step 2: Deduplicate by task_id (keep highest node_count) ──────
    best: dict[str, dict] = {}
    dup_count = 0
    for entry in all_entries:
        tid = entry["task_id"]
        nc = get_node_count(entry)
        if tid not in best:
            best[tid] = entry
        else:
            existing_nc = get_node_count(best[tid])
            if nc > existing_nc:
                best[tid] = entry
                dup_count += 1
            else:
                dup_count += 1

    deduped = list(best.values())
    print(f"Deduplicated: {len(all_entries)} -> {len(deduped)} ({dup_count} duplicates removed)")

    # ── Step 3: Downsample GSM8K to 20% of final dataset ─────────────
    gsm8k = [e for e in deduped if get_prefix(e["task_id"]) == "GSM8K"]
    non_gsm8k = [e for e in deduped if get_prefix(e["task_id"]) != "GSM8K"]

    # Target: GSM8K should be 20% of final → gsm8k_count / total = 0.20
    # So gsm8k_count = 0.20 * (non_gsm8k_count + gsm8k_count)
    # gsm8k_count = 0.20 / 0.80 * non_gsm8k_count = 0.25 * non_gsm8k_count
    target_gsm8k = int(len(non_gsm8k) * 0.25)

    print(f"\nGSM8K downsampling: {len(gsm8k)} -> {target_gsm8k} "
          f"(non-GSM8K: {len(non_gsm8k)}, target ratio: {target_gsm8k / (len(non_gsm8k) + target_gsm8k):.1%})")

    random.shuffle(gsm8k)
    gsm8k_sampled = gsm8k[:target_gsm8k]

    final = non_gsm8k + gsm8k_sampled
    # Shuffle for training
    random.shuffle(final)

    # ── Step 4: Print statistics ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(final)} entries")
    print(f"{'='*60}")

    # Task ID prefix distribution
    prefix_counts = Counter(get_prefix(e["task_id"]) for e in final)
    total = len(final)
    print(f"\n--- Task ID Prefix Distribution ---")
    for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
        print(f"  {prefix:20s}: {count:5d}  ({count/total:5.1%})")

    # Difficulty distribution
    diff_counts = Counter(get_difficulty(e) for e in final)
    print(f"\n--- Difficulty Distribution ---")
    for diff, count in sorted(diff_counts.items(), key=lambda x: -x[1]):
        print(f"  {diff:20s}: {count:5d}  ({count/total:5.1%})")

    # Node count distribution
    nc_counts = Counter(get_node_count(e) for e in final)
    print(f"\n--- Node Count Distribution ---")
    for nc, count in sorted(nc_counts.items()):
        print(f"  {nc:3d} nodes: {count:5d}  ({count/total:5.1%})")

    # Source provenance
    src_counts = Counter(e.get("_source", "unknown") for e in final)
    print(f"\n--- Source Provenance ---")
    for src, count in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:20s}: {count:5d}  ({count/total:5.1%})")

    # ── Step 5: Save ──────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for entry in final:
            # Remove internal tracking field before saving
            out = {k: v for k, v in entry.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
