#!/usr/bin/env python3
"""Convert SFT topology data from YAML to JSON format.

The SAGE pipeline parses JSON first, YAML as fallback.
LLMs generate JSON more reliably than YAML (no indentation issues).
Main branch already switched to JSON (commit e129c48).

Usage:
    python scripts/convert_sft_to_json.py
    python scripts/convert_sft_to_json.py --input data/topology_sft_v2_combined.jsonl --output data/topology_sft_v2_json.jsonl
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Convert SFT YAML → JSON")
    parser.add_argument("--input", default="data/topology_sft_v2_combined.jsonl")
    parser.add_argument("--output", default="data/topology_sft_v2_json.jsonl")
    args = parser.parse_args()

    count = 0
    skipped = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            topology = entry.get("topology")
            if not topology or not isinstance(topology, dict):
                skipped += 1
                continue

            # Replace topology_yaml with topology_json
            new_entry = {
                "task_id": entry.get("task_id", ""),
                "prompt": entry["prompt"],
                "topology": topology,
                "topology_json": json.dumps(topology, indent=2),
                "node_count": entry.get("node_count", len(topology.get("nodes", []))),
                "edge_count": entry.get("edge_count", len(topology.get("edges", []))),
                "difficulty": entry.get("difficulty", "simple"),
                "model": entry.get("model", "converted"),
            }
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} entries to JSON ({skipped} skipped)")
    print(f"Output: {args.output}")

    # Stats
    by_diff = {}
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line).get("difficulty", "unknown")
            by_diff[d] = by_diff.get(d, 0) + 1
    for k, v in sorted(by_diff.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
