"""Convert SAGE topology SFT data to veRL parquet format.

veRL expects parquet files with specific schema:
  - data_source: str (dataset identifier)
  - prompt: list[dict] (chat messages format)
  - ability: str (task category)
  - reward_model: dict (ground truth for reward computation)
  - extra_info: dict (metadata: task_id, difficulty, etc.)

Usage:
    python scripts/verl/convert_sft_to_verl.py \
        --input data/topology_sft_v2_combined.jsonl \
        --output data/verl_topology_train.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("convert_verl")

SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "Given a coding task, design an optimal agent topology as a YAML DAG. "
    "Include: difficulty, reasoning, nodes (role + prompt + model_tier), edges (from_idx + to_idx + flow_type). "
    "The LAST node must be a synthesizer that returns the final code in a ```python block."
)


def convert(input_path: str, output_path: str, limit: int | None = None):
    """Convert JSONL SFT data to veRL parquet."""
    entries = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    if limit:
        entries = entries[:limit]

    rows = []
    for entry in entries:
        task_id = entry.get("task_id", "")
        prompt_text = entry.get("prompt", "")
        difficulty = entry.get("difficulty", "moderate")
        topology = entry.get("topology", {})
        topology_yaml = entry.get("topology_yaml", "")

        if not prompt_text:
            continue

        # veRL chat format: list of message dicts
        chat_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        rows.append({
            "data_source": "sage_topology",
            "prompt": chat_prompt,
            "ability": difficulty,
            "reward_model": {
                "style": "rule",
                "ground_truth": topology_yaml,
            },
            "extra_info": {
                "task_id": task_id,
                "difficulty": difficulty,
                "node_count": entry.get("node_count", 0),
                "edge_count": entry.get("edge_count", 0),
                "source": entry.get("source", entry.get("model", "gpt-5.4")),
            },
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    log.info(
        "Converted %d entries → %s (%d KB)",
        len(df), output_path, Path(output_path).stat().st_size // 1024,
    )

    # Stats
    abilities = df["ability"].value_counts()
    log.info("Difficulty distribution:\n%s", abilities.to_string())


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to veRL parquet")
    parser.add_argument("--input", default="data/topology_sft_v2_combined.jsonl")
    parser.add_argument("--output", default="data/verl_topology_train.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
