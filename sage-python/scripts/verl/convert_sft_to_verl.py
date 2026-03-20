"""Convert SAGE topology SFT data to veRL parquet format.

veRL expects parquet files with specific schema:
  - data_source: str (dataset identifier)
  - prompt: list[dict] (chat messages format)
  - ability: str (task category)
  - reward_model: dict (ground truth for reward computation)
  - extra_info: dict (metadata: task_id, difficulty, etc.)

Loads the main SFT dataset PLUS all GPT-5.4 Pro supplementary data:
  - topology_gpt54_codeforces_gcj.jsonl — 20 complex competition tasks
  - gpt54_deep_reasoning.jsonl — 20 complex with deep reasoning
  - gpt54_simple_calibrated.jsonl — 20 simple calibrated
  - gpt54_error_correction.jsonl — 20 error→correction pairs (uses v2)
  - gpt54_audit.jsonl — 10 original→improved (uses improved)
  - topology_sft_gpt54_pro.jsonl — 60 combined GPT-5.4 Pro
  - topology_raft_phase2.jsonl — 199 execution-verified
  - topology_sft_gpt54_complex.jsonl — 144 complex topologies

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

import yaml
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("convert_verl")

SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "Given a coding task, design an optimal agent topology as a YAML DAG. "
    "Include: difficulty, reasoning, nodes (role + prompt + model_tier), edges (from_idx + to_idx + flow_type). "
    "The LAST node must be a synthesizer that returns the final code in a ```python block."
)

# GPT-5.4 Pro supplementary data files (relative to data/ directory)
GPT54_FILES = [
    # Standard SFT format
    "topology_gpt54_codeforces_gcj.jsonl",
    "gpt54_deep_reasoning.jsonl",
    "gpt54_simple_calibrated.jsonl",
    "topology_sft_gpt54_pro.jsonl",
    "topology_sft_gpt54_complex.jsonl",
    "topology_raft_phase2.jsonl",
]

# Special format files (error correction, audit)
GPT54_CORRECTION = "gpt54_error_correction.jsonl"
GPT54_AUDIT = "gpt54_audit.jsonl"


def _topology_to_yaml(topo: dict) -> str:
    """Convert topology dict to YAML string for ground truth."""
    try:
        return yaml.dump(topo, default_flow_style=False, allow_unicode=True)
    except Exception:
        return json.dumps(topo)


def _make_row(task_id: str, prompt_text: str, difficulty: str,
              topology: dict, topology_yaml: str, source: str,
              node_count: int = 0, edge_count: int = 0) -> dict | None:
    """Create one veRL parquet row."""
    if not prompt_text:
        return None

    if not topology_yaml and topology:
        topology_yaml = _topology_to_yaml(topology)

    if not difficulty and topology:
        difficulty = topology.get("difficulty", "moderate")

    if node_count == 0 and topology:
        node_count = len(topology.get("nodes", []))
    if edge_count == 0 and topology:
        edge_count = len(topology.get("edges", []))

    chat_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]

    return {
        "data_source": "sage_topology",
        "prompt": chat_prompt,
        "ability": difficulty or "moderate",
        "reward_model": {
            "style": "rule",
            "ground_truth": topology_yaml,
        },
        "extra_info": {
            "task_id": task_id,
            "difficulty": difficulty or "moderate",
            "node_count": node_count,
            "edge_count": edge_count,
            "source": source,
        },
    }


def _load_standard_entries(path: Path, source: str) -> list[dict]:
    """Load standard SFT JSONL entries."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            row = _make_row(
                task_id=entry.get("task_id", ""),
                prompt_text=entry.get("prompt", ""),
                difficulty=entry.get("difficulty", ""),
                topology=entry.get("topology", {}),
                topology_yaml=entry.get("topology_yaml", ""),
                source=source,
                node_count=entry.get("node_count", 0),
                edge_count=entry.get("edge_count", 0),
            )
            if row:
                rows.append(row)
    return rows


def _load_correction_entries(path: Path) -> list[dict]:
    """Load error→correction pairs. Uses topology_v2 (corrected) as target."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            v2 = entry.get("topology_v2", {})
            if not v2:
                continue
            row = _make_row(
                task_id=entry.get("task_id", ""),
                prompt_text=entry.get("prompt", ""),
                difficulty=v2.get("difficulty", ""),
                topology=v2,
                topology_yaml="",
                source="gpt54_correction",
            )
            if row:
                rows.append(row)
    return rows


def _load_audit_entries(path: Path) -> list[dict]:
    """Load audit entries. Uses 'improved' topology as target."""
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            improved = entry.get("improved", {})
            if not improved:
                continue
            row = _make_row(
                task_id=entry.get("task_id", ""),
                prompt_text=entry.get("prompt", ""),
                difficulty=improved.get("difficulty", ""),
                topology=improved,
                topology_yaml="",
                source="gpt54_audit",
            )
            if row:
                rows.append(row)
    return rows


def convert(input_path: str, output_path: str, limit: int | None = None):
    """Convert all SFT data sources to veRL parquet."""
    data_dir = Path(input_path).parent

    # 1. Main SFT dataset
    rows = _load_standard_entries(Path(input_path), "sft_v2_combined")
    log.info("Main SFT: %d entries from %s", len(rows), input_path)

    # 2. GPT-5.4 Pro supplementary files
    for fname in GPT54_FILES:
        fpath = data_dir / fname
        new_rows = _load_standard_entries(fpath, fname.replace(".jsonl", ""))
        if new_rows:
            log.info("GPT-5.4 Pro: %d entries from %s", len(new_rows), fname)
            rows.extend(new_rows)

    # 3. Error correction pairs (use v2 as training target)
    correction_rows = _load_correction_entries(data_dir / GPT54_CORRECTION)
    if correction_rows:
        log.info("Corrections: %d entries from %s", len(correction_rows), GPT54_CORRECTION)
        rows.extend(correction_rows)

    # 4. Audit improved entries
    audit_rows = _load_audit_entries(data_dir / GPT54_AUDIT)
    if audit_rows:
        log.info("Audit: %d entries from %s", len(audit_rows), GPT54_AUDIT)
        rows.extend(audit_rows)

    if limit:
        rows = rows[:limit]

    # Deduplicate by task_id (keep last = GPT-5.4 Pro overrides old data)
    seen = {}
    for row in rows:
        tid = row["extra_info"]["task_id"]
        seen[tid] = row
    rows = list(seen.values())

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    log.info(
        "Total: %d unique entries → %s (%d KB)",
        len(df), output_path, Path(output_path).stat().st_size // 1024,
    )

    # Stats
    abilities = df["ability"].value_counts()
    log.info("Difficulty distribution:\n%s", abilities.to_string())

    # Source breakdown
    sources = {}
    for _, row in df.iterrows():
        src = row["extra_info"].get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    log.info("Source breakdown: %s", sources)


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to veRL parquet")
    parser.add_argument("--input", default="data/topology_sft_v2_combined.jsonl")
    parser.add_argument("--output", default="data/verl_topology_train.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
