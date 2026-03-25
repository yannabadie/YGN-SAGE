#!/usr/bin/env python3
"""Enrich topology training data with adaptation/checkpoint metadata.

Phase C multi-step training requires topologies with checkpoint nodes where
the model makes decisions (continue/upgrade/reroute). This script takes
existing training data and adds the `adaptation` field to ground truth YAML.

Rules for checkpoint placement:
  - Simple (1-2 nodes): no checkpoints (single-turn, GRPO-equivalent)
  - Moderate (3-4 nodes): checkpoint after first compute node (node 0)
  - Complex (5+ nodes): checkpoints after first and middle nodes

This ensures Phase C training has a mix of:
  - Single-turn episodes (warm-up, GRPO-equivalent)
  - Multi-step episodes (real GiGPO with decisions)

Usage:
    python scripts/verl/enrich_dataset_checkpoints.py \
        --input data/verl_topology_train.parquet \
        --output data/verl_topology_phase_c.parquet \
        --checkpoint-ratio 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("enrich_checkpoints")


def add_checkpoints_to_yaml(yaml_text: str, force: bool = False) -> str:
    """Add adaptation.checkpoints to a YAML topology.

    Args:
        yaml_text: Raw YAML string from ground truth
        force: If True, overwrite existing adaptation metadata

    Returns:
        Modified YAML string with adaptation metadata
    """
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return yaml_text  # can't parse, return as-is

    if not isinstance(data, dict) or "nodes" not in data:
        return yaml_text

    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        return yaml_text

    n_nodes = len(nodes)

    # Already has adaptation and we're not forcing
    if data.get("adaptation") and not force:
        return yaml_text

    difficulty = str(data.get("difficulty", "moderate")).lower()

    # Determine checkpoint placement based on topology size
    if n_nodes <= 2:
        # Simple: no checkpoints (degenerates to single-turn)
        checkpoints = []
        max_upgrades = 0
    elif n_nodes <= 4:
        # Moderate: checkpoint after first compute node
        checkpoints = [0]
        max_upgrades = 1
    else:
        # Complex: checkpoints at first and middle nodes
        mid = n_nodes // 2
        checkpoints = [0, mid - 1]
        max_upgrades = 2

    # Add fallback_tier to checkpoint nodes
    for idx in checkpoints:
        if idx < n_nodes and isinstance(nodes[idx], dict):
            current_tier = nodes[idx].get("model_tier", "fast")
            # Upgrade path: budget→fast→balanced→reasoner→strong
            upgrade_map = {
                "budget": "fast",
                "fast": "balanced",
                "balanced": "reasoner",
                "reasoner": "strong",
                "strong": "strong",
            }
            nodes[idx]["fallback_tier"] = upgrade_map.get(current_tier, "reasoner")

    # Set adaptation metadata
    data["adaptation"] = {
        "checkpoints": checkpoints,
        "max_upgrades": max_upgrades,
        "max_reroutes": 1,
        "quality_threshold": 0.5,
    }

    # Inject provider_hint on ~30% of moderate/complex nodes
    providers = ["google", "openai", "deepseek", "xai", "kimi", "minimax", "openrouter"]
    if difficulty in ("moderate", "complex") and np.random.random() < 0.3:
        for node in nodes:
            if np.random.random() < 0.5 and isinstance(node, dict):
                node["provider_hint"] = np.random.choice(providers)

    return yaml.dump(data, default_flow_style=False, allow_unicode=True)


def process_parquet(input_path: str, output_path: str, checkpoint_ratio: float = 0.5):
    """Process a verl parquet file and add checkpoints to a fraction of entries."""
    df = pd.read_parquet(input_path)
    log.info("Loaded %d entries from %s", len(df), input_path)
    log.info("Columns: %s", list(df.columns))

    modified = 0
    skipped = 0

    for idx in range(len(df)):
        # Only enrich a fraction (Phase C = mix of single-turn + multi-step)
        if np.random.random() > checkpoint_ratio:
            skipped += 1
            continue

        # Get ground truth YAML from reward_model column
        rm = df.at[idx, "reward_model"]
        if isinstance(rm, dict) and "ground_truth" in rm:
            gt_yaml = rm["ground_truth"]
            enriched = add_checkpoints_to_yaml(gt_yaml)
            if enriched != gt_yaml:
                rm["ground_truth"] = enriched
                df.at[idx, "reward_model"] = rm
                modified += 1
            else:
                skipped += 1
        else:
            skipped += 1

    log.info("Modified: %d, Skipped: %d, Total: %d", modified, skipped, len(df))
    df.to_parquet(output_path, index=False)
    log.info("Saved to %s", output_path)

    # Verify
    df_check = pd.read_parquet(output_path)
    n_with_checkpoints = 0
    for _, row in df_check.iterrows():
        rm = row.get("reward_model", {})
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
            try:
                data = yaml.safe_load(gt)
                if isinstance(data, dict):
                    adapt = data.get("adaptation", {})
                    if isinstance(adapt, dict) and adapt.get("checkpoints"):
                        n_with_checkpoints += 1
            except Exception:
                pass
    log.info("Verification: %d/%d entries have checkpoints (%.1f%%)",
             n_with_checkpoints, len(df_check), 100 * n_with_checkpoints / len(df_check))


def main():
    parser = argparse.ArgumentParser(description="Enrich training data with checkpoints for Phase C")
    parser.add_argument("--input", default="data/verl_topology_train.parquet")
    parser.add_argument("--output", default="data/verl_topology_phase_c.parquet")
    parser.add_argument("--checkpoint-ratio", type=float, default=0.5,
                        help="Fraction of entries to enrich with checkpoints (default 0.5)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing adaptation metadata")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    process_parquet(args.input, args.output, args.checkpoint_ratio)


if __name__ == "__main__":
    main()
