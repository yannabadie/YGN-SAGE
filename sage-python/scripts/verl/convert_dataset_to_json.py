#!/usr/bin/env python3
"""Convert YAML topology training data to JSON tool-call format.

Nemotron-Orchestrator-8B natively emits <tool_call>JSON</tool_call>.
YAML caused 91% malformation. JSON matches the model's pretraining.
"""
import json, yaml
import pandas as pd
import numpy as np
from pathlib import Path


def yaml_to_json(yaml_str: str) -> str:
    """Convert YAML topology ground truth to JSON string."""
    try:
        data = yaml.safe_load(yaml_str)
        if isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False)
    except Exception:
        pass
    return yaml_str


def update_system_prompt(prompt_list):
    """Change system prompt from YAML to JSON tool-call format."""
    updated = []
    for msg in prompt_list:
        if isinstance(msg, dict) and msg.get('role') == 'system':
            content = msg['content']
            # Replace YAML references with JSON tool-call
            content = content.replace('as a YAML DAG', 'as a JSON object using tool calls')
            content = content.replace('YAML DAG', 'JSON topology')
            content = content.replace('as a YAML', 'as JSON')
            content = content.replace('YAML format', 'JSON format')
            content = content.replace('YAML', 'JSON')
            msg = {**msg, 'content': content}
        updated.append(msg)
    return updated


def convert_dataset(input_path, output_path):
    df = pd.read_parquet(input_path)
    converted = 0
    for idx in range(len(df)):
        # Convert ground truth YAML -> JSON
        rm = df.iloc[idx]['reward_model']
        if isinstance(rm, dict) and 'ground_truth' in rm:
            json_gt = yaml_to_json(rm['ground_truth'])
            rm = {**rm, 'ground_truth': json_gt}
            df.at[idx, 'reward_model'] = rm
            converted += 1

        # Update system prompt
        prompt = df.iloc[idx]['prompt']
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            df.at[idx, 'prompt'] = update_system_prompt(prompt)

    df.to_parquet(output_path, index=False)
    print(f"Converted {converted}/{len(df)} entries -> {output_path}")


if __name__ == "__main__":
    base = Path("/workspace/YGN-SAGE/sage-python/data")
    convert_dataset(base / "verl_topology_train.parquet", base / "verl_topology_train_json.parquet")
    convert_dataset(base / "verl_topology_curated.parquet", base / "verl_topology_curated_json.parquet")

    # Verify
    df = pd.read_parquet(base / "verl_topology_train_json.parquet")
    sample = df.iloc[0]
    rm = sample['reward_model']
    gt = rm.get('ground_truth', '') if isinstance(rm, dict) else ''
    print(f"\nVerification:")
    print(f"  Format: {'JSON' if gt.startswith('{') else 'YAML'}")
    print(f"  Preview: {gt[:150]}")
    data = json.loads(gt)
    print(f"  Keys: {list(data.keys())}")
    print(f"  Nodes: {len(data.get('nodes', []))}")
