#!/usr/bin/env python3
"""Convert topology dataset to Nemotron native tool-call format.

Nemotron-Orchestrator-8B generates <tool_call>JSON</tool_call> ONLY when
tools are defined in the chat template. Without tool definitions, it falls
back to <think> mode (Qwen3 default), producing 0% valid output.

Strategy: Bake the full "# Tools" section directly into the system prompt.
This way verl's standard chat template processing works without any special
tool config — the model sees the same <tools> XML it was trained with.
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

TOPOLOGY_TOOL_JSON = json.dumps({
    "type": "function",
    "function": {
        "name": "create_topology",
        "description": "Create a multi-agent topology for the given coding task.",
        "parameters": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex"],
                },
                "reasoning": {"type": "string"},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "model_tier": {
                                "type": "string",
                                "enum": ["fast", "budget", "reasoner", "codex"],
                            },
                            "prompt": {"type": "string"},
                        },
                        "required": ["role", "model_tier", "prompt"],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_idx": {"type": "integer"},
                            "to_idx": {"type": "integer"},
                            "flow_type": {"type": "string"},
                        },
                        "required": ["from_idx", "to_idx"],
                    },
                },
            },
            "required": ["difficulty", "reasoning", "nodes", "edges"],
        },
    },
})

# System prompt with baked-in tool definitions (matches Nemotron chat template)
SYSTEM_PROMPT = f"""You are a multi-agent topology designer for the YGN-SAGE framework. Given a coding task, call the create_topology function to design an optimal agent topology. The LAST node must be a synthesizer.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{TOPOLOGY_TOOL_JSON}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def yaml_to_topology_dict(yaml_str: str) -> dict | None:
    """Parse YAML topology into a dict."""
    try:
        data = yaml.safe_load(yaml_str)
        if isinstance(data, dict) and "nodes" in data:
            return data
    except Exception:
        pass
    return None


def convert_dataset(input_path, output_path):
    df = pd.read_parquet(input_path)
    converted = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        # Extract user message
        prompt = row["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()

        user_content = ""
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg["content"]
                    break

        if not user_content:
            user_content = "Design a topology for a coding task."

        # New prompt with baked-in tool definitions
        new_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        df.at[idx, "prompt"] = new_prompt

        # Convert ground truth YAML → tool_call JSON
        rm = row["reward_model"]
        if isinstance(rm, dict) and "ground_truth" in rm:
            gt_str = rm["ground_truth"]
            topo = yaml_to_topology_dict(gt_str)
            if topo is None:
                # Try JSON
                try:
                    topo = json.loads(gt_str)
                except Exception:
                    topo = None

            if topo and isinstance(topo, dict):
                # Wrap in tool_call format
                tool_call_gt = (
                    '<tool_call>\n'
                    + json.dumps({"name": "create_topology", "arguments": topo})
                    + '\n</tool_call>'
                )
                rm = {**rm, "ground_truth": tool_call_gt}
                df.at[idx, "reward_model"] = rm
                converted += 1

    df.to_parquet(output_path, index=False)
    print(f"Converted {converted}/{len(df)} entries -> {output_path}")


def main():
    base = Path("/workspace/YGN-SAGE/sage-python/data")

    print(f"System prompt length: ~{len(SYSTEM_PROMPT)} chars")
    print(f"Tool section present: {'# Tools' in SYSTEM_PROMPT}")
    print(f"<tools> tag present: {'<tools>' in SYSTEM_PROMPT}")
    print(f"<tool_call> instruction: {'<tool_call>' in SYSTEM_PROMPT}")
    print()

    convert_dataset(
        base / "verl_topology_train.parquet",
        base / "verl_topology_train_toolcall.parquet",
    )
    convert_dataset(
        base / "verl_topology_curated.parquet",
        base / "verl_topology_curated_toolcall.parquet",
    )

    # Verify
    df = pd.read_parquet(base / "verl_topology_train_toolcall.parquet")
    sample = df.iloc[0]
    print(f"\n=== Verification ===")
    prompt = sample["prompt"]
    print(f"Messages: {len(prompt)}")
    print(f"System (first 200): {prompt[0]['content'][:200]}...")
    print(f"User: {prompt[1]['content'][:100]}...")

    rm = sample["reward_model"]
    gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
    print(f"Ground truth (first 200): {gt[:200]}...")
    print(f"Has <tool_call>: {'<tool_call>' in gt}")

    # Estimate token count
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/yann/nemotron_original")
    full_text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    tokens = tok.encode(full_text)
    print(f"Prompt tokens: {len(tokens)}")

    # Show what model would see
    print(f"\n=== Model sees (last 200 chars) ===")
    print(full_text[-200:])


if __name__ == "__main__":
    main()
