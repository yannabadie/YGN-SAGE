#!/usr/bin/env python3
"""Convert SFT topology data to <tool_call> JSON format.

Wraps each topology in <tool_call>{"name": "create_topology", "arguments": ...}</tool_call>
and bakes the 7 SAGE tool definitions into the system prompt.

Usage:
    python scripts/convert_sft_to_toolcall.py
"""
import argparse
import json
import os
import sys

# Import tool schemas (created by Task 1)
sys.path.insert(0, os.path.dirname(__file__))
try:
    from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT, wrap_toolcall
except ImportError:
    # Fallback if sage_tool_schemas not yet created
    def wrap_toolcall(topology_dict: dict) -> str:
        call = {"name": "create_topology", "arguments": topology_dict}
        return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"
    TOOLCALL_SYSTEM_PROMPT = "You are a multi-agent topology designer for YGN-SAGE."


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to tool-call format")
    parser.add_argument("--input", default="data/topology_sft_v2_combined.jsonl")
    parser.add_argument("--output", default="data/topology_sft_v2_toolcall.jsonl")
    args = parser.parse_args()

    count = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            topology = entry.get("topology")
            if not topology or not isinstance(topology, dict):
                continue

            toolcall_text = wrap_toolcall(topology)

            new_entry = {
                "task_id": entry.get("task_id", ""),
                "prompt": entry["prompt"],
                "topology": topology,
                "topology_toolcall": toolcall_text,
                "system_prompt": TOOLCALL_SYSTEM_PROMPT,
                "node_count": entry.get("node_count", len(topology.get("nodes", []))),
                "edge_count": entry.get("edge_count", len(topology.get("edges", []))),
                "difficulty": entry.get("difficulty", "simple"),
                "model": entry.get("model", "converted"),
            }
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} entries to tool-call format")
    print(f"Output: {args.output}")

    # Verify first entry
    with open(args.output, encoding="utf-8") as f:
        first = json.loads(f.readline())
    assert "<tool_call>" in first["topology_toolcall"], "Missing <tool_call> wrapper"
    assert "create_topology" in first["topology_toolcall"], "Missing tool name"
    print("Verification OK: <tool_call> format valid")


if __name__ == "__main__":
    main()
