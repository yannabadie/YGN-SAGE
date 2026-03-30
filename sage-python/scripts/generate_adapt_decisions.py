#!/usr/bin/env python3
"""Generate adapt_topology training data for Phase C.

At each checkpoint in a topology, the model must decide:
- continue: quality is acceptable, proceed to next node
- upgrade: quality too low, upgrade model_tier (budget→fast→reasoner→codex)
- reroute: fundamentally wrong approach, reroute to alternative node

This script generates (context, decision) pairs from existing topologies.
Each entry is a multi-turn conversation:
1. User presents task
2. Model calls create_topology
3. System reports quality at checkpoint
4. Model calls adapt_topology with the right decision

Rules (from SAGE architecture):
- quality >= threshold → continue (most common, ~60%)
- quality < threshold and upgrades_remaining > 0 → upgrade (~25%)
- quality < threshold and no upgrades → reroute if available (~10%)
- quality < 0.2 → always reroute if possible (~5%)

Usage:
    python scripts/generate_adapt_decisions.py
"""
import json
import os
import random
import sys

random.seed(42)

sys.path.insert(0, os.path.dirname(__file__))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT, wrap_toolcall

TIERS_ORDER = ["budget", "fast", "balanced", "reasoner", "codex"]


def next_tier(current: str) -> str | None:
    """Get the next upgrade tier."""
    idx = TIERS_ORDER.index(current) if current in TIERS_ORDER else 0
    if idx < len(TIERS_ORDER) - 1:
        return TIERS_ORDER[idx + 1]
    return None


def generate_decisions(topology: dict) -> list[dict]:
    """Generate adaptation decision examples for a topology's checkpoints."""
    adapt = topology.get("adaptation", {})
    checkpoints = adapt.get("checkpoints", [])
    threshold = adapt.get("quality_threshold", 0.5)
    max_upgrades = adapt.get("max_upgrades", 1)
    max_reroutes = adapt.get("max_reroutes", 0)
    nodes = topology.get("nodes", [])

    if not checkpoints or not nodes:
        return []

    decisions = []
    for cp_idx in checkpoints:
        if cp_idx >= len(nodes):
            continue
        node = nodes[cp_idx]
        tier = node.get("model_tier", "budget")

        # Generate 3 scenarios per checkpoint
        # Scenario 1: Good quality → continue
        quality_good = round(random.uniform(threshold, min(threshold + 0.3, 1.0)), 2)
        decisions.append({
            "checkpoint_node_idx": cp_idx,
            "checkpoint_role": node.get("role", "unknown"),
            "quality_score": quality_good,
            "threshold": threshold,
            "upgrades_remaining": max_upgrades,
            "reroutes_remaining": max_reroutes,
            "decision": {
                "action": "continue",
                "reason": f"Quality {quality_good} >= threshold {threshold}, output acceptable",
            },
        })

        # Scenario 2: Low quality, upgrades available → upgrade
        if max_upgrades > 0 and next_tier(tier):
            quality_low = round(random.uniform(0.2, threshold - 0.01), 2)
            decisions.append({
                "checkpoint_node_idx": cp_idx,
                "checkpoint_role": node.get("role", "unknown"),
                "quality_score": quality_low,
                "threshold": threshold,
                "upgrades_remaining": max_upgrades,
                "reroutes_remaining": max_reroutes,
                "current_tier": tier,
                "upgrade_to": next_tier(tier),
                "decision": {
                    "action": "upgrade",
                    "node_idx": cp_idx,
                    "reason": f"Quality {quality_low} < threshold {threshold}, upgrading {tier} to {next_tier(tier)}",
                },
            })

        # Scenario 3: Very low quality → reroute
        if max_reroutes > 0 and len(nodes) > cp_idx + 1:
            quality_terrible = round(random.uniform(0.0, 0.2), 2)
            decisions.append({
                "checkpoint_node_idx": cp_idx,
                "checkpoint_role": node.get("role", "unknown"),
                "quality_score": quality_terrible,
                "threshold": threshold,
                "upgrades_remaining": 0,
                "reroutes_remaining": max_reroutes,
                "decision": {
                    "action": "reroute",
                    "node_idx": cp_idx,
                    "reason": f"Quality {quality_terrible} critically low, no upgrades left, rerouting",
                },
            })

    return decisions


def format_adapt_toolcall(decision: dict) -> str:
    """Format an adapt_topology tool call."""
    call = {"name": "adapt_topology", "arguments": decision["decision"]}
    return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"


def main():
    input_path = "data/topology_sft_v2_adaptive_toolcall.jsonl"
    output_path = "data/adapt_decisions_toolcall.jsonl"

    count = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            topology = entry.get("topology", {})
            decisions = generate_decisions(topology)

            for dec in decisions:
                # Build multi-turn conversation
                quality_report = (
                    f"Checkpoint at node {dec['checkpoint_node_idx']} "
                    f"(role: {dec['checkpoint_role']}). "
                    f"Quality score: {dec['quality_score']}. "
                    f"Threshold: {dec['threshold']}. "
                    f"Upgrades remaining: {dec['upgrades_remaining']}. "
                    f"Reroutes remaining: {dec['reroutes_remaining']}."
                )

                adapt_entry = {
                    "task_id": f"{entry.get('task_id', 'unknown')}/adapt_{count}",
                    "prompt": quality_report,
                    "topology_toolcall": format_adapt_toolcall(dec),
                    "system_prompt": TOOLCALL_SYSTEM_PROMPT,
                    "tool": "adapt_topology",
                    "decision": dec["decision"],
                    "context": {
                        "original_task": entry.get("prompt", ""),
                        "topology_difficulty": topology.get("difficulty", "simple"),
                        "checkpoint_node_idx": dec["checkpoint_node_idx"],
                    },
                }
                fout.write(json.dumps(adapt_entry, ensure_ascii=False) + "\n")
                count += 1

    # Stats
    actions = {"continue": 0, "upgrade": 0, "reroute": 0}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)["decision"]["action"]
            actions[a] = actions.get(a, 0) + 1

    print(f"Generated {count} adapt_topology decisions")
    print(f"Output: {output_path}")
    for a, n in actions.items():
        print(f"  {a}: {n} ({100*n/count:.0f}%)")


if __name__ == "__main__":
    main()
