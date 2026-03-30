#!/usr/bin/env python3
"""Enrich existing topologies with adaptation fields for Phase C training.

Adds to each topology:
- adaptation.checkpoints: node indices where quality can be checked
- adaptation.max_upgrades/max_reroutes: budget for adaptation
- adaptation.quality_threshold: minimum quality to continue
- provider_hint on LLM nodes (based on model_tier)

Rules (from SAGE architecture):
- Checkpoints placed after first coder/solver node (quality gate)
- Complex topologies get more upgrade budget
- Provider hints follow cards.toml conventions

Usage:
    python scripts/enrich_topologies_adaptive.py
    python scripts/enrich_topologies_adaptive.py --input data/topology_sft_v2_toolcall.jsonl --output data/topology_sft_v2_adaptive.jsonl
"""
import argparse
import json
import random

random.seed(42)

# Provider hints based on model_tier (from cards.toml)
TIER_PROVIDERS = {
    "budget": ["deepseek", "openrouter"],
    "fast": ["google", "minimax"],
    "balanced": ["google", "openai"],
    "reasoner": ["google", "openai"],
    "codex": ["openai"],
}

DIFFICULTY_BUDGET = {
    "simple": {"max_upgrades": 1, "max_reroutes": 0, "quality_threshold": 0.6},
    "moderate": {"max_upgrades": 2, "max_reroutes": 1, "quality_threshold": 0.5},
    "complex": {"max_upgrades": 3, "max_reroutes": 2, "quality_threshold": 0.4},
}


def enrich_topology(topology: dict) -> dict:
    """Add adaptation fields to a topology."""
    nodes = topology.get("nodes", [])
    difficulty = topology.get("difficulty", "simple")
    budget = DIFFICULTY_BUDGET.get(difficulty, DIFFICULTY_BUDGET["simple"])

    # Find checkpoint candidates: after coder/solver nodes (not the last node)
    checkpoints = []
    for i, node in enumerate(nodes[:-1]):  # Never checkpoint the last node
        role = node.get("role", "").lower()
        if any(r in role for r in ["coder", "solver", "planner", "writer"]):
            checkpoints.append(i)

    # If no natural checkpoint, place one after the first node (if multi-node)
    if not checkpoints and len(nodes) > 1:
        checkpoints = [0]

    # Add adaptation block
    topology["adaptation"] = {
        "checkpoints": checkpoints,
        "max_upgrades": budget["max_upgrades"],
        "max_reroutes": budget["max_reroutes"],
        "quality_threshold": budget["quality_threshold"],
    }

    # Add provider_hint to nodes
    for node in nodes:
        tier = node.get("model_tier", "budget")
        providers = TIER_PROVIDERS.get(tier, ["deepseek"])
        node["provider_hint"] = random.choice(providers)

    return topology


def main():
    parser = argparse.ArgumentParser(description="Enrich topologies with adaptation fields")
    parser.add_argument("--input", default="data/topology_sft_v2_toolcall.jsonl")
    parser.add_argument("--output", default="data/topology_sft_v2_adaptive.jsonl")
    args = parser.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sage_tool_schemas import wrap_toolcall, TOOLCALL_SYSTEM_PROMPT

    count = 0
    enriched = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            topology = entry.get("topology", {})

            enriched_topo = enrich_topology(topology)
            has_adaptation = len(enriched_topo.get("adaptation", {}).get("checkpoints", [])) > 0

            entry["topology"] = enriched_topo
            entry["topology_toolcall"] = wrap_toolcall(enriched_topo)
            entry["system_prompt"] = TOOLCALL_SYSTEM_PROMPT

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            if has_adaptation:
                enriched += 1

    print(f"Enriched {count} entries ({enriched} with checkpoints)")
    print(f"Output: {args.output}")

    # Stats
    with open(args.output, encoding="utf-8") as f:
        sample = json.loads(f.readline())
    adapt = sample["topology"].get("adaptation", {})
    print(f"Sample adaptation: {json.dumps(adapt)}")
    hint = sample["topology"]["nodes"][0].get("provider_hint", "none")
    print(f"Sample provider_hint: {hint}")


if __name__ == "__main__":
    main()
