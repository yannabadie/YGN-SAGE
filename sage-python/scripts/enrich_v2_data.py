#!/usr/bin/env python3
"""Enrich training data for V2: complexity prefix, fallback_tier, multi-turn sequences.

Transforms:
1. [Complexity: S1/S2/S3] prefix on all prompts
2. fallback_tier on every node (budget→fast→balanced→reasoner→codex)
3. Fix "unknown" difficulty on adapt_decisions (from context)
4. Generate multi-turn sequences (create_topology → checkpoint → adapt_topology)
5. Add episodic memory context (simulated past episodes)

Input:  data/phase_c_combined.jsonl + data/adapt_decisions_toolcall.jsonl
Output: data/v2_enriched.jsonl (unified, ready for SFT)
"""
import json
import random
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("enrich_v2")

DIFFICULTY_TO_SYSTEM = {"simple": 1, "moderate": 2, "complex": 3}
SYSTEM_TO_DIFFICULTY = {1: "simple", 2: "moderate", 3: "complex"}

TIER_UPGRADE_PATH = {
    "budget": "fast",
    "fast": "balanced",
    "balanced": "reasoner",
    "reasoner": "codex",
    "codex": "codex",
}


def add_complexity_prefix(prompt: str, difficulty: str) -> str:
    """Add [Complexity: X (SN)] prefix to user prompt."""
    system = DIFFICULTY_TO_SYSTEM.get(difficulty, 2)
    return f"[Complexity: {difficulty} (S{system})]\n\n{prompt}"


def add_fallback_tiers(topology_dict: dict) -> dict:
    """Add fallback_tier to every node that doesn't have one."""
    if "arguments" in topology_dict:
        args = topology_dict["arguments"]
    else:
        args = topology_dict

    for node in args.get("nodes", []):
        if "fallback_tier" not in node or not node["fallback_tier"]:
            tier = node.get("model_tier", "budget")
            node["fallback_tier"] = TIER_UPGRADE_PATH.get(tier, "reasoner")

    return topology_dict


def parse_toolcall(tc_str: str) -> dict | None:
    """Parse <tool_call>JSON</tool_call> string."""
    import re
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tc_str, re.DOTALL)
    if not match:
        match = re.search(r'<tool_call>\s*(\{.*)', tc_str, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def format_toolcall(call_dict: dict) -> str:
    """Format dict as <tool_call>JSON</tool_call>."""
    return f"<tool_call>\n{json.dumps(call_dict, indent=2)}\n</tool_call>"


def generate_episodic_context(difficulty: str, n_episodes: int = 2) -> str:
    """Generate simulated episodic memory context (past similar tasks)."""
    if n_episodes == 0:
        return ""

    templates = {
        "simple": [
            "Past episode (similar): 2-node sequential, budget tier, passed. Reward: 0.85.",
            "Past episode (similar): 2-node DAG, fast tier solver + budget verifier, passed. Reward: 0.92.",
            "Past episode (similar): 1-node budget, failed (insufficient reasoning). Upgraded to 2-node, passed.",
        ],
        "moderate": [
            "Past episode (similar): 3-node pipeline (planner→coder→reviewer), balanced+budget tiers, passed. Reward: 0.78.",
            "Past episode (similar): 3-node AVR, reasoner actor + budget verifier, passed after 1 upgrade. Reward: 0.71.",
            "Past episode (related): 4-node with checkpoint at node 1, upgrade triggered once. Final: passed. Reward: 0.82.",
        ],
        "complex": [
            "Past episode (similar): 4-node pipeline with debate, reasoner+codex tiers, passed after 2 upgrades. Reward: 0.65.",
            "Past episode (similar): 3-node with codex coder, checkpoint at node 0, no upgrades needed. Reward: 0.88.",
            "Past episode (related): 5-node hierarchical, failed on budget tier, succeeded after upgrade to reasoner. Reward: 0.72.",
        ],
    }

    episodes = templates.get(difficulty, templates["moderate"])
    selected = random.sample(episodes, min(n_episodes, len(episodes)))
    return "\n".join(selected)


def enrich_create_topology(entry: dict) -> dict:
    """Enrich a create_topology entry with V2 features."""
    difficulty = entry.get("difficulty", "moderate")
    prompt = entry.get("prompt", "")
    tc_str = entry.get("topology_toolcall", "")

    # 1. Add complexity prefix
    enriched_prompt = add_complexity_prefix(prompt, difficulty)

    # 2. Add episodic memory context (70% of entries get it)
    if random.random() < 0.7:
        n_eps = random.choice([1, 2])
        memory_ctx = generate_episodic_context(difficulty, n_eps)
        enriched_prompt = f"{memory_ctx}\n\n{enriched_prompt}"

    # 3. Add fallback_tier to topology
    tc_dict = parse_toolcall(tc_str)
    if tc_dict:
        tc_dict = add_fallback_tiers(tc_dict)
        enriched_tc = format_toolcall(tc_dict)
    else:
        enriched_tc = tc_str

    return {
        "task_id": entry.get("task_id", ""),
        "prompt": enriched_prompt,
        "topology_toolcall": enriched_tc,
        "system_prompt": entry.get("system_prompt", ""),
        "node_count": entry.get("node_count", 0),
        "edge_count": entry.get("edge_count", 0),
        "difficulty": difficulty,
        "model": entry.get("model", ""),
        "tool": "create_topology",
        "version": "v2",
    }


def enrich_adapt_decision(entry: dict) -> dict:
    """Enrich an adapt_topology entry with V2 features."""
    # Fix unknown difficulty from context
    context = entry.get("context", {})
    if isinstance(context, str):
        context = json.loads(context)

    difficulty = context.get("topology_difficulty", "moderate")
    prompt = entry.get("prompt", "")

    # 1. Add complexity prefix
    enriched_prompt = add_complexity_prefix(prompt, difficulty)

    # 2. Add episodic memory context for adaptation decisions too
    if random.random() < 0.5:
        memory_ctx = generate_episodic_context(difficulty, 1)
        enriched_prompt = f"{memory_ctx}\n\n{enriched_prompt}"

    return {
        "task_id": entry.get("task_id", ""),
        "prompt": enriched_prompt,
        "topology_toolcall": entry.get("topology_toolcall", ""),
        "system_prompt": entry.get("system_prompt", ""),
        "difficulty": difficulty,
        "tool": "adapt_topology",
        "context": json.dumps(context),
        "version": "v2",
    }


def generate_multiturn_sequences(create_entries: list, adapt_entries: list) -> list:
    """Generate multi-turn sequences: create_topology → checkpoint context → adapt_topology.

    Each sequence is a conversation with 2 turns:
    Turn 1: User asks for topology → Assistant creates it
    Turn 2: System reports checkpoint quality → Assistant decides adapt action
    """
    # Group adapt decisions by original task_id
    adapt_by_task = {}
    for entry in adapt_entries:
        tid = entry.get("task_id", "").rsplit("/adapt_", 1)[0]
        adapt_by_task.setdefault(tid, []).append(entry)

    sequences = []
    for create in create_entries:
        tid = create.get("task_id", "")
        if tid not in adapt_by_task:
            continue

        decisions = adapt_by_task[tid]
        if not decisions:
            continue

        # Pick one adaptation decision for the sequence
        decision = random.choice(decisions)
        context = decision.get("context", {})
        if isinstance(context, str):
            context = json.loads(context)

        difficulty = create.get("difficulty", "moderate")
        system = DIFFICULTY_TO_SYSTEM.get(difficulty, 2)

        # Build multi-turn conversation
        seq = {
            "task_id": f"{tid}/multiturn",
            "difficulty": difficulty,
            "system_prompt": create.get("system_prompt", ""),
            "version": "v2_multiturn",
            "turns": [
                {
                    "role": "user",
                    "content": add_complexity_prefix(
                        context.get("original_task", create.get("prompt", "")),
                        difficulty,
                    ),
                },
                {
                    "role": "assistant",
                    "content": create.get("topology_toolcall", ""),
                },
                {
                    "role": "user",
                    "content": decision.get("prompt", ""),
                },
                {
                    "role": "assistant",
                    "content": decision.get("topology_toolcall", ""),
                },
            ],
        }

        # Add episodic memory to first turn (30% of sequences)
        if random.random() < 0.3:
            memory_ctx = generate_episodic_context(difficulty, 2)
            seq["turns"][0]["content"] = f"{memory_ctx}\n\n{seq['turns'][0]['content']}"

        sequences.append(seq)

    return sequences


def enforce_min_nodes(entry: dict) -> dict:
    """For S2/S3 tasks, ensure topology has ≥2 nodes."""
    difficulty = entry.get("difficulty", "moderate")
    if difficulty == "simple":
        return entry

    tc_str = entry.get("topology_toolcall", "")
    tc_dict = parse_toolcall(tc_str)
    if not tc_dict:
        return entry

    args = tc_dict.get("arguments", tc_dict)
    nodes = args.get("nodes", [])

    if len(nodes) < 2:
        # Add a reviewer node
        tier = nodes[0].get("model_tier", "budget") if nodes else "budget"
        reviewer_tier = "budget" if tier != "budget" else "fast"
        nodes.append({
            "role": "reviewer",
            "model_tier": reviewer_tier,
            "prompt": "Review the output for correctness, completeness, and edge cases. Provide the final verified answer.",
            "fallback_tier": TIER_UPGRADE_PATH.get(reviewer_tier, "balanced"),
        })
        args["nodes"] = nodes

        # Add edge if missing
        edges = args.get("edges", [])
        if not any(e.get("to_idx") == len(nodes) - 1 for e in edges):
            edges.append({"from_idx": 0, "to_idx": 1, "flow_type": "message"})
            args["edges"] = edges

        entry["topology_toolcall"] = format_toolcall(tc_dict)
        entry["node_count"] = len(nodes)
        entry["edge_count"] = len(edges)

    return entry


def main():
    random.seed(42)
    data_dir = Path("data")
    output_path = data_dir / "v2_enriched.jsonl"

    # Load create_topology entries
    log.info("Loading create_topology entries...")
    create_entries = []
    with open(data_dir / "phase_c_combined.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id", "").endswith("/adapt_"):
                continue
            if "adapt" in d.get("task_id", "").split("/")[-1]:
                continue
            create_entries.append(d)

    # Separate adapt_decisions (they're in phase_c_combined with task_id containing /adapt_)
    adapt_from_combined = []
    with open(data_dir / "phase_c_combined.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "/adapt_" in d.get("task_id", ""):
                adapt_from_combined.append(d)

    # Load dedicated adapt_decisions file
    log.info("Loading adapt_decisions...")
    adapt_entries = []
    with open(data_dir / "adapt_decisions_toolcall.jsonl", encoding="utf-8") as f:
        for line in f:
            adapt_entries.append(json.loads(line))

    log.info(f"  create_topology: {len(create_entries)}")
    log.info(f"  adapt_decisions (dedicated): {len(adapt_entries)}")
    log.info(f"  adapt_decisions (from combined): {len(adapt_from_combined)}")

    # Enrich create_topology
    log.info("Enriching create_topology entries...")
    enriched_create = []
    for entry in create_entries:
        enriched = enrich_create_topology(entry)
        enriched = enforce_min_nodes(enriched)
        enriched_create.append(enriched)

    # Enrich adapt_decisions
    log.info("Enriching adapt_decisions...")
    enriched_adapt = []
    for entry in adapt_entries:
        enriched_adapt.append(enrich_adapt_decision(entry))

    # Generate multi-turn sequences
    log.info("Generating multi-turn sequences...")
    multiturn = generate_multiturn_sequences(create_entries, adapt_entries)
    log.info(f"  Generated {len(multiturn)} multi-turn sequences")

    # Write output
    all_entries = enriched_create + enriched_adapt + multiturn
    random.shuffle(all_entries)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"Written {len(all_entries)} entries to {output_path}")

    # Stats
    n_with_prefix = sum(1 for e in all_entries if "[Complexity:" in e.get("prompt", ""))
    n_with_memory = sum(1 for e in all_entries if "Past episode" in e.get("prompt", ""))
    n_with_fallback = sum(1 for e in all_entries if "fallback_tier" in e.get("topology_toolcall", ""))
    n_multiturn = sum(1 for e in all_entries if e.get("version") == "v2_multiturn")
    n_min2_enforced = sum(1 for e in all_entries
                         if e.get("difficulty") in ("moderate", "complex")
                         and e.get("node_count", 0) >= 2
                         and e.get("tool") == "create_topology")

    log.info("=== V2 Enrichment Stats ===")
    log.info(f"  Total entries: {len(all_entries)}")
    log.info(f"  [Complexity:] prefix: {n_with_prefix}")
    log.info(f"  Episodic memory context: {n_with_memory}")
    log.info(f"  fallback_tier present: {n_with_fallback}")
    log.info(f"  Multi-turn sequences: {n_multiturn}")
    log.info(f"  Min 2 nodes (S2/S3): {n_min2_enforced}")


if __name__ == "__main__":
    main()
