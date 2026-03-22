#!/usr/bin/env python3
"""Generate adaptive topology training data for GiGPO micro-decisions.

Creates 3 JSONL files:
  - gpt54_adaptive_topologies.jsonl (~120 entries) — topologies with checkpoints + fallback_tier
  - gpt54_static_to_adaptive.jsonl (~60 entries) — static→adaptive migration pairs
  - gpt54_recovery_scenarios.jsonl (~40 entries) — initial + recovered topology pairs

These teach the model WHEN to upgrade/continue/reroute at checkpoint nodes.
The adaptation metadata (checkpoints, max_upgrades, quality_threshold, fallback_tier)
is what SageTopologyEnv uses for micro-decision steps.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# --- Task templates ---
CODING_TASKS = [
    ("Sort array with custom comparator", "simple"),
    ("Implement binary search tree", "moderate"),
    ("Build a REST API endpoint", "moderate"),
    ("Parse and evaluate mathematical expressions", "moderate"),
    ("Implement Dijkstra shortest path", "moderate"),
    ("Build a web scraper with rate limiting", "moderate"),
    ("Implement LRU cache", "moderate"),
    ("Design a pub/sub event system", "moderate"),
    ("Build a markdown parser", "moderate"),
    ("Implement a concurrent task queue", "complex"),
    ("Build a SQL query optimizer", "complex"),
    ("Implement a B-tree with disk persistence", "complex"),
    ("Design a distributed rate limiter", "complex"),
    ("Build a compiler frontend (lexer+parser)", "complex"),
    ("Implement raft consensus protocol", "complex"),
    ("Design a real-time collaborative editor", "complex"),
    ("Build a constraint solver for scheduling", "complex"),
    ("Implement a neural network from scratch", "complex"),
    ("Design a garbage collector", "complex"),
    ("Build a database query planner", "complex"),
    ("Implement topological sort with cycle detection", "moderate"),
    ("Build a regex engine", "moderate"),
    ("Implement A* pathfinding on a grid", "moderate"),
    ("Design a connection pool manager", "moderate"),
    ("Build a JSON schema validator", "moderate"),
    ("Implement consistent hashing", "moderate"),
    ("Design a circuit breaker pattern", "moderate"),
    ("Build a trie with autocomplete", "moderate"),
    ("Implement a bloom filter", "simple"),
    ("Build a simple key-value store", "simple"),
    ("Implement merge sort", "simple"),
    ("Build a stack-based calculator", "simple"),
    ("Implement a priority queue", "simple"),
    ("Design a URL shortener", "simple"),
    ("Build a CSV parser", "simple"),
    ("Implement a linked list", "simple"),
    ("Build a basic HTTP server", "moderate"),
    ("Implement a thread pool", "moderate"),
    ("Design a retry mechanism with backoff", "moderate"),
    ("Build a simple ORM", "complex"),
]

ROLES = ["coder", "reviewer", "synthesizer", "planner", "tester", "debugger", "architect", "optimizer"]
MODEL_TIERS = ["budget", "fast", "reasoner"]
FALLBACK_TIERS = {"budget": "fast", "fast": "reasoner", "reasoner": "reasoner"}

def _make_adaptive_topology(task: str, difficulty: str, n_nodes: int, n_checkpoints: int) -> dict:
    """Build a topology with adaptation metadata."""
    nodes = []
    # First node is always planner for moderate+
    if difficulty != "simple" and n_nodes >= 3:
        nodes.append({
            "role": "planner",
            "prompt": f"Plan a solution for: {task}",
            "model_tier": "fast" if difficulty == "moderate" else "reasoner",
            "fallback_tier": "reasoner",
        })

    # Core coder node
    coder_tier = "budget" if difficulty == "simple" else "fast"
    nodes.append({
        "role": "coder",
        "prompt": f"Implement a solution for: {task}",
        "model_tier": coder_tier,
        "fallback_tier": FALLBACK_TIERS[coder_tier],
    })

    # Additional nodes based on complexity
    if n_nodes >= 3:
        nodes.append({
            "role": "reviewer",
            "prompt": f"Review the code for correctness and edge cases",
            "model_tier": "fast",
            "fallback_tier": "reasoner",
        })

    if n_nodes >= 4:
        nodes.append({
            "role": "tester",
            "prompt": "Write comprehensive test cases",
            "model_tier": "budget",
            "fallback_tier": "fast",
        })

    if n_nodes >= 5:
        role = random.choice(["optimizer", "debugger"])
        nodes.append({
            "role": role,
            "prompt": f"{'Optimize performance' if role == 'optimizer' else 'Debug edge cases'}",
            "model_tier": "fast",
            "fallback_tier": "reasoner",
        })

    # Synthesizer is always last
    nodes.append({
        "role": "synthesizer",
        "prompt": "Produce the final solution in a ```python block",
        "model_tier": "fast",
    })

    # Edges: sequential chain
    edges = [
        {"from_idx": i, "to_idx": i + 1, "flow_type": "message"}
        for i in range(len(nodes) - 1)
    ]

    # Checkpoints: select fragile nodes (coder, reviewer — not synthesizer)
    candidate_checkpoints = [i for i, n in enumerate(nodes) if n["role"] in ("coder", "reviewer", "planner", "tester")]
    checkpoints = sorted(random.sample(candidate_checkpoints, min(n_checkpoints, len(candidate_checkpoints))))

    max_upgrades = min(n_checkpoints, 2)
    quality_threshold = random.choice([0.4, 0.45, 0.5, 0.55, 0.6])

    return {
        "difficulty": difficulty,
        "reasoning": f"Adaptive topology for '{task}' with {len(checkpoints)} checkpoint(s). "
                     f"Fragile nodes get fallback_tier for upgrade. Max {max_upgrades} upgrades allowed.",
        "nodes": nodes,
        "edges": edges,
        "adaptation": {
            "checkpoints": checkpoints,
            "max_upgrades": max_upgrades,
            "quality_threshold": quality_threshold,
        },
    }


def generate_adaptive_topologies() -> list[dict]:
    """Generate ~120 adaptive topology entries."""
    entries = []
    for i, (task, difficulty) in enumerate(CODING_TASKS * 3):  # 3x = 120
        n_nodes = {"simple": random.randint(2, 3), "moderate": random.randint(3, 5), "complex": random.randint(4, 6)}[difficulty]
        n_checkpoints = {"simple": 1, "moderate": random.randint(1, 2), "complex": random.randint(1, 3)}[difficulty]
        topo = _make_adaptive_topology(task, difficulty, n_nodes, n_checkpoints)
        entries.append({
            "task_id": f"adaptive/{i:04d}",
            "prompt": task,
            "difficulty": difficulty,
            "topology": topo,
        })
    return entries


def generate_static_to_adaptive() -> list[dict]:
    """Generate ~60 static→adaptive migration entries."""
    entries = []
    tasks = random.sample(CODING_TASKS, min(30, len(CODING_TASKS)))
    for i, (task, difficulty) in enumerate(tasks * 2):  # 2x = 60
        # Static version: no adaptation, no fallback_tier
        n_nodes = {"simple": 2, "moderate": 3, "complex": 4}[difficulty]
        static_nodes = []
        for j in range(n_nodes):
            role = ["coder", "reviewer", "tester", "synthesizer"][min(j, 3)]
            if j == n_nodes - 1:
                role = "synthesizer"
            static_nodes.append({
                "role": role,
                "prompt": f"{role.capitalize()} for: {task}",
                "model_tier": "fast",
            })
        static_topo = {
            "difficulty": difficulty,
            "reasoning": f"Static topology for '{task}'",
            "nodes": static_nodes,
            "edges": [{"from_idx": j, "to_idx": j + 1, "flow_type": "message"} for j in range(len(static_nodes) - 1)],
        }

        # Adaptive version: add checkpoints + fallback_tier
        adaptive_topo = _make_adaptive_topology(task, difficulty, max(n_nodes, 3), random.randint(1, 2))

        entries.append({
            "task_id": f"static_to_adaptive/{i:04d}",
            "prompt": task,
            "difficulty": difficulty,
            "topology": static_topo,
            "topology_adaptive": adaptive_topo,
        })
    return entries


def generate_recovery_scenarios() -> list[dict]:
    """Generate ~40 recovery scenario entries (initial failed + recovered)."""
    entries = []
    tasks = random.sample(CODING_TASKS, min(40, len(CODING_TASKS)))
    for i, (task, difficulty) in enumerate(tasks):
        n_nodes = {"simple": 2, "moderate": 3, "complex": 4}[difficulty]

        # Initial topology: budget model, likely to fail on complex tasks
        init_nodes = []
        for j in range(n_nodes):
            role = ["coder", "reviewer", "tester", "synthesizer"][min(j, 3)]
            if j == n_nodes - 1:
                role = "synthesizer"
            init_nodes.append({
                "role": role,
                "prompt": f"{role.capitalize()} for: {task}",
                "model_tier": "budget",
            })
        initial_topo = {
            "difficulty": difficulty,
            "reasoning": f"Initial attempt with budget models for '{task}' — may fail on complex logic",
            "nodes": init_nodes,
            "edges": [{"from_idx": j, "to_idx": j + 1, "flow_type": "message"} for j in range(len(init_nodes) - 1)],
        }

        # Recovered topology: upgraded model tiers + more nodes + checkpoints
        recovered_topo = _make_adaptive_topology(
            task, difficulty,
            n_nodes=max(n_nodes + 1, 3),
            n_checkpoints=random.randint(1, 2),
        )

        entries.append({
            "task_id": f"recovery/{i:04d}",
            "prompt": task,
            "difficulty": difficulty,
            "initial_topology": initial_topo,
            "recovered_topology": recovered_topo,
        })
    return entries


def main():
    # 1. Adaptive topologies
    adaptive = generate_adaptive_topologies()
    path = DATA_DIR / "gpt54_adaptive_topologies.jsonl"
    with open(path, "w") as f:
        for entry in adaptive:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Generated {len(adaptive)} adaptive entries -> {path}")

    # 2. Static→Adaptive
    sta = generate_static_to_adaptive()
    path = DATA_DIR / "gpt54_static_to_adaptive.jsonl"
    with open(path, "w") as f:
        for entry in sta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Generated {len(sta)} static→adaptive entries -> {path}")

    # 3. Recovery scenarios
    recovery = generate_recovery_scenarios()
    path = DATA_DIR / "gpt54_recovery_scenarios.jsonl"
    with open(path, "w") as f:
        for entry in recovery:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Generated {len(recovery)} recovery entries -> {path}")

    # Total adaptive count
    total_adaptive = len(adaptive) + len(sta) + len(recovery) * 2  # recovery has 2 entries each
    print(f"\nTotal adaptive entries: {total_adaptive} (target: 260)")


if __name__ == "__main__":
    main()
