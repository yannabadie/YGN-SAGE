#!/usr/bin/env python3
"""Generate V2 expert topologies using Claude Opus 4.6.

Generates high-quality V2 training data with:
- Complexity-conditioned topologies
- fallback_tier on every node
- Realistic episodic memory context
- Multi-turn sequences (create → checkpoint → adapt)
- Provider-aware model assignments

Uses the OpenAI-compatible API to call Claude or GPT for generation.
"""
import json
import os
import random
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("gen_v2")

# SSL bypass
import urllib3
urllib3.disable_warnings()
import requests
_orig = requests.adapters.HTTPAdapter.send
def _patched(self, req, **kw): kw["verify"] = False; return _orig(self, req, **kw)
requests.adapters.HTTPAdapter.send = _patched

SYSTEM_PROMPT = """You are an expert multi-agent topology designer for the YGN-SAGE framework.
You generate OPTIMAL agent DAG topologies for coding/reasoning tasks.

RULES:
- Use <tool_call> format with create_topology or adapt_topology
- Every node MUST have: role, model_tier, prompt, fallback_tier
- model_tier: budget (cheap), fast (low-latency), balanced (mid), reasoner (strong), codex (best)
- fallback_tier: the NEXT tier up (budget→fast, fast→balanced, balanced→reasoner, reasoner→codex)
- flow_type: message (data), control (ordering), state (shared)
- S1 tasks: 2 nodes min. S2: 2-3 nodes. S3: 3-5 nodes.
- The LAST node must be a synthesizer/reviewer that produces the final answer
- Include checkpoints on critical nodes (where quality matters)
- reasoning field: explain WHY this topology, not just WHAT

AVAILABLE PROVIDERS (7):
- DeepSeek (budget, $0.28/M) — best cost/quality ratio
- Google Gemini (fast/reasoner) — low latency
- OpenAI GPT-5.4 (codex) — best quality
- Grok/xAI (balanced) — 2M context
- Kimi (reasoner) — strong reasoning
- MiniMax (balanced) — self-evolving
- OpenRouter/Qwen (balanced) — Qwen 3.5 Plus"""

TOOLS_JSON = json.dumps([
    {
        "type": "function",
        "function": {
            "name": "create_topology",
            "description": "Design a multi-agent DAG topology to solve a coding task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
                    "reasoning": {"type": "string"},
                    "nodes": {"type": "array", "items": {"type": "object", "properties": {
                        "role": {"type": "string"},
                        "model_tier": {"type": "string", "enum": ["budget", "fast", "balanced", "reasoner", "codex"]},
                        "prompt": {"type": "string"},
                        "fallback_tier": {"type": "string", "enum": ["fast", "balanced", "reasoner", "codex"]},
                    }, "required": ["role", "model_tier", "prompt", "fallback_tier"]}},
                    "edges": {"type": "array", "items": {"type": "object", "properties": {
                        "from_idx": {"type": "integer"}, "to_idx": {"type": "integer"},
                        "flow_type": {"type": "string", "enum": ["message", "control", "state"]},
                    }, "required": ["from_idx", "to_idx", "flow_type"]}},
                    "checkpoints": {"type": "array", "items": {"type": "integer"}},
                    "max_upgrades": {"type": "integer"},
                    "quality_threshold": {"type": "number"},
                },
                "required": ["difficulty", "reasoning", "nodes", "edges", "checkpoints", "max_upgrades", "quality_threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_topology",
            "description": "Runtime adaptation decision at a checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["continue", "upgrade", "reroute"]},
                    "node_idx": {"type": "integer"},
                    "reason": {"type": "string"},
                    "new_tier": {"type": "string"},
                },
                "required": ["action", "reason"],
            },
        },
    },
], indent=2)

FULL_SYSTEM = f"{SYSTEM_PROMPT}\n\n<tools>\n{TOOLS_JSON}\n</tools>"

# Task templates for diverse V2 training data
TASK_TEMPLATES = {
    "simple": [
        "Write a Python function that {action}. Include docstring and type hints.",
        "Implement a function to {action}. Handle edge cases like empty input and None.",
        "Create a utility function that {action}. Return the result, don't print it.",
    ],
    "moderate": [
        "Build a Python class that {action}. Include error handling, logging, and unit tests.",
        "Implement an algorithm to {action}. Optimize for time complexity O(n log n) or better.",
        "Design a data pipeline that {action}. Use async where beneficial.",
    ],
    "complex": [
        "Architect a system that {action}. Consider concurrency, error recovery, and monitoring.",
        "Implement a distributed algorithm that {action}. Handle network partitions and eventual consistency.",
        "Build a production-ready service that {action}. Include rate limiting, caching, circuit breakers, and observability.",
    ],
}

ACTIONS = {
    "simple": [
        "calculates the nth Fibonacci number using memoization",
        "validates an email address using regex",
        "finds all prime numbers up to n using the Sieve of Eratosthenes",
        "converts a Roman numeral string to an integer",
        "checks if a string is a valid palindrome ignoring spaces and punctuation",
        "merges two sorted lists into one sorted list",
        "counts word frequency in a text file",
        "flattens a nested dictionary into dot-notation keys",
        "calculates the Levenshtein distance between two strings",
        "parses a cron expression and returns the next 5 run times",
    ],
    "moderate": [
        "implements a thread-safe LRU cache with TTL expiration",
        "solves the traveling salesman problem for up to 15 cities using dynamic programming",
        "builds a REST API rate limiter using the token bucket algorithm",
        "implements a B-tree with insert, delete, and range query operations",
        "creates a SQL query optimizer that rewrites queries to use indexes",
        "designs a connection pool manager with health checks and automatic recycling",
        "implements a Raft consensus protocol for leader election",
        "builds a streaming JSON parser that handles incomplete data",
        "creates a dependency injection container with lifecycle management",
        "implements a custom profiler that traces function call graphs",
    ],
    "complex": [
        "builds an event-driven microservice framework with saga pattern for distributed transactions",
        "implements a distributed hash table with consistent hashing and virtual nodes",
        "creates a query planner for a columnar database with cost-based optimization",
        "designs a real-time collaborative editing system using CRDTs",
        "builds a compiler for a simple programming language (lexer, parser, codegen to WASM)",
        "implements a distributed tracing system compatible with OpenTelemetry",
        "creates a ML model serving platform with A/B testing, canary deployment, and auto-scaling",
        "builds a graph database query engine supporting Cypher-like queries",
        "implements a network protocol analyzer that can decode multiple protocols in real-time",
        "designs a chaos engineering framework that injects faults and measures system resilience",
    ],
}

EPISODIC_MEMORIES = {
    "simple": [
        "Past episode: 2-node (coder→reviewer), budget+fast tiers, passed on first try. Reward: 0.91. Key insight: simple tasks don't need complex topologies.",
        "Past episode: 2-node sequential, DeepSeek+Gemini providers, solved in 4.2s. Reward: 0.88. Simple validation catches 90% of errors.",
        "Past episode: 1-node budget failed (no verification). Upgraded to 2-node, passed. Lesson: always include a reviewer even for simple tasks.",
    ],
    "moderate": [
        "Past episode: 3-node pipeline (planner→coder→reviewer), balanced+budget+fast tiers. Checkpoint at node 1 triggered upgrade from budget to balanced. Final: passed. Reward: 0.78.",
        "Past episode: 3-node AVR topology, reasoner actor + budget verifier + fast formatter. No upgrades needed. Reward: 0.82. Key: reasoner tier handles moderate tasks without fallback.",
        "Past episode: 4-node with debate between two coders + judge. Budget tiers. Took 45s but high quality. Reward: 0.76. Debate topology improves code quality but adds latency.",
    ],
    "complex": [
        "Past episode: 4-node hierarchical (planner→2 parallel coders→synthesizer). Codex planner, reasoner coders, balanced synthesizer. 2 checkpoints, 1 upgrade triggered. Passed after 120s. Reward: 0.71.",
        "Past episode: 5-node pipeline with formal verification node. Checkpoint at verification node caught type error. Upgrade from balanced to reasoner fixed it. Reward: 0.68. Formal verification is worth the cost on complex tasks.",
        "Past episode: 3-node with codex tier throughout. Expensive ($0.15) but passed on first try. Reward: 0.85. For complex tasks, paying for quality upfront beats iterating with cheap models.",
    ],
}


def call_llm(messages: list[dict], model: str = "deepseek-chat") -> str:
    """Call LLM via OpenAI-compatible API."""
    import httpx

    # Try DeepSeek first (cheapest), then OpenAI
    providers = [
        ("deepseek", os.environ.get("DEEPSEEK_API_KEY", ""), "https://api.deepseek.com/v1", "deepseek-chat"),
        ("openai", os.environ.get("OPENAI_API_KEY", ""), "https://api.openai.com/v1", "gpt-5.4"),
    ]

    for pname, api_key, base_url, default_model in providers:
        if not api_key:
            continue
        try:
            r = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": default_model, "messages": messages, "max_tokens": 2048, "temperature": 0.7},
                verify=False, timeout=60,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning(f"  {pname} failed: {e}")
            continue

    raise RuntimeError("No LLM provider available")


def generate_topology_entry(task: str, difficulty: str, with_memory: bool = True) -> dict | None:
    """Generate a single V2 topology entry using LLM."""
    system = {"simple": 1, "moderate": 2, "complex": 3}[difficulty]

    # Build prompt with complexity prefix
    user_content = f"[Complexity: {difficulty} (S{system})]\n\n{task}"

    # Add episodic memory
    if with_memory and random.random() < 0.7:
        memories = random.sample(EPISODIC_MEMORIES[difficulty], min(2, len(EPISODIC_MEMORIES[difficulty])))
        memory_text = "\n".join(memories)
        user_content = f"{memory_text}\n\n{user_content}"

    messages = [
        {"role": "system", "content": FULL_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    try:
        response = call_llm(messages)
        if "<tool_call>" not in response:
            log.warning(f"  No <tool_call> in response for: {task[:50]}")
            return None

        return {
            "task_id": f"v2_expert/{hashlib.md5(task.encode()).hexdigest()[:8]}",
            "prompt": user_content,
            "topology_toolcall": response.strip(),
            "system_prompt": FULL_SYSTEM,
            "difficulty": difficulty,
            "tool": "create_topology",
            "version": "v2_expert",
        }
    except Exception as e:
        log.warning(f"  LLM call failed: {e}")
        return None


def generate_adapt_entry(topology_entry: dict, quality: float, action: str) -> dict | None:
    """Generate an adaptation decision for a topology."""
    tc = topology_entry.get("topology_toolcall", "")

    # Parse to find checkpoint nodes
    import re
    match = re.search(r'"checkpoints"\s*:\s*\[([^\]]*)\]', tc)
    if not match:
        checkpoints = [0]
    else:
        try:
            checkpoints = json.loads(f"[{match.group(1)}]")
        except:
            checkpoints = [0]

    if not checkpoints:
        return None

    node_idx = random.choice(checkpoints)

    # Find node role
    match = re.search(r'"nodes"\s*:\s*\[(.*?)\]', tc, re.DOTALL)
    role = "worker"
    tier = "budget"
    if match:
        try:
            nodes = json.loads(f"[{match.group(1)}]")
            if node_idx < len(nodes):
                role = nodes[node_idx].get("role", "worker")
                tier = nodes[node_idx].get("model_tier", "budget")
        except:
            pass

    threshold = 0.6
    match_t = re.search(r'"quality_threshold"\s*:\s*([\d.]+)', tc)
    if match_t:
        threshold = float(match_t.group(1))

    # Build checkpoint prompt
    difficulty = topology_entry.get("difficulty", "moderate")
    system = {"simple": 1, "moderate": 2, "complex": 3}.get(difficulty, 2)

    checkpoint_prompt = (
        f"[Complexity: {difficulty} (S{system})]\n\n"
        f"Checkpoint at node {node_idx} (role: {role}). "
        f"Quality score: {quality:.2f}. Threshold: {threshold}. "
        f"Upgrades remaining: {2 if action == 'upgrade' else 1}. "
        f"Reroutes remaining: {1 if action == 'reroute' else 0}."
    )

    # Build decision
    tier_upgrade = {"budget": "fast", "fast": "balanced", "balanced": "reasoner", "reasoner": "codex"}
    if action == "continue":
        decision = {
            "name": "adapt_topology",
            "arguments": {
                "action": "continue",
                "reason": f"Quality {quality:.2f} exceeds threshold {threshold}, node output is acceptable. No upgrade needed.",
            }
        }
    elif action == "upgrade":
        new_tier = tier_upgrade.get(tier, "reasoner")
        decision = {
            "name": "adapt_topology",
            "arguments": {
                "action": "upgrade",
                "node_idx": node_idx,
                "reason": f"Quality {quality:.2f} below threshold {threshold}. Upgrading {role} from {tier} to {new_tier} for better output quality.",
                "new_tier": new_tier,
            }
        }
    else:  # reroute
        decision = {
            "name": "adapt_topology",
            "arguments": {
                "action": "reroute",
                "node_idx": node_idx,
                "reason": f"Quality {quality:.2f} critically low. Rerouting {role} to alternative execution path.",
            }
        }

    return {
        "task_id": f"{topology_entry['task_id']}/adapt_{node_idx}_{action}",
        "prompt": checkpoint_prompt,
        "topology_toolcall": f"<tool_call>\n{json.dumps(decision, indent=2)}\n</tool_call>",
        "system_prompt": topology_entry.get("system_prompt", ""),
        "difficulty": difficulty,
        "tool": "adapt_topology",
        "version": "v2_expert",
    }


def generate_multiturn(topology_entry: dict, adapt_entry: dict) -> dict:
    """Build a multi-turn sequence from topology + adaptation."""
    ctx = topology_entry.get("prompt", "")
    # Extract just the task (after any memory context)
    lines = ctx.split("\n\n")
    task_part = lines[-1] if lines else ctx

    return {
        "task_id": f"{topology_entry['task_id']}/multiturn",
        "difficulty": topology_entry.get("difficulty", "moderate"),
        "system_prompt": topology_entry.get("system_prompt", ""),
        "version": "v2_multiturn_expert",
        "turns": [
            {"role": "user", "content": topology_entry["prompt"]},
            {"role": "assistant", "content": topology_entry["topology_toolcall"]},
            {"role": "user", "content": adapt_entry["prompt"]},
            {"role": "assistant", "content": adapt_entry["topology_toolcall"]},
        ],
    }


import hashlib

def main():
    random.seed(42)
    output_path = Path("data/v2_expert.jsonl")
    entries = []

    # Generate tasks
    all_tasks = []
    for difficulty in ["simple", "moderate", "complex"]:
        for template in TASK_TEMPLATES[difficulty]:
            for action in ACTIONS[difficulty]:
                all_tasks.append((template.format(action=action), difficulty))

    random.shuffle(all_tasks)
    log.info(f"Total task candidates: {len(all_tasks)}")

    # Generate topologies (limit to manageable count)
    limit = int(os.environ.get("V2_LIMIT", "30"))
    # Distribution: 10 simple, 10 moderate, 10 complex
    per_diff = limit // 3
    selected = []
    for diff in ["simple", "moderate", "complex"]:
        diff_tasks = [(t, d) for t, d in all_tasks if d == diff]
        selected.extend(diff_tasks[:per_diff])

    log.info(f"Generating {len(selected)} expert topologies...")
    topology_entries = []

    for i, (task, difficulty) in enumerate(selected):
        log.info(f"  [{i+1}/{len(selected)}] {difficulty}: {task[:60]}...")
        entry = generate_topology_entry(task, difficulty)
        if entry:
            topology_entries.append(entry)
            entries.append(entry)

            # Generate 2-3 adaptation decisions per topology
            # Continue (quality > threshold)
            q_good = round(random.uniform(0.65, 0.95), 2)
            adapt_continue = generate_adapt_entry(entry, q_good, "continue")
            if adapt_continue:
                entries.append(adapt_continue)

            # Upgrade (quality < threshold)
            q_bad = round(random.uniform(0.15, 0.45), 2)
            adapt_upgrade = generate_adapt_entry(entry, q_bad, "upgrade")
            if adapt_upgrade:
                entries.append(adapt_upgrade)

                # Multi-turn: topology → upgrade
                mt = generate_multiturn(entry, adapt_upgrade)
                entries.append(mt)

            # Reroute (quality very low, 50% chance)
            if random.random() < 0.5:
                q_terrible = round(random.uniform(0.05, 0.20), 2)
                adapt_reroute = generate_adapt_entry(entry, q_terrible, "reroute")
                if adapt_reroute:
                    entries.append(adapt_reroute)

        # Rate limit
        time.sleep(0.5)

    # Write
    random.shuffle(entries)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"Written {len(entries)} entries to {output_path}")
    log.info(f"  Topologies: {len(topology_entries)}")
    log.info(f"  Adapt decisions: {sum(1 for e in entries if e.get('tool') == 'adapt_topology')}")
    log.info(f"  Multi-turn: {sum(1 for e in entries if 'multiturn' in e.get('version', ''))}")


if __name__ == "__main__":
    main()
