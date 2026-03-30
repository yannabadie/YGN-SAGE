#!/usr/bin/env python3
"""Generate Nemotron-native tool-call training dataset for SAGE.

Creates a dataset where Nemotron-Orchestrator-8B learns to use ALL SAGE
capabilities via its native <tool_call> JSON format.

The model learns to orchestrate:
1. TopologyGraph construction (nodes, edges, templates)
2. Model assignment via ModelAssigner
3. Routing via kNN/SystemRouter/ContextualBandit
4. Quality checking via QualityLabeler/HybridVerifier
5. Memory management via S-MMU/WorkingMemory
6. Tool execution via ToolExecutor sandbox
7. Evolution via MAP-Elites/TopologyEngine

Each training example is a (system_prompt, user_task, tool_call_response) tuple.
"""
import json
import random
import hashlib
import pandas as pd
from pathlib import Path

# ─── SAGE Tool Definitions ───────────────────────────────────

SAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_topology",
            "description": "Create a multi-agent topology DAG for a coding task. Design nodes with roles, model tiers, and prompts. Connect with typed edges. The LAST node should synthesize the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_type": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "avr", "selfmoa", "hierarchical", "hub", "debate", "brainstorming"],
                        "description": "Base topology template"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["simple", "moderate", "complex"],
                        "description": "Task difficulty (determines node budget: simple=4, moderate=7, complex=10)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this topology design is optimal"
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "description": "Agent role: planner, coder, reviewer, debugger, synthesizer, researcher, tester, architect"},
                                "model_tier": {"type": "string", "enum": ["fast", "budget", "reasoner", "codex"], "description": "Model tier for this node"},
                                "prompt": {"type": "string", "description": "Detailed instructions for this agent"},
                                "is_checkpoint": {"type": "boolean", "description": "Whether to save state at this node for adaptation"},
                                "fallback_tier": {"type": "string", "enum": ["", "fast", "budget", "reasoner", "codex"], "description": "Fallback model tier if primary fails"}
                            },
                            "required": ["role", "model_tier", "prompt"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_idx": {"type": "integer"},
                                "to_idx": {"type": "integer"},
                                "flow_type": {"type": "string", "enum": ["message", "review", "code", "control", "state"], "description": "Data flow type"}
                            },
                            "required": ["from_idx", "to_idx"]
                        }
                    }
                },
                "required": ["template_type", "difficulty", "reasoning", "nodes", "edges"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_task",
            "description": "Route a task to the appropriate cognitive system (S1=fast/intuitive, S2=deliberate/tools, S3=formal/reasoning) and select the best model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {"type": "integer", "enum": [1, 2, 3], "description": "Cognitive system: S1(simple), S2(moderate), S3(complex)"},
                    "model_id": {"type": "string", "description": "Recommended model ID from cards.toml"},
                    "confidence": {"type": "number", "description": "Routing confidence 0.0-1.0"},
                    "reasoning": {"type": "string", "description": "Why this routing decision"}
                },
                "required": ["system", "model_id", "confidence", "reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_models",
            "description": "Assign specific models to topology nodes based on affinity, domain score, and cost constraints. Score = 0.4*affinity + 0.4*domain + 0.2*(1-cost).",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_idx": {"type": "integer"},
                                "model_id": {"type": "string", "description": "Model from cards.toml (e.g., deepseek-chat, gemini-3.1-pro, gpt-5.4)"},
                                "provider": {"type": "string", "enum": ["google", "openai", "deepseek", "xai", "minimax", "kimi", "openrouter"]},
                                "reason": {"type": "string"}
                            },
                            "required": ["node_idx", "model_id", "provider"]
                        }
                    },
                    "total_budget_usd": {"type": "number"},
                    "domain_hint": {"type": "string"}
                },
                "required": ["assignments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_topology",
            "description": "Run HybridVerifier on a topology: acyclicity, connectivity, fan-in/out limits, security labels, liveness. Returns valid/invalid with errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["acyclicity", "connectivity", "fan_limits", "security", "liveness", "ltl_safety"]},
                        "description": "Which verification checks to run"
                    }
                },
                "required": ["checks"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_topology",
            "description": "Make an adaptation decision for a node based on quality assessment. Actions: continue (quality OK), upgrade_model (quality low), prune_node (quality critical), reroute_topology (structural issue), spawn_subagent (decompose further).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["continue", "upgrade_model", "prune_node", "reroute_topology", "spawn_subagent"]},
                    "target_node": {"type": "integer"},
                    "reason": {"type": "string"},
                    "new_model_id": {"type": "string", "description": "For upgrade_model: the replacement model"},
                    "quality_score": {"type": "number", "description": "Estimated quality 0.0-1.0"}
                },
                "required": ["action", "target_node", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in the 3-layer sandbox (tree-sitter validation → Wasm WASI → subprocess fallback). Returns stdout, stderr, exit_code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout_secs": {"type": "integer", "description": "Execution timeout in seconds (default 30)"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_memory",
            "description": "Manage S-MMU working memory: store events, compact to Arrow, retrieve relevant chunks, or evict old data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add_event", "compact", "retrieve", "evict"]},
                    "event_type": {"type": "string", "description": "For add_event: event type"},
                    "content": {"type": "string", "description": "For add_event: event content"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "For compact: indexing keywords"},
                    "max_hops": {"type": "integer", "description": "For retrieve: BFS depth (default 2)"},
                    "evict_count": {"type": "integer", "description": "For evict: number of chunks to evict"}
                },
                "required": ["operation"]
            }
        }
    }
]

# ─── Task Templates ──────────────────────────────────────────

# Difficulty-based templates
SIMPLE_TASKS = [
    "Write a Python function that {action}.",
    "Create a function to {action}.",
    "Implement {action} in Python.",
    "Write code that {action}.",
]

SIMPLE_ACTIONS = [
    "checks if a number is prime",
    "reverses a string",
    "finds the maximum in a list",
    "calculates factorial of n",
    "checks if a string is a palindrome",
    "counts vowels in a string",
    "sorts a list using bubble sort",
    "converts Celsius to Fahrenheit",
    "finds the GCD of two numbers",
    "generates Fibonacci sequence up to n",
    "flattens a nested list",
    "removes duplicates from a list",
    "finds the second largest element",
    "checks if two strings are anagrams",
    "implements binary search",
    "calculates the sum of digits",
    "merges two sorted lists",
    "rotates a list by k positions",
    "counts word frequency in a string",
    "validates an email address format",
]

MODERATE_TASKS = [
    "Design a {thing} that supports {features}.",
    "Implement a {thing} with {features}.",
    "Build a {thing} that handles {features}.",
    "Create a {thing} supporting {features}.",
]

MODERATE_THINGS = [
    ("LRU cache", "get/put operations with O(1) time complexity"),
    ("rate limiter", "token bucket algorithm with configurable rate"),
    ("JSON parser", "nested objects, arrays, strings, numbers, booleans, and null"),
    ("event emitter", "subscribe, unsubscribe, and emit with wildcard patterns"),
    ("task scheduler", "priority queue with deadline-based scheduling"),
    ("expression evaluator", "arithmetic with parentheses and operator precedence"),
    ("graph class", "BFS, DFS, shortest path, and cycle detection"),
    ("file watcher", "monitoring file changes with debouncing"),
    ("connection pool", "max connections, timeout, and health checks"),
    ("retry decorator", "exponential backoff with configurable max retries"),
    ("simple ORM", "CRUD operations with SQLite backend"),
    ("pub/sub system", "topic-based message routing with acknowledgments"),
    ("circuit breaker", "failure counting, open/closed/half-open states"),
    ("data pipeline", "chained transformations with error handling"),
    ("state machine", "transitions, guards, and callbacks"),
]

COMPLEX_TASKS = [
    "Design and implement {description}.",
    "Build a complete {description}.",
    "Create a production-ready {description}.",
    "Architect and implement {description}.",
]

COMPLEX_DESCRIPTIONS = [
    "a distributed key-value store with consistent hashing, replication factor 3, and vector clock conflict resolution",
    "a SQL query parser and optimizer that handles SELECT, JOIN, WHERE, GROUP BY with query plan generation",
    "a concurrent web crawler with politeness rules, robots.txt parsing, and duplicate detection using bloom filters",
    "a type inference engine for a functional programming language with let-polymorphism and unification",
    "a compiler frontend with lexer, parser, and AST optimizer for a simple expression language",
    "a consensus protocol (Raft) with leader election, log replication, and membership changes",
    "a garbage collector using mark-and-sweep with generational collection and weak references",
    "a neural network framework with automatic differentiation, supporting dense, conv, and LSTM layers",
    "a database query engine with B-tree indices, join algorithms (nested loop, hash, merge), and query optimization",
    "a real-time collaborative text editor using CRDTs with conflict-free concurrent editing",
]

# ─── Topology Generation ────────────────────────────────────

ROLES = ["planner", "coder", "reviewer", "debugger", "synthesizer", "researcher", "tester", "architect"]
MODEL_TIERS = ["fast", "budget", "reasoner", "codex"]
FLOW_TYPES = ["message", "review", "code", "control"]
TEMPLATES = ["sequential", "parallel", "avr", "selfmoa", "hierarchical", "debate"]
MODELS = {
    "fast": ["gemini-3.1-flash-lite", "deepseek-chat"],
    "budget": ["deepseek-chat", "gpt-5.4-mini"],
    "reasoner": ["gemini-3.1-pro", "gpt-5.4"],
    "codex": ["gpt-5.4", "deepseek-chat"],
}
PROVIDERS = {
    "gemini-3.1-flash-lite": "google",
    "gemini-3.1-pro": "google",
    "deepseek-chat": "deepseek",
    "gpt-5.4": "openai",
    "gpt-5.4-mini": "openai",
}


def generate_simple_topology(task: str, seed: int) -> dict:
    """Generate a simple 1-2 node topology."""
    rng = random.Random(seed)
    n_nodes = rng.choice([1, 2])

    if n_nodes == 1:
        nodes = [{"role": "coder", "model_tier": "budget", "prompt": f"Solve: {task}. Return Python code in ```python block."}]
        edges = []
        template = "sequential"
        reasoning = f"Simple task, single coder node sufficient. Budget model handles straightforward coding."
    else:
        roles = rng.sample(["coder", "reviewer"], 2)
        tiers = [rng.choice(["budget", "fast"]), rng.choice(["fast", "budget"])]
        nodes = [
            {"role": roles[0], "model_tier": tiers[0], "prompt": f"Implement: {task}. Write clean Python code."},
            {"role": roles[1], "model_tier": tiers[1], "prompt": f"Review the code. Fix bugs. Return final code in ```python block."},
        ]
        edges = [{"from_idx": 0, "to_idx": 1, "flow_type": "code"}]
        template = "sequential"
        reasoning = f"Simple task benefits from code review. {roles[0]}→{roles[1]} pipeline catches basic errors."

    return {
        "template_type": template,
        "difficulty": "simple",
        "reasoning": reasoning,
        "nodes": nodes,
        "edges": edges,
    }


def generate_moderate_topology(task: str, seed: int) -> dict:
    """Generate a moderate 3-4 node topology."""
    rng = random.Random(seed)
    template = rng.choice(["sequential", "avr", "parallel"])
    n_nodes = rng.choice([3, 4])

    if template == "avr":
        nodes = [
            {"role": "planner", "model_tier": "reasoner", "prompt": f"Analyze the task and plan the implementation: {task}"},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement the plan. Write production-quality Python code."},
            {"role": "reviewer", "model_tier": "reasoner", "prompt": "Review code for correctness, edge cases, and performance. Suggest fixes."},
            {"role": "synthesizer", "model_tier": "budget", "prompt": "Apply review feedback and return final code in ```python block."},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "code"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "review"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "code"},
        ]
        reasoning = "Actor-Verifier-Reviewer pattern: planner designs, coder implements, reviewer validates, synthesizer integrates."
    elif template == "parallel":
        nodes = [
            {"role": "researcher", "model_tier": "reasoner", "prompt": f"Research approaches for: {task}. List pros/cons of each approach."},
            {"role": "coder", "model_tier": "codex", "prompt": f"Implement: {task}. Focus on correctness and clean code."},
            {"role": "tester", "model_tier": "budget", "prompt": "Write comprehensive test cases for the implementation."},
            {"role": "synthesizer", "model_tier": "budget", "prompt": "Combine research insights, code, and tests. Return final code in ```python block."},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 3, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "code"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "message"},
        ]
        reasoning = "Parallel: researcher, coder, tester work independently then synthesizer merges all outputs."
    else:
        nodes = [
            {"role": "planner", "model_tier": "reasoner", "prompt": f"Plan implementation for: {task}"},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement based on plan. Write clean Python code."},
            {"role": "synthesizer", "model_tier": "budget", "prompt": "Review and return final code in ```python block."},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "code"},
        ]
        reasoning = "Sequential pipeline: planner→coder→synthesizer. Reasoner plans, codex implements, budget synthesizes."

    return {
        "template_type": template,
        "difficulty": "moderate",
        "reasoning": reasoning,
        "nodes": nodes[:n_nodes],
        "edges": [e for e in edges if e["from_idx"] < n_nodes and e["to_idx"] < n_nodes],
    }


def generate_complex_topology(task: str, seed: int) -> dict:
    """Generate a complex 4-7 node topology."""
    rng = random.Random(seed)
    template = rng.choice(["avr", "hierarchical", "debate", "selfmoa"])

    if template == "debate":
        nodes = [
            {"role": "architect", "model_tier": "reasoner", "prompt": f"Design architecture for: {task}. Define interfaces and modules."},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement approach A based on architecture.", "is_checkpoint": True},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement approach B with different algorithms.", "is_checkpoint": True},
            {"role": "reviewer", "model_tier": "reasoner", "prompt": "Compare both implementations. Identify strengths and weaknesses."},
            {"role": "tester", "model_tier": "budget", "prompt": "Write test cases covering edge cases and performance."},
            {"role": "synthesizer", "model_tier": "codex", "prompt": "Synthesize best parts of both approaches. Return final code in ```python block.", "fallback_tier": "reasoner"},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "code"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "code"},
            {"from_idx": 3, "to_idx": 5, "flow_type": "review"},
            {"from_idx": 4, "to_idx": 5, "flow_type": "message"},
        ]
        reasoning = "Debate: architect designs, two coders compete, reviewer judges, tester validates, synthesizer picks best."
    elif template == "hierarchical":
        nodes = [
            {"role": "architect", "model_tier": "reasoner", "prompt": f"Decompose: {task}. Define modules and interfaces."},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement core module.", "is_checkpoint": True},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement utility module."},
            {"role": "tester", "model_tier": "budget", "prompt": "Write unit tests for all modules."},
            {"role": "debugger", "model_tier": "reasoner", "prompt": "Fix any test failures. Analyze edge cases."},
            {"role": "synthesizer", "model_tier": "budget", "prompt": "Integrate modules. Return final code in ```python block."},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            {"from_idx": 0, "to_idx": 2, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 3, "flow_type": "code"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "code"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "message"},
            {"from_idx": 4, "to_idx": 5, "flow_type": "code"},
        ]
        reasoning = "Hierarchical: architect decomposes, parallel coders implement modules, tester validates, debugger fixes, synthesizer integrates."
    else:  # avr or selfmoa
        nodes = [
            {"role": "planner", "model_tier": "reasoner", "prompt": f"Plan implementation for: {task}. Consider edge cases and performance."},
            {"role": "coder", "model_tier": "codex", "prompt": "Implement the plan. Production-quality code.", "is_checkpoint": True},
            {"role": "reviewer", "model_tier": "reasoner", "prompt": "Deep code review: correctness, performance, security."},
            {"role": "coder", "model_tier": "codex", "prompt": "Apply review feedback. Fix all issues."},
            {"role": "tester", "model_tier": "budget", "prompt": "Write comprehensive tests."},
            {"role": "synthesizer", "model_tier": "budget", "prompt": "Final integration. Return code in ```python block.", "fallback_tier": "fast"},
        ]
        edges = [
            {"from_idx": 0, "to_idx": 1, "flow_type": "message"},
            {"from_idx": 1, "to_idx": 2, "flow_type": "code"},
            {"from_idx": 2, "to_idx": 3, "flow_type": "review"},
            {"from_idx": 3, "to_idx": 4, "flow_type": "code"},
            {"from_idx": 4, "to_idx": 5, "flow_type": "message"},
        ]
        reasoning = "AVR extended: plan→code→review→fix→test→synthesize. Checkpoints enable adaptation if quality drops."

    return {
        "template_type": template,
        "difficulty": "complex",
        "reasoning": reasoning,
        "nodes": nodes,
        "edges": edges,
    }


def generate_routing_decision(task: str, difficulty: str, seed: int) -> dict:
    """Generate a routing decision for a task."""
    rng = random.Random(seed)
    system_map = {"simple": 1, "moderate": 2, "complex": 3}
    system = system_map[difficulty]

    model_options = {
        1: [("deepseek-chat", "deepseek", 0.9), ("gemini-3.1-flash-lite", "google", 0.85)],
        2: [("gpt-5.4-mini", "openai", 0.8), ("deepseek-chat", "deepseek", 0.75)],
        3: [("gemini-3.1-pro", "google", 0.85), ("gpt-5.4", "openai", 0.9)],
    }
    model_id, provider, conf = rng.choice(model_options[system])

    return {
        "system": system,
        "model_id": model_id,
        "confidence": round(conf + rng.uniform(-0.1, 0.05), 2),
        "reasoning": f"S{system} routing: {'simple task, fast model' if system == 1 else 'moderate complexity, deliberate reasoning' if system == 2 else 'complex task, formal reasoning required'}.",
    }


def generate_model_assignments(topology: dict, seed: int) -> dict:
    """Generate model assignments for a topology."""
    rng = random.Random(seed)
    assignments = []
    for i, node in enumerate(topology["nodes"]):
        tier = node["model_tier"]
        model_id = rng.choice(MODELS.get(tier, ["deepseek-chat"]))
        provider = PROVIDERS.get(model_id, "deepseek")
        assignments.append({
            "node_idx": i,
            "model_id": model_id,
            "provider": provider,
            "reason": f"{node['role']} needs {tier} tier for {'code generation' if node['role'] in ('coder', 'synthesizer') else 'analysis'}"
        })
    return {"assignments": assignments, "total_budget_usd": 0.10 if topology["difficulty"] == "simple" else 0.50 if topology["difficulty"] == "moderate" else 2.00}


def make_tool_call(name: str, arguments: dict) -> str:
    """Format a tool call in Nemotron's native format."""
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def generate_example(difficulty: str, idx: int) -> dict:
    """Generate a single training example with tool calls."""
    seed = int(hashlib.md5(f"{difficulty}_{idx}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Generate task
    if difficulty == "simple":
        action = SIMPLE_ACTIONS[idx % len(SIMPLE_ACTIONS)]
        template = rng.choice(SIMPLE_TASKS)
        task = template.format(action=action)
    elif difficulty == "moderate":
        thing, features = MODERATE_THINGS[idx % len(MODERATE_THINGS)]
        template = rng.choice(MODERATE_TASKS)
        task = template.format(thing=thing, features=features)
    else:
        desc = COMPLEX_DESCRIPTIONS[idx % len(COMPLEX_DESCRIPTIONS)]
        template = rng.choice(COMPLEX_TASKS)
        task = template.format(description=desc)

    # Generate topology
    if difficulty == "simple":
        topo = generate_simple_topology(task, seed)
    elif difficulty == "moderate":
        topo = generate_moderate_topology(task, seed)
    else:
        topo = generate_complex_topology(task, seed)

    # Decide which tool calls to include
    tool_calls = []

    # Always include routing
    routing = generate_routing_decision(task, difficulty, seed)
    tool_calls.append(make_tool_call("route_task", routing))

    # Always include topology creation
    tool_calls.append(make_tool_call("create_topology", topo))

    # Sometimes include model assignment (50% for moderate+complex)
    if difficulty != "simple" and rng.random() < 0.5:
        assignments = generate_model_assignments(topo, seed)
        tool_calls.append(make_tool_call("assign_models", assignments))

    # Sometimes include verification (30% for moderate+complex)
    if difficulty != "simple" and rng.random() < 0.3:
        checks = rng.sample(["acyclicity", "connectivity", "fan_limits", "liveness"], rng.randint(2, 4))
        tool_calls.append(make_tool_call("verify_topology", {"checks": checks}))

    # Sometimes include adaptation (20% for complex)
    if difficulty == "complex" and rng.random() < 0.2:
        action = rng.choice(["continue", "upgrade_model", "continue"])
        adapt = {
            "action": action,
            "target_node": rng.randint(0, len(topo["nodes"]) - 1),
            "reason": "Quality threshold met" if action == "continue" else "Quality below THETA_CRITICAL=0.3, upgrading to reasoner",
        }
        if action == "upgrade_model":
            adapt["new_model_id"] = "gemini-3.1-pro"
            adapt["quality_score"] = round(rng.uniform(0.1, 0.3), 2)
        else:
            adapt["quality_score"] = round(rng.uniform(0.7, 0.95), 2)
        tool_calls.append(make_tool_call("adapt_topology", adapt))

    response = "\n".join(tool_calls)

    # Ground truth is the topology data (for reward function)
    gt = json.dumps(topo)

    return {
        "task": task,
        "difficulty": difficulty,
        "response": response,
        "ground_truth": gt,
        "n_tool_calls": len(tool_calls),
    }


# ─── System Prompt ───────────────────────────────────────────

TOOLS_JSON = json.dumps(SAGE_TOOLS, indent=None)

SYSTEM_PROMPT = f"""You are the YGN-SAGE orchestrator. You design multi-agent topologies and manage the full SAGE pipeline: routing, topology creation, model assignment, verification, adaptation, and execution.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{TOOLS_JSON}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def main():
    print("Generating SAGE tool-call training dataset...")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"Tools defined: {len(SAGE_TOOLS)}")
    print()

    examples = []

    # Generate balanced dataset
    # Simple: 5000, Moderate: 5000, Complex: 3000
    for i in range(5000):
        ex = generate_example("simple", i)
        examples.append(ex)

    for i in range(5000):
        ex = generate_example("moderate", i)
        examples.append(ex)

    for i in range(3000):
        ex = generate_example("complex", i)
        examples.append(ex)

    random.shuffle(examples)

    # Build parquet
    records = []
    for ex in examples:
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["task"]},
        ]
        records.append({
            "data_source": "sage_topology",
            "prompt": prompt,
            "ability": "tool_calling",
            "reward_model": {"ground_truth": ex["ground_truth"], "style": "tool_call"},
            "extra_info": {
                "difficulty": ex["difficulty"],
                "n_tool_calls": ex["n_tool_calls"],
                "source": "generated_v2",
            },
        })

    df = pd.DataFrame(records)

    base = Path("/workspace/YGN-SAGE/sage-python/data")

    # Split: 90% train, 10% val (curated)
    val_size = max(500, len(df) // 10)
    val_df = df.sample(n=val_size, random_state=42)
    train_df = df.drop(val_df.index)

    train_path = base / "verl_topology_train_toolcall.parquet"
    val_path = base / "verl_topology_curated_toolcall.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train: {len(train_df)} examples -> {train_path}")
    print(f"Val:   {len(val_df)} examples -> {val_path}")

    # Stats
    for diff in ["simple", "moderate", "complex"]:
        count = sum(1 for e in examples if e["difficulty"] == diff)
        avg_calls = sum(e["n_tool_calls"] for e in examples if e["difficulty"] == diff) / count
        print(f"  {diff}: {count} examples, avg {avg_calls:.1f} tool calls")

    # Verify
    print(f"\n=== Verification ===")
    sample = train_df.iloc[0]
    print(f"Prompt: {len(sample['prompt'])} messages")
    print(f"System (first 200): {sample['prompt'][0]['content'][:200]}...")
    print(f"User: {sample['prompt'][1]['content']}")
    rm = sample["reward_model"]
    gt = rm.get("ground_truth", "")
    print(f"Ground truth: {gt[:150]}...")

    # Token count estimate
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/yann/nemotron_original")
    full = tok.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True)
    n_tokens = len(tok.encode(full))
    print(f"Prompt tokens: {n_tokens}")

    if n_tokens > 512:
        print(f"WARNING: Prompt is {n_tokens} tokens, exceeds max_prompt_length=512!")
        print(f"  → Need to increase data.max_prompt_length to {((n_tokens // 128) + 1) * 128}")


if __name__ == "__main__":
    main()
