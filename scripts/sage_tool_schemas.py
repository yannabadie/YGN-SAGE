#!/usr/bin/env python3
"""7 SAGE tool schemas for tool-call training.

Defines the JSON function schemas that go into the <tools> block
of the system prompt. Matches the Rust+Python SAGE pipeline.
"""
import json

SAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_topology",
            "description": "Design a multi-agent DAG topology to solve a coding task. Choose nodes (role + model_tier + prompt), edges (flow_type), and difficulty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": ["simple", "moderate", "complex"]},
                    "reasoning": {"type": "string", "description": "Why this topology is optimal for the task"},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "model_tier": {"type": "string", "enum": ["budget", "fast", "balanced", "reasoner", "codex"]},
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
                                "flow_type": {"type": "string", "enum": ["message", "control", "state"]},
                            },
                            "required": ["from_idx", "to_idx", "flow_type"],
                        },
                    },
                },
                "required": ["difficulty", "reasoning", "nodes", "edges"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_task",
            "description": "Classify task complexity as S1 (simple), S2 (moderate), or S3 (complex) using kNN routing (92% accuracy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "system": {"type": "string", "enum": ["S1", "S2", "S3"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"},
                },
                "required": ["system", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_models",
            "description": "Map model_tier to real model from cards.toml (affinity 0.4 + domain 0.4 + cost 0.2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_idx": {"type": "integer"},
                                "model_id": {"type": "string"},
                                "provider": {"type": "string"},
                            },
                            "required": ["node_idx", "model_id", "provider"],
                        },
                    },
                },
                "required": ["assignments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_topology",
            "description": "Run HybridVerifier (Rust Z3/OxiZ + LTL temporal checks) on the topology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checks": {"type": "array", "items": {"type": "string", "enum": ["reachability", "acyclicity", "role_coverage", "budget_constraint"]}},
                },
                "required": ["checks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_topology",
            "description": "Runtime adaptation: upgrade model_tier, reroute to different node, or continue execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["continue", "upgrade", "reroute"]},
                    "node_idx": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run code in 3-layer sandbox (tree-sitter, Wasm WASI, subprocess).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string", "enum": ["python", "javascript", "rust"]},
                    "timeout_sec": {"type": "integer", "default": 30},
                },
                "required": ["code", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_memory",
            "description": "S-MMU operations: write to STM, read from episodic/semantic, evict stale entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["write", "read", "evict", "consolidate"]},
                    "tier": {"type": "string", "enum": ["stm", "episodic", "semantic", "causal"]},
                    "content": {"type": "string"},
                },
                "required": ["operation", "tier"],
            },
        },
    },
]

TOOLS_JSON = json.dumps(SAGE_TOOLS, indent=2)

TOOLCALL_SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "You have access to 7 tools that control the SAGE pipeline: topology creation, "
    "task routing, model assignment, verification, adaptation, code execution, and memory management.\n\n"
    f"<tools>\n{TOOLS_JSON}\n</tools>\n\n"
    "For each task, call the appropriate tool(s) using <tool_call> JSON format. "
    "Always start by calling create_topology to design the agent DAG."
)


def wrap_toolcall(topology_dict: dict) -> str:
    """Wrap a topology dict as a <tool_call> string."""
    call = {"name": "create_topology", "arguments": topology_dict}
    return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"
