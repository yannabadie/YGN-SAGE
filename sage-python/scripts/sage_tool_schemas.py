#!/usr/bin/env python3
"""2 SAGE tool schemas for tool-call training.

Only the tools the 4B model needs to learn:
- create_topology: design multi-agent DAGs (Phase A/B)
- adapt_topology: runtime adaptation decisions (Phase C)

The other 5 SAGE modules (routing, model assignment, verification,
execution, memory) are handled by Rust and don't need a learned policy.
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
            "name": "adapt_topology",
            "description": "Runtime adaptation: upgrade model_tier, reroute to different node, or continue execution at a checkpoint.",
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
]

TOOLS_JSON = json.dumps(SAGE_TOOLS, indent=2)

TOOLCALL_SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "You design optimal agent DAG topologies and make runtime adaptation decisions.\n\n"
    f"<tools>\n{TOOLS_JSON}\n</tools>\n\n"
    "For each task, call create_topology with a JSON topology. "
    "Use <tool_call> format."
)


def wrap_toolcall(topology_dict: dict) -> str:
    """Wrap a topology dict as a <tool_call> string."""
    call = {"name": "create_topology", "arguments": topology_dict}
    return f"<tool_call>\n{json.dumps(call, indent=2)}\n</tool_call>"
