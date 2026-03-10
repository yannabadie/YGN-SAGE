"""LLM topology synthesis caller -- completes Path 3 in the Rust TopologyEngine.

Calls the LLM with structured prompts to generate:
1. Role assignments (Stage 1 JSON)
2. Structure design (Stage 2 JSON)

Then feeds both JSONs to Rust for graph construction and validation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)

VALID_TEMPLATES = [
    "sequential", "parallel", "avr", "self_moa",
    "hierarchical", "hub", "debate", "brainstorming",
]


def build_role_prompt(
    task: str,
    max_agents: int = 4,
    available_models: list[str] | None = None,
) -> str:
    """Build the Stage 1 prompt: role assignment."""
    models_str = ", ".join(available_models or ["gemini-2.5-flash"])
    return (
        "You are a multi-agent topology designer. Given a task, assign roles to agents.\n\n"
        f"TASK: {task}\n\n"
        f"CONSTRAINTS:\n"
        f"- Maximum {max_agents} agents\n"
        f"- Available models: {models_str}\n"
        "- Each agent needs: name, model, system tier (1=fast, 2=deliberate, 3=formal), capabilities list\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation):\n"
        '{\n  "roles": [\n'
        '    {"name": "agent_name", "model": "model_id", "system": 2, "capabilities": ["cap1"]}\n'
        "  ]\n}"
    )


def build_structure_prompt(roles_json: str) -> str:
    """Build the Stage 2 prompt: structure design."""
    return (
        "Given these agent roles, design the communication structure.\n\n"
        f"ROLES:\n{roles_json}\n\n"
        "Design an adjacency matrix and edge types. Use these edge types:\n"
        '- "control" -- scheduling dependency (A must finish before B starts)\n'
        '- "message" -- data flows from A to B\n'
        '- "state" -- shared state synchronization\n\n'
        f"Choose the best topology template from: {', '.join(VALID_TEMPLATES)}\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation):\n"
        '{\n  "adjacency": [[0, 1], [0, 0]],\n'
        '  "edge_types": [["", "control"], ["", ""]],\n'
        '  "template": "sequential"\n}'
    )


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (may be wrapped in markdown fences)."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("{"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def parse_and_build_topology(
    roles_json: str,
    structure_json: str,
) -> Any:
    """Parse role + structure JSONs and build a TopologyGraph via Rust.

    Returns the TopologyGraph, or None if construction fails.
    """
    try:
        from sage_core import TopologyGraph, TopologyNode, TopologyEdge
    except ImportError:
        _log.warning("sage_core not available, cannot build topology")
        return None

    roles_data = json.loads(roles_json)
    roles = roles_data.get("roles", [])

    struct_data = json.loads(structure_json)
    adjacency = struct_data.get("adjacency", [])
    edge_types = struct_data.get("edge_types", [])
    template = struct_data.get("template", "sequential")

    n = len(roles)
    if len(adjacency) != n:
        _log.error("Dimension mismatch: %d roles but %dx adjacency", n, len(adjacency))
        return None

    graph = TopologyGraph(template)

    for role in roles:
        # TopologyNode(role, model_id, system, required_capabilities,
        #              security_label, max_cost_usd, max_wall_time_s)
        node = TopologyNode(
            role.get("name", "agent"),
            role.get("model", "gemini-2.5-flash"),
            role.get("system", 2),
            role.get("capabilities", []),
            0,     # security_label
            1.0,   # max_cost_usd
            60.0,  # max_wall_time_s
        )
        graph.add_node(node)

    for i in range(n):
        for j in range(n):
            if i < len(adjacency) and j < len(adjacency[i]) and adjacency[i][j] == 1:
                et = ""
                if i < len(edge_types) and j < len(edge_types[i]):
                    et = edge_types[i][j]
                edge = TopologyEdge(et or "control", None, "open", None, 1.0)
                graph.add_edge(i, j, edge)

    _log.info(
        "Built topology from LLM: template=%s, nodes=%d, edges=%d",
        template, graph.node_count(), graph.edge_count(),
    )
    return graph


async def synthesize_topology(
    llm_provider: Any,
    task: str,
    max_agents: int = 4,
    available_models: list[str] | None = None,
) -> Any | None:
    """Full LLM synthesis pipeline: prompt -> LLM -> JSON -> Rust graph.

    Args:
        llm_provider: An LLMProvider instance (GoogleProvider, etc.)
        task: The task description to design a topology for.
        max_agents: Maximum number of agents.
        available_models: List of available model IDs.

    Returns:
        TopologyGraph if synthesis succeeds, None otherwise.
    """
    from sage.llm.base import Message, Role, LLMConfig

    config = LLMConfig(provider="google", model="gemini-2.5-flash")

    # Stage 1: Role assignment
    role_prompt = build_role_prompt(task, max_agents, available_models)
    try:
        response1 = await llm_provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content="You are a JSON-only topology designer."),
                Message(role=Role.USER, content=role_prompt),
            ],
            config=config,
        )
        roles_json = _extract_json(response1.content or "")
        json.loads(roles_json)  # validate parses
    except Exception as e:
        _log.warning("Stage 1 (role assignment) failed: %s", e)
        return None

    # Stage 2: Structure design
    structure_prompt = build_structure_prompt(roles_json)
    try:
        response2 = await llm_provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content="You are a JSON-only topology designer."),
                Message(role=Role.USER, content=structure_prompt),
            ],
            config=config,
        )
        structure_json = _extract_json(response2.content or "")
        json.loads(structure_json)  # validate parses
    except Exception as e:
        _log.warning("Stage 2 (structure design) failed: %s", e)
        return None

    # Stage 3: Build + validate via Rust
    graph = parse_and_build_topology(roles_json, structure_json)
    if graph is None:
        _log.warning("Stage 3 (build) failed")
        return None

    _log.info(
        "LLM synthesis complete: task=%r, nodes=%d, edges=%d",
        task[:60], graph.node_count(), graph.edge_count(),
    )
    return graph
