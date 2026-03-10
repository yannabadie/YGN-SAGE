"""Test LLM topology synthesis caller."""
import pytest
import json


def test_build_role_prompt():
    """Role assignment prompt should include task and constraints."""
    from sage.topology.llm_caller import build_role_prompt
    prompt = build_role_prompt(
        task="Write a sorting function",
        max_agents=4,
        available_models=["gemini-2.5-flash", "gemini-3.1-pro-preview"],
    )
    assert "sorting" in prompt.lower()
    assert "gemini-2.5-flash" in prompt
    assert "4" in prompt


def test_build_structure_prompt():
    """Structure design prompt should include roles."""
    from sage.topology.llm_caller import build_structure_prompt
    roles_json = json.dumps({
        "roles": [
            {"name": "coder", "model": "m", "system": 2, "capabilities": ["code"]},
            {"name": "reviewer", "model": "m", "system": 2, "capabilities": ["review"]},
        ]
    })
    prompt = build_structure_prompt(roles_json)
    assert "coder" in prompt
    assert "reviewer" in prompt
    assert "adjacency" in prompt.lower()


def test_extract_json_plain():
    """Extract JSON from plain text."""
    from sage.topology.llm_caller import _extract_json
    assert _extract_json('{"key": 1}') == '{"key": 1}'


def test_extract_json_fenced():
    """Extract JSON from markdown fences."""
    from sage.topology.llm_caller import _extract_json
    text = '```json\n{"key": 1}\n```'
    assert _extract_json(text) == '{"key": 1}'


def test_parse_and_build_topology():
    """End-to-end: parse role + structure JSON into TopologyGraph via Rust."""
    try:
        from sage_core import TopologyGraph
    except ImportError:
        pytest.skip("sage_core not compiled")

    from sage.topology.llm_caller import parse_and_build_topology

    roles_json = json.dumps({
        "roles": [
            {"name": "coder", "model": "gemini-2.5-flash", "system": 2, "capabilities": ["code_generation"]},
            {"name": "reviewer", "model": "gemini-3.1-pro-preview", "system": 2, "capabilities": ["code_review"]},
        ]
    })
    structure_json = json.dumps({
        "adjacency": [[0, 1], [0, 0]],
        "edge_types": [["", "control"], ["", ""]],
        "template": "sequential",
    })

    graph = parse_and_build_topology(roles_json, structure_json)
    assert graph is not None
    assert graph.node_count() == 2
    assert graph.edge_count() == 1


def test_parse_dimension_mismatch():
    """Dimension mismatch should return None."""
    try:
        from sage_core import TopologyGraph
    except ImportError:
        pytest.skip("sage_core not compiled")

    from sage.topology.llm_caller import parse_and_build_topology

    roles_json = json.dumps({"roles": [{"name": "a", "model": "m", "system": 1}]})
    structure_json = json.dumps({
        "adjacency": [[0, 1], [0, 0]],
        "edge_types": [["", "control"], ["", ""]],
        "template": "sequential",
    })
    result = parse_and_build_topology(roles_json, structure_json)
    assert result is None
