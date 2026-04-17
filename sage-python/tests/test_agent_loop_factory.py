"""Tests for per-node AgentLoop factory.

Phase 2: each topology node gets an independent AgentLoop with
role-filtered tools and appropriate validation level.
"""
import pytest
from unittest.mock import MagicMock

from sage.agent_loop_factory import create_node_agent_loop


def _make_tool_registry():
    """Create a mock registry with known tools."""
    registry = MagicMock()
    registry.list_tools.return_value = [
        "execute_bash",
        "create_python_tool",
        "stm_read",
        "stm_write",
        "ltm_recall",
    ]
    return registry


def test_actor_gets_all_tools():
    """Actor nodes should have access to all tools."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0-actor",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are an actor.",
        system_level=2,
    )
    assert loop.config.tools is None  # None = all tools


def test_verifier_gets_limited_tools():
    """Verifier nodes should only get execute_bash + memory tools (H6)."""
    loop = create_node_agent_loop(
        node_role="verifier",
        node_name="node-1-verifier",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a verifier.",
        system_level=2,
    )
    assert loop.config.tools is not None
    assert "execute_bash" in loop.config.tools
    assert "create_python_tool" not in loop.config.tools


def test_verifier_validation_level_zero():
    """H6: verifier nodes must have validation_level=0 to avoid recursive AVR."""
    loop = create_node_agent_loop(
        node_role="verifier",
        node_name="node-1-verifier",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a verifier.",
        system_level=3,
    )
    assert loop.config.validation_level == 0


def test_actor_s3_gets_validation_3():
    """Actor nodes in S3 system should get full Z3 validation."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0-actor",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are an actor.",
        system_level=3,
    )
    assert loop.config.validation_level == 3


def test_actor_s2_gets_validation_2():
    """Actor nodes in S2 system should get AVR validation."""
    loop = create_node_agent_loop(
        node_role="coder",
        node_name="node-0-coder",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a coder.",
        system_level=2,
    )
    assert loop.config.validation_level == 2


def test_skip_routing_set():
    """H1 carryover: _skip_routing must be True (pipeline already routed)."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop._skip_routing is True


def test_topology_cleared():
    """H4 carryover: _current_topology must be None."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop._current_topology is None


def test_independent_instances():
    """H8: two factory calls must produce independent instances (no shared state)."""
    registry = _make_tool_registry()
    provider = MagicMock()

    loop_a = create_node_agent_loop(
        node_role="actor", node_name="a",
        llm_provider=provider, llm_config=MagicMock(),
        tool_registry=registry, system_prompt="A", system_level=2,
    )
    loop_b = create_node_agent_loop(
        node_role="actor", node_name="b",
        llm_provider=provider, llm_config=MagicMock(),
        tool_registry=registry, system_prompt="B", system_level=2,
    )

    assert loop_a is not loop_b
    assert loop_a.working_memory is not loop_b.working_memory
    assert loop_a.config.name != loop_b.config.name


def test_output_formatter_minimal_tools():
    """Output formatter nodes get memory tools only (no code execution)."""
    loop = create_node_agent_loop(
        node_role="output_formatter",
        node_name="node-2-formatter",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="Format the output.",
        system_level=1,
    )
    assert loop.config.tools is not None
    assert "execute_bash" not in loop.config.tools
    assert "create_python_tool" not in loop.config.tools


def test_max_steps_bounded():
    """Per-node loops should have bounded max_steps (lighter than standalone)."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop.config.max_steps <= 30


def test_max_steps_scales_with_system_level():
    """S3 topology nodes need a bigger step budget than S1/S2.

    The flat `max_steps=5` cap produced empty patches on SWE-bench Lite
    (2026-04-17 smoke): planner / coder / synthesizer burned their 5
    steps on execute_bash exploration before reaching a final answer.
    See docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
    """
    def _mk(level):
        return create_node_agent_loop(
            node_role="actor",
            node_name="n",
            llm_provider=MagicMock(),
            llm_config=MagicMock(),
            tool_registry=_make_tool_registry(),
            system_prompt="prompt",
            system_level=level,
        )

    assert _mk(1).config.max_steps == 5
    assert _mk(2).config.max_steps == 10
    assert _mk(3).config.max_steps == 20
    # Monotonic with respect to level.
    s1, s2, s3 = _mk(1).config.max_steps, _mk(2).config.max_steps, _mk(3).config.max_steps
    assert s1 < s2 < s3
