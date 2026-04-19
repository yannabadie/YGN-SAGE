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


def test_actor_s3_math_gets_validation_3():
    """Actor on S3 + math domain → full Z3 PRM validation. Z3 assertions
    in <think> blocks are MEANINGFUL for math/formal reasoning.
    """
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0-actor",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are an actor.",
        system_level=3,
        task_domain="math",
    )
    assert loop.config.validation_level == 3


def test_actor_s3_code_gets_validation_2():
    """Regression for 2026-04-17 SWE-bench thrashing bug. Pre-fix, S3 +
    code → validation=3 → PRM expects <think> blocks → 17 CEGAR
    failures × 6 RESET_AGENT × 6 SWITCH_MODEL → 0 patches generated.

    Domain gate: code/general S3 tasks degrade to AVR (level 2) so the
    Z3 <think>-block requirement doesn't trigger the thrash loop.
    """
    for domain in ["code", "general", "", "swe_bench"]:
        loop = create_node_agent_loop(
            node_role="actor",
            node_name="node-0-actor",
            llm_provider=MagicMock(),
            llm_config=MagicMock(),
            tool_registry=_make_tool_registry(),
            system_prompt="You are an actor.",
            system_level=3,
            task_domain=domain,
        )
        assert loop.config.validation_level == 2, (
            f"S3+`{domain}` must degrade to AVR (validation=2), "
            f"got {loop.config.validation_level}"
        )


def test_actor_s3_formal_gets_validation_3():
    """Same as math: formal verification benefits from Z3 PRM.
    Substring match handles `formal`, `formal_verification`, etc.
    """
    for domain in ["formal", "formal_verification", "Formal"]:
        loop = create_node_agent_loop(
            node_role="actor",
            node_name="node-0-actor",
            llm_provider=MagicMock(),
            llm_config=MagicMock(),
            tool_registry=_make_tool_registry(),
            system_prompt="You are an actor.",
            system_level=3,
            task_domain=domain,
        )
        assert loop.config.validation_level == 3


def test_actor_s2_gets_validation_2():
    """Actor nodes in S2 system should get AVR validation regardless of domain."""
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


# -- G-series: write-gate wiring regression (2026-04-19) -----------------------
#
# The gate-bypass failure mode (RustCompositeWriteGate built but never
# called) persisted for months because `_gate_allows` returns True when
# `loop.write_gate is None`. If these factory lines regress silently, every
# other test keeps passing and memory writes quietly go ungated again. This
# test is specifically a guard against that silent regression.

def test_factory_wires_write_gate_onto_loop():
    """Regression guard: create_node_agent_loop must propagate the gate+task."""
    gate_sentinel = object()  # Identity check, not a real gate
    llm_cfg = MagicMock()
    llm_cfg.model = "gemini-3.1-pro-preview"

    loop = create_node_agent_loop(
        node_role="actor",
        node_name="test-gate-wire",
        llm_provider=MagicMock(),
        llm_config=llm_cfg,
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=2,
        write_gate=gate_sentinel,
        task_text="fix astropy units parser bug",
    )

    assert loop.write_gate is gate_sentinel, (
        "factory must forward write_gate onto the loop — without this, "
        "phases/act.py falls back to ungated writes and the gate bypass "
        "is silently restored"
    )
    assert loop.gate_current_task == "fix astropy units parser bug"
    # Source tier lookup must run; exact value depends on whether cards.toml
    # is reachable (Rust compiled) — either way, it must be a valid tier.
    assert loop.gate_source_tier in {"reasoner", "fast", "budget", "unknown"}


def test_factory_gate_defaults_to_none_when_omitted():
    """Backward compat: direct callers that don't know about write_gate get None.
    phases/act.py treats None as "allow all writes", preserving legacy behavior."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="legacy-call",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop.write_gate is None
    assert loop.gate_current_task == ""
