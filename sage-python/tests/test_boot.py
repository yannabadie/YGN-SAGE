"""Tests for sage.boot — boot sequence and AgentSystem.run() routing."""
from __future__ import annotations

import pytest


def test_default_validation_level_is_s1():
    """Default validation should be S1 (no AVR) per Sprint 3 evidence."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.config.validation_level == 1


def test_evolution_enabled_when_topology_engine_available():
    """Evolution should be enabled when Rust TopologyEngine is available (SA-3)."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    has_engine = system.agent_loop.topology_engine is not None
    auto_evolve = getattr(system.agent_loop, '_auto_evolve', False)
    # _auto_evolve matches topology_engine availability
    assert auto_evolve == has_engine


@pytest.mark.asyncio
async def test_run_s1_keeps_validation_level_1():
    """S1-routed tasks should keep validation_level=1."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    # Mock router to return S1
    system.metacognition._route_override = None  # ensure no override

    result = await system.run("What is 2+2?")
    # After run, validation level should still be 1 (S1 task)
    assert system.agent_loop.config.validation_level == 1
    assert result  # non-empty response
