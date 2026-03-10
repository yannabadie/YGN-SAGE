"""Test that boot wires TopologyEngine from sage_core."""
import pytest


def test_boot_has_topology_engine():
    """AgentSystem should have a topology_engine attribute after boot."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert hasattr(system, "topology_engine")
    if system.topology_engine is not None:
        assert system.topology_engine.topology_count() == 0
        assert system.topology_engine.archive_cell_count() == 0


def test_boot_has_bandit():
    """AgentSystem should have a bandit attribute after boot."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert hasattr(system, "bandit")


def test_agent_loop_has_topology_engine_attr():
    """AgentLoop should have topology_engine injected by boot."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert hasattr(system.agent_loop, "topology_engine")
