"""Test topology-aware routing in AgentSystem."""
import pytest


@pytest.mark.asyncio
async def test_run_generates_topology():
    """AgentSystem.run() should call topology_engine.generate() and cache the result."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    result = await system.run("Write a sorting function")
    # After execution, the engine should have cached at least one topology
    assert system.topology_engine.topology_count() >= 1


@pytest.mark.asyncio
async def test_run_without_topology_engine():
    """AgentSystem.run() works normally when topology_engine is None."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    system.topology_engine = None

    result = await system.run("Hello, what is 2+2?")
    assert isinstance(result, str)
    assert len(result) > 0
