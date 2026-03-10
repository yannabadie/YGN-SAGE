"""Test topology events on EventBus."""
import pytest


@pytest.mark.asyncio
async def test_topology_events_emitted():
    """EventBus should receive topology-related data during run."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    events = []
    system.event_bus.subscribe(lambda e: events.append(e))

    await system.run("Hello world")

    if system.topology_engine is not None:
        # Check that TOPOLOGY event was emitted
        topo_events = [e for e in events if getattr(e, "type", "") == "TOPOLOGY"]
        assert len(topo_events) >= 1
        evt = topo_events[0]
        assert "topology_source" in evt.meta
        assert "topology_template" in evt.meta


@pytest.mark.asyncio
async def test_no_topology_events_without_engine():
    """No TOPOLOGY events when engine is None."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    system.topology_engine = None

    events = []
    system.event_bus.subscribe(lambda e: events.append(e))

    await system.run("Hello")

    topo_events = [e for e in events if getattr(e, "type", "") == "TOPOLOGY"]
    assert len(topo_events) == 0
