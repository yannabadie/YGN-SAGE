"""Test topology events on EventBus.

After the system.run() simplification, topology events are emitted by
the pipeline's _stage_topology(), not by system.run() directly.
Mock mode bypasses the pipeline (H9), so no TOPOLOGY events are
emitted in mock mode. This is correct by design.

TODO: Add pipeline-level tests that verify TOPOLOGY events are emitted
during _stage_topology() with non-mock providers.
"""
import pytest


@pytest.mark.asyncio
async def test_mock_mode_no_topology_events():
    """Mock mode bypasses pipeline — no TOPOLOGY events emitted.

    TOPOLOGY events are now emitted by the pipeline's _stage_topology(),
    which is not invoked in mock mode (H9 bypass).
    """
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    events = []
    system.event_bus.subscribe(lambda e: events.append(e))

    await system.run("Hello world")

    # Mock mode bypasses pipeline, so no TOPOLOGY events
    assert system._last_execution_path == "mock"
    topo_events = [e for e in events if getattr(e, "type", "") == "TOPOLOGY"]
    assert len(topo_events) == 0


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
