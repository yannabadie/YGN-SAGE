"""Test outcome recording feeds learning loop."""
import pytest


@pytest.mark.asyncio
async def test_outcome_recorded_after_run():
    """After a successful run, S-MMU should have chunks from outcome recording."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    initial_chunks = system.topology_engine.smmu_chunk_count()
    await system.run("Write a fibonacci function in Python")

    # Outcome recording should have added at least one S-MMU chunk
    assert system.topology_engine.smmu_chunk_count() > initial_chunks


@pytest.mark.asyncio
async def test_multiple_runs_build_knowledge():
    """Multiple runs should accumulate S-MMU chunks."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    for task in ["Write hello world", "Calculate fibonacci", "Sort a list"]:
        await system.run(task)

    assert system.topology_engine.smmu_chunk_count() >= 3
    # topology_count may be < 3 if similar tasks produce the same template
    assert system.topology_engine.topology_count() >= 1
