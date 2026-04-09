"""Test outcome recording feeds learning loop.

After the system.run() simplification, outcome recording (S-MMU chunks,
MAP-Elites archive) is handled by the pipeline's _stage_learn(), not by
system.run() directly. Mock mode bypasses the pipeline (H9), so no
outcome recording happens in mock mode.

TODO: Add pipeline-level tests that verify _stage_learn() records
outcomes with non-mock providers.
"""
import pytest


@pytest.mark.asyncio
async def test_mock_mode_no_outcome_recording():
    """Mock mode bypasses pipeline — no S-MMU outcome recording.

    Outcome recording is now in pipeline._stage_learn(), which is
    not invoked in mock mode (H9 bypass).
    """
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    initial_chunks = system.topology_engine.smmu_chunk_count()
    await system.run("Write a fibonacci function in Python")

    # Mock mode bypasses pipeline, so no outcome recording
    assert system._last_execution_path == "mock"
    # S-MMU chunk count should not increase (no pipeline _stage_learn)
    assert system.topology_engine.smmu_chunk_count() == initial_chunks


@pytest.mark.asyncio
async def test_mock_mode_no_topology_accumulation():
    """Mock mode does not accumulate topology data across runs."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)

    if system.topology_engine is None:
        pytest.skip("sage_core not compiled")

    initial_chunks = system.topology_engine.smmu_chunk_count()
    for task in ["Write hello world", "Calculate fibonacci", "Sort a list"]:
        await system.run(task)

    # Mock mode bypasses pipeline — no S-MMU accumulation
    assert system._last_execution_path == "mock"
    assert system.topology_engine.smmu_chunk_count() == initial_chunks
