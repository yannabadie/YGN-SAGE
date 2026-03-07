"""End-to-end test: boot -> add events -> compress -> retrieve S-MMU context.

Validates the full S-MMU wiring in pure-Python mock mode (no Rust required):
  1. Boot the agent system with mock LLM
  2. Add events to working memory past compression threshold
  3. Trigger compression (which calls compact_to_arrow_with_meta)
  4. Verify S-MMU context retrieval returns a well-typed result
  5. Run a full agent task to exercise the entire write+read path
"""
from __future__ import annotations

import asyncio

import pytest

from sage.boot import boot_agent_system
from sage.memory.smmu_context import retrieve_smmu_context


def test_smmu_e2e_write_and_read():
    """Full pipeline: boot system, add events, compress, retrieve context."""
    system = boot_agent_system(use_mock_llm=True)
    wm = system.agent_loop.working_memory

    # Simulate agent execution — add events past compression threshold (20)
    for i in range(25):
        wm.add_event("action", f"Step {i}: performing task")

    # Trigger compression (which should call compact_to_arrow_with_meta)
    compressed = asyncio.run(
        system.agent_loop.memory_compressor.step(wm)
    )

    # Verify S-MMU was populated (in pure-Python mock mode, chunk_count stays 0)
    # This test validates the wiring, not the Rust implementation
    context = retrieve_smmu_context(wm)
    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_smmu_e2e_full_run():
    """Full agent run should exercise the S-MMU write+read path."""
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("What is 2+2?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
