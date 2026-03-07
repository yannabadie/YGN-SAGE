"""Tests for S-MMU context injection into agent loop THINK phase (Task 6).

Verifies that the agent loop integrates S-MMU retrieval without breaking
the existing execution flow.
"""
from __future__ import annotations

import sys
import types

# Ensure sage_core mock exists before importing agent_loop
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

_mock_core = sys.modules["sage_core"]

if not hasattr(_mock_core, "WorkingMemory"):

    class _MockMemoryEvent:
        def __init__(self, id, event_type, content, timestamp_str, is_summary=False):
            self.id = id
            self.event_type = event_type
            self.content = content
            self.timestamp_str = timestamp_str
            self.is_summary = is_summary

    class _MockWorkingMemory:
        def __init__(self, agent_id, parent_id=None):
            self.agent_id = agent_id
            self.parent_id = parent_id
            self._events = []
            self._counter = 0
            self._children = []

        def add_event(self, event_type, content):
            self._counter += 1
            eid = f"evt-{self._counter}"
            import time
            self._events.append(_MockMemoryEvent(
                id=eid, event_type=event_type, content=content,
                timestamp_str=str(time.time()),
            ))
            return eid

        def get_event(self, event_id):
            for e in self._events:
                if e.id == event_id:
                    return e
            return None

        def recent_events(self, n):
            return self._events[-n:] if n > 0 else []

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, child_id):
            self._children.append(child_id)

        def child_agents(self):
            return list(self._children)

        def compress_old_events(self, keep_recent, summary):
            kept = self._events[-keep_recent:] if keep_recent > 0 else []
            self._events = [_MockMemoryEvent(
                id="summary-0", event_type="summary", content=summary,
                timestamp_str="0", is_summary=True,
            )] + kept

        def compact_to_arrow(self):
            return 0

        def compact_to_arrow_with_meta(self, keywords, embedding=None, parent_chunk_id=None, summary=None):
            return 0

        def retrieve_relevant_chunks(self, active_chunk_id, max_hops, weights=None):
            return []

        def get_page_out_candidates(self, active_chunk_id, max_hops, budget):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock_core.WorkingMemory = _MockWorkingMemory

import pytest

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.llm.base import LLMConfig
from sage.llm.mock import MockProvider


@pytest.mark.asyncio
async def test_smmu_context_injection_runs_without_error():
    """Agent loop with S-MMU context injection must complete without error."""
    provider = MockProvider(responses=["Task completed successfully."])
    config = AgentConfig(
        name="test-smmu-inject",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3,
        validation_level=1,
    )
    events = []
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        on_event=events.append,
    )

    result = await loop.run("What is 2+2?")

    # Must produce a result without crashing
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

    # Must have gone through the standard phases
    event_types = [e.type for e in events]
    assert "PERCEIVE" in event_types
    assert "THINK" in event_types


@pytest.mark.asyncio
async def test_smmu_injection_with_semantic_memory():
    """S-MMU injection should coexist with semantic memory injection."""
    from unittest.mock import MagicMock

    provider = MockProvider(responses=["Done."])
    config = AgentConfig(
        name="test-coexist",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3,
        validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)

    # Wire a mock semantic memory
    sem = MagicMock()
    sem.get_context_for.return_value = "Entity: Python -> Language"
    sem.entity_count.return_value = 1
    loop.semantic_memory = sem

    result = await loop.run("test coexistence")
    assert result is not None
    assert isinstance(result, str)
