"""Tests for SemanticMemory + MemoryAgent wiring into AgentLoop."""
import sys
import types

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

        def compact_to_arrow_with_meta(self, keywords, embedding, parent_chunk_id):
            return 0

        def retrieve_relevant_chunks(self, active_chunk_id, max_hops, weights):
            return []

        def get_page_out_candidates(self, active_chunk_id, max_hops, budget):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock_core.WorkingMemory = _MockWorkingMemory

import pytest
from sage.boot import boot_agent_system
from sage.events.bus import EventBus


def test_memory_agent_wired():
    """MemoryAgent is injected into AgentLoop by boot sequence."""
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.memory_agent is not None


def test_semantic_memory_wired():
    """SemanticMemory is created and injected into AgentLoop by boot sequence."""
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.semantic_memory is not None


def test_semantic_memory_is_correct_type():
    """SemanticMemory instance has expected API surface."""
    system = boot_agent_system(use_mock_llm=True)
    sm = system.agent_loop.semantic_memory
    assert hasattr(sm, "add_extraction")
    assert hasattr(sm, "entity_count")
    assert hasattr(sm, "get_context_for")
    assert sm.entity_count() == 0  # Empty at boot


def test_memory_agent_matches_system():
    """MemoryAgent in loop is the same instance as in AgentSystem."""
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.memory_agent is system.memory_agent


@pytest.mark.asyncio
async def test_semantic_memory_in_learn_events():
    """LEARN events include semantic_entities count after a run."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("Tell me about Python programming")
    learn_events = [e for e in bus.query(last_n=200) if e.type == "LEARN"]
    # At least one LEARN event should exist
    assert len(learn_events) >= 1
    # The final LEARN event (completion) or in-loop LEARN should have semantic_entities
    has_semantic = any("semantic_entities" in e.meta for e in learn_events)
    assert has_semantic, "No LEARN event contained semantic_entities stat"


@pytest.mark.asyncio
async def test_entity_extraction_during_run():
    """Entity extraction populates SemanticMemory during agent run."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    # MockProvider returns "<think>Processing</think>\nDone." which is short.
    # We need a longer response to trigger extraction (>50 chars).
    # Override mock with a longer response containing capitalized terms.
    from sage.llm.mock import MockProvider
    long_response = (
        "Python is a Programming Language created by Guido van Rossum. "
        "It uses Dynamic Typing and supports Object Oriented Programming. "
        "The CPython interpreter is the reference implementation."
    )
    system.agent_loop._llm = MockProvider(responses=[long_response])
    await system.agent_loop.run("Tell me about Python")
    # The heuristic extractor should have found capitalized entities
    assert system.agent_loop.semantic_memory.entity_count() > 0


@pytest.mark.asyncio
async def test_semantic_context_injection_empty_graph():
    """Semantic context injection is safe with empty graph (no crash)."""
    system = boot_agent_system(use_mock_llm=True)
    # Empty semantic memory should not crash
    result = await system.agent_loop.run("Hello world")
    assert result is not None
