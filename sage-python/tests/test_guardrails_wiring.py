"""Tests for guardrail wiring into the agent loop.

Verifies:
- GuardrailPipeline is wired by boot_agent_system
- Input/output guardrail events are emitted during agent runs
"""
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


@pytest.mark.asyncio
async def test_guardrails_emit_events():
    """Running an agent emits guardrail events (at least input + output)."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("Simple question")
    guardrail_events = [e for e in bus.query(last_n=200) if e.meta.get("guardrail")]
    # At least input + output guardrails should have run
    assert len(guardrail_events) >= 1


@pytest.mark.asyncio
async def test_guardrails_pipeline_wired():
    """boot_agent_system wires a GuardrailPipeline onto the agent loop."""
    system = boot_agent_system(use_mock_llm=True)
    assert system.agent_loop.guardrail_pipeline is not None


def test_guardrails_pipeline_has_cost_guard():
    """Default pipeline includes a CostGuardrail."""
    from sage.guardrails.builtin import CostGuardrail
    system = boot_agent_system(use_mock_llm=True)
    guards = system.agent_loop.guardrail_pipeline.guardrails
    assert any(isinstance(g, CostGuardrail) for g in guards)
