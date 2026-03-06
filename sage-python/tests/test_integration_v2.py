"""Integration tests v2 — cross-component tests using MockProvider (no API key).

Covers:
- EventBus + boot_agent_system + agent loop (event emission)
- Metacognitive routing metadata in PERCEIVE events
- EpisodicMemory SQLite persistence across sessions
- SemanticMemory entity graph accumulation
- GuardrailPipeline integration (multi-guard, cost blocking)
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

from sage.events.bus import EventBus
from sage.boot import boot_agent_system
from sage.memory.episodic import EpisodicMemory
from sage.memory.semantic import SemanticMemory
from sage.memory.memory_agent import ExtractionResult
from sage.guardrails.base import GuardrailPipeline
from sage.guardrails.builtin import CostGuardrail, SchemaGuardrail


# ---------------------------------------------------------------------------
# Test 1: EventBus receives all phases from a full agent run
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_eventbus_receives_all_phases():
    """Boot with EventBus, run a task, verify PERCEIVE and THINK events appear."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("What is 2+2?")
    types_seen = {e.type for e in bus.query(last_n=100)}
    assert "PERCEIVE" in types_seen
    assert "THINK" in types_seen


# ---------------------------------------------------------------------------
# Test 2: PERCEIVE event contains routing metadata
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_routing_decision_in_perceive():
    """PERCEIVE event contains routing metadata (system, complexity)."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("Simple task")
    perceive = [e for e in bus.query(last_n=100) if e.type == "PERCEIVE"]
    assert len(perceive) >= 1
    # PERCEIVE should have routing info: either system field or in meta
    assert "system" in perceive[0].meta or perceive[0].system is not None


# ---------------------------------------------------------------------------
# Test 3: Episodic memory persistence across sessions (SQLite)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_episodic_persistence_cross_session(tmp_path):
    """Store in episodic memory, reopen, verify data persists."""
    db = str(tmp_path / "ep.db")
    mem1 = EpisodicMemory(db_path=db)
    await mem1.initialize()
    await mem1.store("fact1", "The speed of light is 299792458 m/s", {})
    del mem1

    mem2 = EpisodicMemory(db_path=db)
    await mem2.initialize()
    results = await mem2.search("light")
    assert len(results) == 1
    assert "299792458" in results[0]["content"]


# ---------------------------------------------------------------------------
# Test 4: Semantic memory accumulates entities across extractions
# ---------------------------------------------------------------------------
def test_semantic_memory_accumulates():
    """SemanticMemory accumulates entities across multiple extractions."""
    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["A", "B"],
        relationships=[("A", "links", "B")],
        summary="",
    ))
    sem.add_extraction(ExtractionResult(
        entities=["B", "C"],
        relationships=[("B", "links", "C")],
        summary="",
    ))
    assert sem.entity_count() == 3
    rels = sem.query_entities("B")
    assert len(rels) == 2


# ---------------------------------------------------------------------------
# Test 5: GuardrailPipeline integration — all pass
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guardrail_pipeline_integration():
    """GuardrailPipeline runs multiple guards, detects blocks."""
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),
        SchemaGuardrail(required_fields=["answer"]),
    ])
    results = await pipeline.check_all(
        context={"cost_usd": 0.5},
        output='{"answer": "42"}',
    )
    assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Test 6: CostGuardrail blocks when over budget
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_guardrail_cost_blocks():
    """CostGuardrail blocks when over budget."""
    pipeline = GuardrailPipeline([CostGuardrail(max_usd=0.01)])
    results = await pipeline.check_all(context={"cost_usd": 1.0})
    assert not results[0].passed
    assert pipeline.any_blocked(results)
