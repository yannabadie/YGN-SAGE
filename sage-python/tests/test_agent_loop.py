import sys, types

# Ensure sage_core mock exists with a WorkingMemory class.
# Other test files may have already inserted a bare ModuleType for sage_core,
# so we must always patch WorkingMemory onto whatever module is present.
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
from sage.agent_loop import AgentLoop, LoopEvent, LoopPhase

def test_loop_phases_exist():
    assert LoopPhase.PERCEIVE.value == "perceive"
    assert LoopPhase.THINK.value == "think"
    assert LoopPhase.ACT.value == "act"
    assert LoopPhase.LEARN.value == "learn"

def test_loop_event_structure():
    evt = LoopEvent(phase=LoopPhase.THINK, data={"content": "reasoning"})
    assert evt.phase == LoopPhase.THINK
    assert "content" in evt.data

@pytest.fixture
def mock_llm():
    from sage.llm.mock import MockProvider
    return MockProvider(responses=["<think>Analyzing task</think>\nDone."])

@pytest.mark.asyncio
async def test_agent_loop_emits_events(mock_llm):
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    events = []
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, enforce_system3=False,
    )
    loop = AgentLoop(config=config, llm_provider=mock_llm, on_event=events.append)
    result = await loop.run("test task")
    phases = [e.phase for e in events]
    assert LoopPhase.PERCEIVE in phases
    assert LoopPhase.THINK in phases

@pytest.mark.asyncio
async def test_agent_loop_learn_updates_memory(mock_llm):
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, enforce_system3=False,
    )
    loop = AgentLoop(config=config, llm_provider=mock_llm)
    await loop.run("test task")
    assert loop.working_memory.event_count() > 0
