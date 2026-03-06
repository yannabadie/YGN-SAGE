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
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=mock_llm, on_event=events.append)
    result = await loop.run("test task")
    phases = [e.phase for e in events]
    assert LoopPhase.PERCEIVE in phases
    assert LoopPhase.THINK in phases

@pytest.mark.asyncio
async def test_agent_loop_compresses_memory_on_pressure():
    """Memory compressor fires when event count exceeds threshold."""
    from sage.memory.compressor import MemoryCompressor
    from sage.llm.mock import MockProvider
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig

    provider = MockProvider(responses=["SUMMARY: test summary\nDISCOVERIES:\n- discovery 1"])
    compressor = MemoryCompressor(
        llm=provider,
        compression_threshold=3,
        keep_recent=1,
    )

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider(responses=["Done."]))
    loop.memory_compressor = compressor

    # Manually add events to exceed threshold
    for i in range(4):
        loop.working_memory.add_event("TEST", f"event {i}")

    # Run compression step directly
    compressed = await compressor.step(loop.working_memory)
    assert compressed is True
    # After compression, event count should be reduced (at most keep_recent + summary)
    assert loop.working_memory.event_count() < 4


@pytest.mark.asyncio
async def test_agent_loop_learn_updates_memory(mock_llm):
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=mock_llm)
    await loop.run("test task")
    assert loop.working_memory.event_count() > 0


@pytest.mark.asyncio
async def test_self_brake_stores_in_working_memory():
    """CGRS self-brake must store response in working_memory before breaking."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.strategy.metacognition import MetacognitiveController

    ctrl = MetacognitiveController(brake_window=1, brake_entropy_threshold=1.0)
    ctrl.record_output_entropy(0.01)

    provider = MockProvider(responses=["Braked response content here."])
    config = AgentConfig(
        name="test-brake", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=5, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)
    loop.metacognition = ctrl

    result = await loop.run("test task")

    events = loop.working_memory._events
    assistant_events = [e for e in events if e["type"] == "ASSISTANT"]
    assert len(assistant_events) >= 1, "Braked response must be stored in working_memory"
    assert "Braked response" in assistant_events[-1]["content"]


def test_separate_retry_counters_exist():
    """S2 and S3 must have independent retry counters."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider())

    assert hasattr(loop, '_s3_retries'), "Must have _s3_retries counter"
    assert hasattr(loop, '_s2_avr_retries'), "Must have _s2_avr_retries counter"
    assert hasattr(loop, '_max_s3_retries'), "Must have _max_s3_retries"
    assert hasattr(loop, '_max_s2_avr_retries'), "Must have _max_s2_avr_retries"
    assert loop._s3_retries == 0
    assert loop._s2_avr_retries == 0
    assert not hasattr(loop, '_prm_retries'), "_prm_retries must be removed"
    assert not hasattr(loop, '_max_prm_retries'), "_max_prm_retries must be removed"


@pytest.mark.asyncio
async def test_loop_uses_async_metacognition():
    """Agent loop must call assess_complexity_async, not sync assess_complexity."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.strategy.metacognition import MetacognitiveController
    from unittest.mock import AsyncMock

    ctrl = MetacognitiveController()
    provider = MockProvider(responses=["Done."])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)
    loop.metacognition = ctrl

    events = []
    loop._on_event = events.append

    ctrl.assess_complexity_async = AsyncMock(return_value=ctrl._assess_heuristic("test"))

    await loop.run("test task")

    ctrl.assess_complexity_async.assert_called_once()
    perceive_events = [e for e in events if e.phase == LoopPhase.PERCEIVE]
    assert len(perceive_events) >= 1
    assert "routing_source" in perceive_events[0].data


def test_extract_code_blocks_returns_multiple():
    """_extract_code_blocks must return ALL blocks from multi-block content."""
    from sage.agent_loop import _extract_code_blocks

    content = 'First:\n```python\nx = 1\n```\nFixed:\n```python\nx = 2\n```\n'
    blocks = _extract_code_blocks(content)
    assert len(blocks) == 2
    assert "x = 1" in blocks[0]
    assert "x = 2" in blocks[1]


@pytest.mark.asyncio
async def test_compressor_generates_internal_state():
    """MEM1: compressor generates rolling <IS_t> every step."""
    from sage.memory.compressor import MemoryCompressor
    from sage.llm.mock import MockProvider

    provider = MockProvider(responses=[
        "Current state: user asked about sorting algorithms.",
        "Current state: user asked about sorting. Explored quicksort and mergesort.",
    ])
    compressor = MemoryCompressor(llm=provider, compression_threshold=20, keep_recent=5)

    is_1 = await compressor.generate_internal_state("User asked: explain sorting algorithms")
    assert is_1 != ""
    assert compressor.internal_state == is_1

    is_2 = await compressor.generate_internal_state("Assistant explained quicksort and mergesort")
    assert is_2 != ""
    assert compressor.internal_state == is_2
