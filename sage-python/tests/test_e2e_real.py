"""E2E tests with real LLM. Require GOOGLE_API_KEY. Skip in CI."""
import os
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

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
]


@pytest.fixture(autouse=True)
def _patch_ssl():
    """Bypass SSL verification for corporate proxy."""
    import httpx
    original_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        original_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_init

    original_async_init = httpx.AsyncClient.__init__

    def patched_async_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        original_async_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = patched_async_init
    yield
    httpx.Client.__init__ = original_init
    httpx.AsyncClient.__init__ = original_async_init


# ---------------------------------------------------------------------------
# Test 1: S1 simple question — real LLM, fast tier
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_s1_simple_question():
    """S1 fast path: simple factual question via real Gemini."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
    result = await system.run("What is the capital of France?")
    assert "paris" in result.lower()


# ---------------------------------------------------------------------------
# Test 2: S2 code generation — real LLM, fast tier
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_s2_code_generation():
    """S2 code generation: real LLM produces a Python function."""
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast", event_bus=bus)
    result = await system.run("Write a Python function that checks if a number is prime.")
    assert "def " in result
