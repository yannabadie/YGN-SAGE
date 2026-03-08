"""Tests for EventBus — central event dispatch system.

TDD: Tests written BEFORE implementation.
"""
import sys
import types as _types

# Ensure sage_core mock exists (same pattern as test_agent_loop.py)
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")

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


import asyncio
import threading
import time

import pytest

from sage.agent_loop import AgentEvent
from sage.events.bus import EventBus


def _make_event(phase: str = "THINK", step: int = 0) -> AgentEvent:
    """Helper to create a test AgentEvent."""
    return AgentEvent(type=phase, step=step, timestamp=time.time())


# ---------------------------------------------------------------------------
# Test 1: emit and query basics
# ---------------------------------------------------------------------------
def test_emit_and_query():
    """Events emitted via emit() are retrievable via query()."""
    bus = EventBus()
    e1 = _make_event("PERCEIVE", step=0)
    e2 = _make_event("THINK", step=1)
    bus.emit(e1)
    bus.emit(e2)

    results = bus.query()
    assert len(results) == 2
    assert results[0].type == "PERCEIVE"
    assert results[1].type == "THINK"


# ---------------------------------------------------------------------------
# Test 2: query filters by phase (event.type field)
# ---------------------------------------------------------------------------
def test_query_filters_by_phase():
    """query(phase=...) only returns events matching that phase."""
    bus = EventBus()
    bus.emit(_make_event("PERCEIVE", step=0))
    bus.emit(_make_event("THINK", step=1))
    bus.emit(_make_event("THINK", step=2))
    bus.emit(_make_event("ACT", step=3))
    bus.emit(_make_event("LEARN", step=4))

    think_events = bus.query(phase="THINK")
    assert len(think_events) == 2
    assert all(e.type == "THINK" for e in think_events)

    act_events = bus.query(phase="ACT")
    assert len(act_events) == 1
    assert act_events[0].type == "ACT"


# ---------------------------------------------------------------------------
# Test 3: query last_n limits results
# ---------------------------------------------------------------------------
def test_query_last_n():
    """query(last_n=N) returns at most N most recent events."""
    bus = EventBus()
    for i in range(10):
        bus.emit(_make_event("THINK", step=i))

    results = bus.query(last_n=3)
    assert len(results) == 3
    # Should be the last 3 events
    assert results[0].step == 7
    assert results[1].step == 8
    assert results[2].step == 9


def test_query_last_n_with_phase_filter():
    """last_n applies AFTER phase filtering."""
    bus = EventBus()
    for i in range(5):
        bus.emit(_make_event("THINK", step=i))
        bus.emit(_make_event("ACT", step=i))

    results = bus.query(phase="THINK", last_n=2)
    assert len(results) == 2
    assert all(e.type == "THINK" for e in results)
    assert results[0].step == 3
    assert results[1].step == 4


# ---------------------------------------------------------------------------
# Test 4: subscribe receives events
# ---------------------------------------------------------------------------
def test_subscribe_receives_events():
    """Subscribed callbacks receive emitted events."""
    bus = EventBus()
    received = []
    bus.subscribe(received.append)

    e1 = _make_event("THINK", step=0)
    bus.emit(e1)

    assert len(received) == 1
    assert received[0] is e1


def test_multiple_subscribers():
    """Multiple subscribers each receive all events."""
    bus = EventBus()
    received_a = []
    received_b = []
    bus.subscribe(received_a.append)
    bus.subscribe(received_b.append)

    bus.emit(_make_event("THINK", step=0))

    assert len(received_a) == 1
    assert len(received_b) == 1


# ---------------------------------------------------------------------------
# Test 5: unsubscribe stops receiving
# ---------------------------------------------------------------------------
def test_unsubscribe_stops_receiving():
    """After unsubscribe, callback no longer receives events."""
    bus = EventBus()
    received = []
    sub_id = bus.subscribe(received.append)

    bus.emit(_make_event("THINK", step=0))
    assert len(received) == 1

    bus.unsubscribe(sub_id)
    bus.emit(_make_event("THINK", step=1))
    assert len(received) == 1  # No new events after unsubscribe


def test_unsubscribe_returns_gracefully_on_invalid_id():
    """unsubscribe with unknown ID does not raise."""
    bus = EventBus()
    bus.unsubscribe("nonexistent-id")  # Should not raise


# ---------------------------------------------------------------------------
# Test 6: async stream yields events
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_yields_events():
    """stream() yields events as they are emitted."""
    bus = EventBus()
    collected = []

    async def consumer():
        async for event in bus.stream():
            collected.append(event)
            if len(collected) >= 3:
                break

    task = asyncio.create_task(consumer())

    # Small delay to ensure consumer is waiting
    await asyncio.sleep(0.05)

    bus.emit(_make_event("PERCEIVE", step=0))
    bus.emit(_make_event("THINK", step=1))
    bus.emit(_make_event("ACT", step=2))

    await asyncio.wait_for(task, timeout=2.0)

    assert len(collected) == 3
    assert collected[0].type == "PERCEIVE"
    assert collected[1].type == "THINK"
    assert collected[2].type == "ACT"


@pytest.mark.asyncio
async def test_multiple_streams():
    """Multiple stream() consumers each get all events."""
    bus = EventBus()
    collected_a = []
    collected_b = []

    async def consumer(target_list):
        async for event in bus.stream():
            target_list.append(event)
            if len(target_list) >= 2:
                break

    task_a = asyncio.create_task(consumer(collected_a))
    task_b = asyncio.create_task(consumer(collected_b))

    await asyncio.sleep(0.05)

    bus.emit(_make_event("THINK", step=0))
    bus.emit(_make_event("ACT", step=1))

    await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=2.0)

    assert len(collected_a) == 2
    assert len(collected_b) == 2


@pytest.mark.asyncio
async def test_stream_threaded_emit_safe_in_debug_loop():
    """Threaded emit() must remain safe when stream() queue has waiting getters."""
    bus = EventBus()
    collected = []
    done = asyncio.Event()

    async def consumer():
        async for event in bus.stream():
            collected.append(event)
            done.set()
            break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)

    loop = asyncio.get_running_loop()
    prev_debug = loop.get_debug()
    loop.set_debug(True)
    try:
        errors = []

        def emitter():
            try:
                bus.emit(_make_event("THINK", step=99))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=emitter)
        t.start()
        t.join()

        await asyncio.wait_for(done.wait(), timeout=2.0)
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        loop.set_debug(prev_debug)

    assert errors == []
    assert len(collected) == 1
    assert collected[0].step == 99


# ---------------------------------------------------------------------------
# Test 7: buffer max size enforcement
# ---------------------------------------------------------------------------
def test_buffer_max_size():
    """Buffer evicts oldest events when max_buffer is exceeded."""
    bus = EventBus(max_buffer=5)

    for i in range(10):
        bus.emit(_make_event("THINK", step=i))

    results = bus.query(last_n=100)
    assert len(results) == 5
    # Should keep the most recent 5
    assert results[0].step == 5
    assert results[-1].step == 9


def test_default_max_buffer():
    """Default max_buffer is 5000."""
    bus = EventBus()
    assert bus.max_buffer == 5000


def test_custom_max_buffer():
    """Custom max_buffer is respected."""
    bus = EventBus(max_buffer=100)
    assert bus.max_buffer == 100


# ---------------------------------------------------------------------------
# Test 8: subscribe returns string ID
# ---------------------------------------------------------------------------
def test_subscribe_returns_string_id():
    """subscribe() returns a string subscription ID."""
    bus = EventBus()
    sub_id = bus.subscribe(lambda e: None)
    assert isinstance(sub_id, str)
    assert len(sub_id) > 0


# ---------------------------------------------------------------------------
# Test 9: thread safety (basic — emit from multiple threads)
# ---------------------------------------------------------------------------
def test_thread_safe_emit():
    """EventBus.emit is safe to call from multiple threads."""
    import threading

    bus = EventBus()
    n_threads = 10
    n_events_per_thread = 50
    barrier = threading.Barrier(n_threads)

    def emitter(thread_id):
        barrier.wait()
        for i in range(n_events_per_thread):
            bus.emit(_make_event("THINK", step=thread_id * 100 + i))

    threads = [threading.Thread(target=emitter, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results = bus.query(last_n=10000)
    assert len(results) == n_threads * n_events_per_thread


# ---------------------------------------------------------------------------
# Test 10: empty query returns empty list
# ---------------------------------------------------------------------------
def test_empty_query():
    """query() on empty bus returns empty list."""
    bus = EventBus()
    assert bus.query() == []
    assert bus.query(phase="THINK") == []
