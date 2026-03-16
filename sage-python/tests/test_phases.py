"""Tests for the phases package — LoopContext and phase extraction."""
import sys
import types

# Ensure sage_core mock exists (same pattern as test_agent_loop.py)
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
from sage.phases import LoopContext


# ── LoopContext tests ──


def test_loop_context_defaults():
    ctx = LoopContext(task="test task", messages=[])
    assert ctx.task == "test task"
    assert ctx.step == 0
    assert ctx.done is False
    assert ctx.result_text == ""
    assert ctx.cost == 0.0
    assert ctx.routing_decision is None
    assert ctx.tool_calls == []
    assert ctx.has_tool_calls is False
    assert ctx.guardrail_results == []
    assert ctx.is_code_task is False
    assert ctx.validation_level == "default"
    assert ctx.topology_result is None


def test_loop_context_custom_values():
    ctx = LoopContext(
        task="build a function",
        messages=[{"role": "user", "content": "hello"}],
        step=3,
        done=True,
        result_text="done",
        cost=0.05,
        is_code_task=True,
    )
    assert ctx.step == 3
    assert ctx.done is True
    assert ctx.result_text == "done"
    assert ctx.cost == 0.05
    assert ctx.is_code_task is True


def test_loop_context_mutable_defaults_isolated():
    """Mutable defaults (lists) must NOT be shared between instances."""
    ctx1 = LoopContext(task="a", messages=[])
    ctx2 = LoopContext(task="b", messages=[])
    ctx1.tool_calls.append("tool1")
    assert ctx2.tool_calls == []


# ── Phase module import tests ──


def test_phase_modules_importable():
    from sage.phases import perceive, think, act, learn  # noqa: F401


# ── Legacy fallback test ──


@pytest.mark.asyncio
async def test_legacy_fallback_env_var():
    """When SAGE_AGENT_LOOP_LEGACY=1, run() uses _run_legacy()."""
    import os
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.agent_loop import AgentLoop

    provider = MockProvider(responses=["Legacy result."])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)

    old_val = os.environ.get("SAGE_AGENT_LOOP_LEGACY")
    try:
        os.environ["SAGE_AGENT_LOOP_LEGACY"] = "1"
        result = await loop.run("test task")
        assert result  # Should return something (legacy path works)
    finally:
        if old_val is None:
            os.environ.pop("SAGE_AGENT_LOOP_LEGACY", None)
        else:
            os.environ["SAGE_AGENT_LOOP_LEGACY"] = old_val


# ── Full agent loop still works (regression) ──


@pytest.mark.asyncio
async def test_agent_loop_still_works_after_refactor():
    """Critical regression test: agent loop run() must produce the same output."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.agent_loop import AgentLoop

    provider = MockProvider(responses=["The answer is 42."])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    events = []
    loop = AgentLoop(config=config, llm_provider=provider, on_event=events.append)
    result = await loop.run("What is the meaning of life?")

    assert "42" in result
    types_seen = {e.type for e in events}
    assert "PERCEIVE" in types_seen
    assert "THINK" in types_seen
    assert "LEARN" in types_seen
