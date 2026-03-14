"""Tests for agent loop phase decomposition.

Verifies:
- LoopContext can be constructed (from both sage.phases and sage.agent_loop_context)
- Each phase function exists and is callable
- AgentLoop still works after refactoring to delegate to phase modules
"""
from __future__ import annotations

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


# ── LoopContext construction tests ──


class TestLoopContext:
    """Verify LoopContext can be constructed and has correct defaults."""

    def test_construct_from_phases(self):
        from sage.phases import LoopContext
        ctx = LoopContext(task="test task", messages=[])
        assert ctx.task == "test task"
        assert ctx.step == 0
        assert ctx.done is False
        assert ctx.result_text == ""
        assert ctx.cost == 0.0

    def test_construct_from_agent_loop_context_module(self):
        from sage.agent_loop_context import LoopContext
        ctx = LoopContext(task="another task", messages=[{"role": "user"}])
        assert ctx.task == "another task"
        assert len(ctx.messages) == 1

    def test_defaults_are_correct(self):
        from sage.phases import LoopContext
        ctx = LoopContext(task="t", messages=[])
        assert ctx.routing_decision is None
        assert ctx.tool_calls == []
        assert ctx.has_tool_calls is False
        assert ctx.guardrail_results == []
        assert ctx.is_code_task is False
        assert ctx.validation_level == "default"
        assert ctx.topology_result is None

    def test_mutable_defaults_isolated(self):
        """Mutable defaults (lists) must NOT be shared between instances."""
        from sage.phases import LoopContext
        ctx1 = LoopContext(task="a", messages=[])
        ctx2 = LoopContext(task="b", messages=[])
        ctx1.tool_calls.append("tool1")
        assert ctx2.tool_calls == []
        ctx1.guardrail_results.append("r1")
        assert ctx2.guardrail_results == []

    def test_custom_values(self):
        from sage.phases import LoopContext
        ctx = LoopContext(
            task="code task",
            messages=[{"role": "system"}],
            step=5,
            done=True,
            result_text="result",
            cost=0.12,
            is_code_task=True,
            validation_level="s3",
        )
        assert ctx.step == 5
        assert ctx.done is True
        assert ctx.result_text == "result"
        assert ctx.cost == 0.12
        assert ctx.is_code_task is True
        assert ctx.validation_level == "s3"


# ── Phase function existence and callability tests ──


class TestPhaseFunctionsExist:
    """Verify each phase function is importable and callable."""

    def test_perceive_importable_and_callable(self):
        from sage.phases.perceive import perceive
        assert callable(perceive)

    def test_think_importable_and_callable(self):
        from sage.phases.think import think
        assert callable(think)

    def test_act_importable_and_callable(self):
        from sage.phases.act import act
        assert callable(act)

    def test_learn_step_importable_and_callable(self):
        from sage.phases.learn import learn_step
        assert callable(learn_step)

    def test_learn_final_importable_and_callable(self):
        from sage.phases.learn import learn_final
        assert callable(learn_final)

    def test_all_importable_from_submodules(self):
        """All phase functions should be importable from sage.phases submodules."""
        from sage.phases.perceive import perceive
        from sage.phases.think import think
        from sage.phases.act import act
        from sage.phases.learn import learn_step, learn_final
        assert callable(perceive)
        assert callable(think)
        assert callable(act)
        assert callable(learn_step)
        assert callable(learn_final)

    def test_phase_submodules_importable(self):
        """Phase submodules should be importable as modules."""
        from sage.phases import perceive, think, act, learn  # noqa: F811,F401
        # These import as modules (perceive.py, think.py, etc.)
        import sage.phases.perceive
        import sage.phases.think
        import sage.phases.act
        import sage.phases.learn
        assert sage.phases.perceive is not None
        assert sage.phases.think is not None
        assert sage.phases.act is not None
        assert sage.phases.learn is not None


# ── AgentLoop backward compatibility tests ──


class TestAgentLoopBackwardCompat:
    """Verify AgentLoop still works after refactoring to delegate to phases."""

    def test_agent_loop_importable(self):
        from sage.agent_loop import AgentLoop
        assert AgentLoop is not None

    def test_agent_loop_public_api_unchanged(self):
        """AgentLoop must still expose the same public attributes and methods."""
        from sage.agent_loop import AgentLoop, LoopPhase, LoopEvent, AgentEvent
        assert hasattr(AgentLoop, "run")
        assert LoopPhase.PERCEIVE.value == "perceive"
        assert LoopPhase.THINK.value == "think"
        assert LoopPhase.ACT.value == "act"
        assert LoopPhase.LEARN.value == "learn"
        # LoopEvent and AgentEvent are dataclasses
        evt = LoopEvent(phase=LoopPhase.THINK)
        assert evt.phase == LoopPhase.THINK
        ae = AgentEvent(type="THINK", step=1, timestamp=0.0)
        assert ae.type == "THINK"

    def test_helper_functions_still_exported(self):
        """Helper functions used by external code must still be importable."""
        from sage.agent_loop import (
            _extract_code_blocks,
            _validate_code_syntax,
            _is_stagnating,
            _strip_markdown_fences,
            _is_code_task,
            _estimate_tokens,
            _text_entropy,
            _COST_PER_1K,
            S2_MAX_RETRIES_BEFORE_ESCALATION,
            S2_AVR_MAX_ITERATIONS,
            MAX_MESSAGES,
        )
        assert callable(_extract_code_blocks)
        assert callable(_validate_code_syntax)
        assert callable(_is_stagnating)
        assert callable(_strip_markdown_fences)
        assert callable(_is_code_task)
        assert callable(_estimate_tokens)
        assert callable(_text_entropy)
        assert isinstance(_COST_PER_1K, dict)
        assert isinstance(S2_MAX_RETRIES_BEFORE_ESCALATION, int)
        assert isinstance(S2_AVR_MAX_ITERATIONS, int)
        assert isinstance(MAX_MESSAGES, int)

    @pytest.mark.asyncio
    async def test_agent_loop_run_produces_output(self):
        """AgentLoop.run() must still produce correct output via phase delegation."""
        from sage.agent import AgentConfig
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider
        from sage.agent_loop import AgentLoop

        provider = MockProvider(responses=["The answer is 42."])
        config = AgentConfig(
            name="test",
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
        result = await loop.run("What is the meaning of life?")

        assert "42" in result
        types_seen = {e.type for e in events}
        assert "PERCEIVE" in types_seen
        assert "THINK" in types_seen
        assert "LEARN" in types_seen

    @pytest.mark.asyncio
    async def test_agent_loop_emits_all_phases(self):
        """All four phase types should appear in events during a normal run."""
        from sage.agent import AgentConfig
        from sage.llm.base import LLMConfig
        from sage.llm.mock import MockProvider
        from sage.agent_loop import AgentLoop

        provider = MockProvider(responses=["Simple response."])
        config = AgentConfig(
            name="test-phases",
            llm=LLMConfig(provider="mock", model="mock"),
            max_steps=2,
            validation_level=1,
        )
        events = []
        loop = AgentLoop(
            config=config,
            llm_provider=provider,
            on_event=events.append,
        )
        await loop.run("hello")

        types_seen = {e.type for e in events}
        # PERCEIVE, THINK, and LEARN are always emitted
        assert "PERCEIVE" in types_seen
        assert "THINK" in types_seen
        assert "LEARN" in types_seen
