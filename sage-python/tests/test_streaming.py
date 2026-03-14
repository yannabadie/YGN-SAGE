"""Tests for AgentLoop.stream() — Phase 1, non-AVR streaming."""
import sys
import types as builtins_types

# Ensure sage_core mock exists (same pattern as test_agent_loop.py).
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = builtins_types.ModuleType("sage_core")

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
from collections.abc import AsyncIterator

import pytest

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop, AgentEvent, _is_code_task
from sage.llm.base import LLMConfig, LLMResponse, Message, StreamingLLMProvider


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------

class MockStreamingProvider:
    """Provider that implements both generate() and generate_stream()."""

    name = "mock-streaming"

    def __init__(self, chunks: list[str] | None = None, full_response: str = "full"):
        self._chunks = chunks or ["Hello", " ", "World"]
        self._full_response = full_response

    async def generate(
        self,
        messages: list[Message],
        tools=None,
        config=None,
        **kwargs,
    ) -> LLMResponse:
        return LLMResponse(content=self._full_response, tool_calls=[], model="mock")

    async def generate_stream(
        self,
        messages: list[Message],
        config=None,
    ) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class MockNonStreamingProvider:
    """Provider that only supports generate() — no streaming."""

    name = "mock-no-stream"

    def __init__(self, response: str = "non-stream result"):
        self._response = response

    async def generate(
        self,
        messages: list[Message],
        tools=None,
        config=None,
        **kwargs,
    ) -> LLMResponse:
        return LLMResponse(content=self._response, tool_calls=[], model="mock")


class MockFailingStreamProvider:
    """Provider whose generate_stream raises on the first chunk."""

    name = "mock-fail-stream"

    def __init__(self, fallback_response: str = "fallback"):
        self._fallback = fallback_response

    async def generate(
        self,
        messages: list[Message],
        tools=None,
        config=None,
        **kwargs,
    ) -> LLMResponse:
        return LLMResponse(content=self._fallback, tool_calls=[], model="mock")

    async def generate_stream(
        self,
        messages: list[Message],
        config=None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream exploded")
        # Make this a generator (yield is never reached but satisfies typing)
        yield ""  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(provider, events: list | None = None) -> AgentLoop:
    """Build an AgentLoop with minimal config and the given provider."""
    config = AgentConfig(
        name="test-stream",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3,
        validation_level=1,
    )
    return AgentLoop(
        config=config,
        llm_provider=provider,
        on_event=(events.append if events is not None else None),
    )


async def _collect_stream(loop: AgentLoop, task: str) -> list[str]:
    """Collect all chunks from stream() into a list."""
    chunks = []
    async for chunk in loop.stream(task):
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamIsAsyncGenerator:
    """Verify stream() returns a proper async generator."""

    @pytest.mark.asyncio
    async def test_returns_async_iterator(self):
        provider = MockStreamingProvider()
        loop = _make_loop(provider)
        result = loop.stream("What is 2+2?")
        assert hasattr(result, "__aiter__")
        assert hasattr(result, "__anext__")
        # Consume to avoid warnings
        async for _ in result:
            pass

    @pytest.mark.asyncio
    async def test_yields_multiple_chunks(self):
        provider = MockStreamingProvider(chunks=["a", "b", "c"])
        loop = _make_loop(provider)
        chunks = await _collect_stream(loop, "Explain gravity")
        assert chunks == ["a", "b", "c"]


class TestCodeTaskFallback:
    """Code tasks must fall back to run() and yield the full result."""

    @pytest.mark.asyncio
    async def test_code_task_single_chunk(self):
        provider = MockStreamingProvider(
            chunks=["should", "not", "stream"],
            full_response="code result from run()",
        )
        loop = _make_loop(provider)
        chunks = await _collect_stream(loop, "Write a Python function to sort a list")
        # Should get the full run() result, not the streaming chunks
        assert len(chunks) == 1
        assert chunks[0] == "code result from run()"

    @pytest.mark.asyncio
    async def test_is_code_task_detection(self):
        """Verify _is_code_task identifies various code prompts."""
        assert _is_code_task("implement a binary search")
        assert _is_code_task("Write a function that computes factorial")
        assert _is_code_task("def fibonacci(n):")
        assert _is_code_task("Write a Python script to download files")
        assert not _is_code_task("What is the capital of France?")
        assert not _is_code_task("Explain quantum physics")


class TestNonStreamingProviderFallback:
    """Providers without generate_stream fall back to run()."""

    @pytest.mark.asyncio
    async def test_non_streaming_provider_yields_full_result(self):
        provider = MockNonStreamingProvider(response="plain answer")
        loop = _make_loop(provider)
        chunks = await _collect_stream(loop, "Explain gravity")
        assert len(chunks) == 1
        assert chunks[0] == "plain answer"


class TestStreamingHappyPath:
    """Non-code tasks on a streaming provider yield multiple chunks."""

    @pytest.mark.asyncio
    async def test_non_code_task_streams(self):
        provider = MockStreamingProvider(chunks=["The ", "answer ", "is 42."])
        events: list[AgentEvent] = []
        loop = _make_loop(provider, events)
        chunks = await _collect_stream(loop, "What is the meaning of life?")
        assert chunks == ["The ", "answer ", "is 42."]

    @pytest.mark.asyncio
    async def test_emits_think_event_after_streaming(self):
        provider = MockStreamingProvider(chunks=["Hello", " World"])
        events: list[AgentEvent] = []
        loop = _make_loop(provider, events)
        await _collect_stream(loop, "Explain AI")
        # Should have emitted at least one THINK event with the full content
        think_events = [e for e in events if e.type == "THINK" and e.meta.get("content")]
        assert len(think_events) >= 1
        assert think_events[-1].meta["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_records_in_working_memory(self):
        provider = MockStreamingProvider(chunks=["chunk1", "chunk2"])
        loop = _make_loop(provider)
        await _collect_stream(loop, "Tell me a joke")
        # Working memory should contain the full assembled text.
        # Events may be dicts (real WorkingMemory) or objects (mock).
        recent = loop.working_memory.recent_events(5)
        found = False
        for e in recent:
            etype = e.event_type if hasattr(e, "event_type") else e.get("type", "")
            econtent = e.content if hasattr(e, "content") else e.get("content", "")
            if etype == "ASSISTANT" and "chunk1chunk2" in econtent:
                found = True
        assert found, f"Expected ASSISTANT event with 'chunk1chunk2', got {recent}"


class TestStreamingErrorFallback:
    """If streaming fails mid-way, graceful fallback."""

    @pytest.mark.asyncio
    async def test_stream_failure_falls_back_to_run(self):
        provider = MockFailingStreamProvider(fallback_response="safe fallback")
        loop = _make_loop(provider)
        chunks = await _collect_stream(loop, "Explain relativity")
        # Should fall back to run() since no chunks were yielded before failure
        assert len(chunks) == 1
        assert chunks[0] == "safe fallback"


class TestStreamingProtocol:
    """Verify StreamingLLMProvider protocol detection."""

    def test_streaming_provider_detected(self):
        provider = MockStreamingProvider()
        assert isinstance(provider, StreamingLLMProvider)

    def test_non_streaming_provider_not_detected(self):
        provider = MockNonStreamingProvider()
        assert not isinstance(provider, StreamingLLMProvider)


class TestBlockedInputStreaming:
    """Guardrail-blocked input yields the blocked reason."""

    @pytest.mark.asyncio
    async def test_blocked_task_yields_reason(self):
        """If perceive blocks the task, stream yields the block reason."""
        provider = MockStreamingProvider()
        loop = _make_loop(provider)

        # Inject a guardrail pipeline that blocks everything.
        # Must match the real GuardrailResult(passed, reason, severity) signature
        # and provide any_blocked() used by the perceive phase.
        class _BlockAll:
            async def check_all(self, input, context):
                from sage.guardrails.base import GuardrailResult
                return [GuardrailResult(
                    passed=False,
                    reason="blocked by test",
                    severity="block",
                )]

            def any_blocked(self, results):
                return any(not r.passed and r.severity == "block" for r in results)

        loop.guardrail_pipeline = _BlockAll()
        chunks = await _collect_stream(loop, "Tell me something")
        # The perceive phase should block and return a blocked reason
        assert len(chunks) == 1
        assert "blocked" in chunks[0].lower() or "guardrail" in chunks[0].lower()
