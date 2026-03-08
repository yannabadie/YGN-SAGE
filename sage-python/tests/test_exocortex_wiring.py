"""Tests for ExoCortex end-to-end wiring in agent_loop + search_exocortex tool.

TDD: These tests define the expected behavior BEFORE implementation.
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

import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.llm.base import LLMConfig, LLMResponse, Message
from sage.llm.mock import MockProvider


# ---------------------------------------------------------------------------
# Test 1: AgentLoop has exocortex attribute after init
# ---------------------------------------------------------------------------
def test_agent_loop_has_exocortex_attribute():
    """AgentLoop.__init__ must define self.exocortex = None."""
    config = AgentConfig(
        name="test",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3,
        validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider())
    assert hasattr(loop, "exocortex"), "AgentLoop must have 'exocortex' attribute"
    assert loop.exocortex is None, "exocortex should default to None"


# ---------------------------------------------------------------------------
# Test 2: _think() source contains file_search_store_names
# ---------------------------------------------------------------------------
def test_exocortex_passive_grounding_removed():
    """ExoCortex passive grounding removed per Sprint 3 evidence.

    Passive injection was removed because it adds latency to every task.
    ExoCortex is now available as an active tool (search_exocortex) only.
    The exocortex attribute still exists for tool registration.
    """
    source = inspect.getsource(AgentLoop)
    # Passive injection comment should be present (documenting the decision)
    assert "passive grounding removed" in source.lower() or "active tool" in source.lower(), (
        "AgentLoop should document that passive ExoCortex grounding was removed"
    )


# ---------------------------------------------------------------------------
# Test 3: ExoCortex has query() method
# ---------------------------------------------------------------------------
def test_exocortex_query_method_exists():
    """ExoCortex must expose a query() method."""
    from sage.memory.remote_rag import ExoCortex
    exo = ExoCortex(store_name=None)
    assert hasattr(exo, "query"), "ExoCortex must have a 'query' method"
    assert callable(exo.query), "ExoCortex.query must be callable"


# ---------------------------------------------------------------------------
# Test 4: search_exocortex tool is importable and callable
# ---------------------------------------------------------------------------
def test_search_exocortex_tool_exists():
    """create_exocortex_tools must be importable and return tools."""
    from sage.tools.exocortex_tools import create_exocortex_tools

    # With a None exocortex, should still return a list
    tools = create_exocortex_tools(None)
    assert isinstance(tools, list)
    assert len(tools) > 0, "Must return at least one tool"

    # The first tool should be named 'search_exocortex'
    tool_names = [t.spec.name for t in tools]
    assert "search_exocortex" in tool_names, (
        "search_exocortex tool must be in the returned tools"
    )


# ---------------------------------------------------------------------------
# Test 5: passive grounding removed — generate() must NOT receive store names
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_loop_does_not_pass_store_names_to_generate():
    """Passive ExoCortex grounding removed per Sprint 3 evidence.

    Even when loop.exocortex is set, generate() must NOT receive
    file_search_store_names (active tool only, no passive injection).
    """
    config = AgentConfig(
        name="test",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3,
        validation_level=1,
    )

    # Create a mock provider whose generate() captures kwargs
    mock_provider = MockProvider(responses=["Test answer."])
    original_generate = mock_provider.generate
    generate_calls = []

    async def tracking_generate(*args, **kwargs):
        generate_calls.append(kwargs)
        return await original_generate(*args, **kwargs)

    mock_provider.generate = tracking_generate

    loop = AgentLoop(config=config, llm_provider=mock_provider)

    # Set up a mock exocortex with store_name and is_available
    mock_exocortex = MagicMock()
    mock_exocortex.store_name = "stores/test-store-123"
    mock_exocortex.is_available = True
    loop.exocortex = mock_exocortex

    await loop.run("What is MARL?")

    # Verify generate was called but WITHOUT file_search_store_names
    assert len(generate_calls) > 0, "generate() must have been called at least once"
    first_call = generate_calls[0]
    assert "file_search_store_names" not in first_call or not first_call.get("file_search_store_names"), (
        "generate() must NOT receive file_search_store_names (passive grounding removed)"
    )


# ---------------------------------------------------------------------------
# Test 6: MockProvider accepts file_search_store_names without error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_mock_provider_accepts_file_search_store_names():
    """MockProvider.generate() must accept file_search_store_names kwarg."""
    provider = MockProvider(responses=["ok"])
    response = await provider.generate(
        messages=[Message(role="user", content="test")],
        tools=None,
        config=None,
        file_search_store_names=["stores/test"],
    )
    assert response.content == "ok"


# ---------------------------------------------------------------------------
# Test 7: ExoCortex.query() returns empty string when not configured
# ---------------------------------------------------------------------------
def test_exocortex_query_returns_empty_when_not_configured(monkeypatch):
    """query() returns empty string when API key is missing."""
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from sage.memory.remote_rag import ExoCortex
    exo = ExoCortex(store_name="")
    result = exo.query("test question")
    assert result == ""
