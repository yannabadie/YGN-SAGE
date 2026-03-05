"""Tests for the memory system."""
import pytest
from sage.memory.working import WorkingMemory
from sage.memory.episodic import EpisodicMemory


def test_working_memory_add_event():
    mem = WorkingMemory(agent_id="agent-1")
    event_id = mem.add_event("tool_call", "Called bash with 'ls'")
    assert event_id is not None
    event = mem.get_event(event_id)
    assert event is not None
    assert event["content"] == "Called bash with 'ls'"


def test_working_memory_recent():
    mem = WorkingMemory(agent_id="agent-1")
    for i in range(10):
        mem.add_event("step", f"Step {i}")
    recent = mem.recent_events(3)
    assert len(recent) == 3
    assert recent[0]["content"] == "Step 7"


def test_working_memory_to_messages():
    mem = WorkingMemory(agent_id="agent-1")
    mem.add_event("user", "Hello")
    mem.add_event("assistant", "Hi there!")
    messages = mem.to_context_string()
    assert "Hello" in messages
    assert "Hi there!" in messages


def test_working_memory_arrow_compaction():
    mem = WorkingMemory(agent_id="agent-1")
    for i in range(5):
        mem.add_event("action", f"Run {i}")

    # Active buffer has 5
    assert len(mem.recent_events(5)) == 5

    # Compact to Arrow (returns chunk_id >= 0)
    chunk_id = mem.compact_to_arrow()
    assert chunk_id >= 0


@pytest.mark.asyncio
async def test_episodic_memory_store_and_search():
    mem = EpisodicMemory()
    await mem.store(
        key="fix-auth-bug",
        content="Fixed the auth bug by checking token expiry before validation.",
        metadata={"task": "bug-fix", "files": ["auth.py"]},
    )

    results = await mem.search("authentication bug fix")
    assert len(results) >= 1
    assert "auth" in results[0]["content"].lower()


@pytest.mark.asyncio
async def test_episodic_memory_empty_search():
    mem = EpisodicMemory()
    results = await mem.search("nonexistent query")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_memory_tool():
    """search_memory tool queries episodic memory and returns results."""
    from sage.tools.memory_tools import create_search_memory_tool

    episodic = EpisodicMemory()
    await episodic.store("debug-session", "Found null pointer bug in parser.py line 42")
    await episodic.store("architecture", "Agent uses perceive-think-act-learn loop")

    tool = create_search_memory_tool(episodic)
    result = await tool.execute({"query": "parser bug", "top_k": 3})
    assert "null pointer" in result.output


@pytest.mark.asyncio
async def test_episodic_memory_update():
    """AgeMem: agent can update existing memory entries."""
    mem = EpisodicMemory()
    await mem.store("auth-fix", "Fixed auth by checking token expiry")
    updated = await mem.update("auth-fix", content="Fixed auth by adding JWT refresh + token expiry check")
    assert updated is True
    results = await mem.search("auth")
    assert "JWT refresh" in results[0]["content"]


@pytest.mark.asyncio
async def test_episodic_memory_update_nonexistent():
    mem = EpisodicMemory()
    updated = await mem.update("nonexistent", content="new")
    assert updated is False


@pytest.mark.asyncio
async def test_episodic_memory_delete():
    """AgeMem: agent can delete obsolete memory entries."""
    mem = EpisodicMemory()
    await mem.store("temp-debug", "Temporary debug finding")
    await mem.store("permanent", "Important architecture note")
    deleted = await mem.delete("temp-debug")
    assert deleted is True
    results = await mem.search("debug")
    assert len(results) == 0
    results = await mem.search("architecture")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_episodic_memory_delete_nonexistent():
    mem = EpisodicMemory()
    deleted = await mem.delete("nonexistent")
    assert deleted is False


@pytest.mark.asyncio
async def test_episodic_memory_list_keys():
    mem = EpisodicMemory()
    await mem.store("key-a", "content a")
    await mem.store("key-b", "content b")
    keys = mem.list_keys()
    assert set(keys) == {"key-a", "key-b"}
