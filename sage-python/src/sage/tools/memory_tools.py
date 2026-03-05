"""AgeMem memory tools: 7 learnable actions for STM + LTM management.

Research basis:
- AgeMem: Unified tool-based memory policy (RETRIEVE/SUMMARY/FILTER + ADD/UPDATE/DELETE)
- NotebookLM Technical: Two-stage provenance-aware retrieval
- MEM1: Rolling internal state via compressor
"""
from __future__ import annotations

from typing import Any

from sage.tools.base import Tool
from sage.memory.episodic import EpisodicMemory
from sage.memory.working import WorkingMemory


def create_search_memory_tool(episodic: EpisodicMemory) -> Tool:
    """Legacy: create a single search_memory tool. Use create_memory_tools() instead."""

    @Tool.define(
        name="search_memory",
        description=(
            "Search long-term episodic memory for relevant past experiences. "
            "Use when current context is insufficient to answer the task."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in memory"},
                "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        },
    )
    async def search_memory(query: str, top_k: int = 5) -> str:
        results = await episodic.search(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = [f"[{r['key']}] {r['content']}" for r in results]
        return "\n".join(lines)

    return search_memory


def create_memory_tools(
    working_memory: WorkingMemory,
    episodic: EpisodicMemory,
    compressor: Any | None = None,
) -> list[Tool]:
    """Create all 7 AgeMem memory tools bound to the memory instances.

    STM (Working Memory): retrieve_context, summarize_context, filter_context
    LTM (Episodic): search_memory, store_memory, update_memory, delete_memory
    """
    tools: list[Tool] = []

    # --- STM Tools (Working Memory) ---

    @Tool.define(
        name="retrieve_context",
        description="Retrieve the N most recent events from short-term working memory.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of recent events to retrieve", "default": 10},
            },
            "required": [],
        },
    )
    async def retrieve_context(n: int = 10) -> str:
        events = working_memory.recent_events(n)
        if not events:
            return "Working memory is empty."
        lines = [f"[{e['type']}] {e['content']}" for e in events]
        return "\n".join(lines)

    tools.append(retrieve_context)

    @Tool.define(
        name="summarize_context",
        description="Get the current internal state summary (rolling MEM1 IS) of the agent's memory.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    async def summarize_context() -> str:
        if compressor and hasattr(compressor, "internal_state") and compressor.internal_state:
            return f"Internal State: {compressor.internal_state}"
        ctx = working_memory.to_context_string()
        return ctx[:1000] if ctx else "No context available."

    tools.append(summarize_context)

    @Tool.define(
        name="filter_context",
        description="Trim working memory to keep only the N most recent events. Use to drop irrelevant early context.",
        parameters={
            "type": "object",
            "properties": {
                "keep_recent": {"type": "integer", "description": "Number of recent events to keep", "default": 5},
            },
            "required": [],
        },
    )
    async def filter_context(keep_recent: int = 5) -> str:
        count_before = working_memory.event_count()
        if count_before <= keep_recent:
            return f"Nothing to filter. Working memory has {count_before} events."
        working_memory.compress(keep_recent, "Filtered by agent request.")
        return f"Filtered working memory from {count_before} to ~{keep_recent} events."

    tools.append(filter_context)

    # --- LTM Tools (Episodic Memory) ---

    @Tool.define(
        name="search_memory",
        description="Search long-term episodic memory for relevant past experiences.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in memory"},
                "top_k": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        },
    )
    async def search_memory(query: str, top_k: int = 5) -> str:
        results = await episodic.search(query, top_k=top_k)
        if not results:
            return "No relevant memories found."
        lines = [f"[{r['key']}] {r['content']}" for r in results]
        return "\n".join(lines)

    tools.append(search_memory)

    @Tool.define(
        name="store_memory",
        description="Store a new entry in long-term episodic memory for future retrieval.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Unique identifier for this memory"},
                "content": {"type": "string", "description": "The content to store"},
            },
            "required": ["key", "content"],
        },
    )
    async def store_memory(key: str, content: str) -> str:
        await episodic.store(key, content)
        return f"Stored memory '{key}' successfully."

    tools.append(store_memory)

    @Tool.define(
        name="update_memory",
        description="Update an existing long-term memory entry by key.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key of the memory to update"},
                "content": {"type": "string", "description": "New content to replace the old"},
            },
            "required": ["key", "content"],
        },
    )
    async def update_memory(key: str, content: str) -> str:
        updated = await episodic.update(key, content=content)
        if updated:
            return f"Updated memory '{key}' successfully."
        return f"Memory '{key}' not found. Use store_memory to create it."

    tools.append(update_memory)

    @Tool.define(
        name="delete_memory",
        description="Delete an obsolete or incorrect long-term memory entry.",
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key of the memory to delete"},
            },
            "required": ["key"],
        },
    )
    async def delete_memory(key: str) -> str:
        deleted = await episodic.delete(key)
        if deleted:
            return f"Deleted memory '{key}' successfully."
        return f"Memory '{key}' not found."

    tools.append(delete_memory)

    return tools
