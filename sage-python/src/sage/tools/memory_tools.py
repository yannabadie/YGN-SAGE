"""Memory tools: on-demand episodic recall for the agent loop.

Research basis: NotebookLM Technical — two-stage provenance-aware retrieval.
Stage 1 (default): compressed summary in working memory (handled by compressor).
Stage 2 (on-demand): deep search via this tool.
"""
from __future__ import annotations

from sage.tools.base import Tool
from sage.memory.episodic import EpisodicMemory


def create_search_memory_tool(episodic: EpisodicMemory) -> Tool:
    """Create a search_memory tool bound to a specific EpisodicMemory instance."""

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
        lines = []
        for r in results:
            lines.append(f"[{r['key']}] {r['content']}")
        return "\n".join(lines)

    return search_memory
