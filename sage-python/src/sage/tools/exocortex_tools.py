"""ExoCortex agent tools: search_exocortex."""
from __future__ import annotations

import asyncio
from typing import Any

from sage.tools.base import Tool


def create_exocortex_tools(exocortex: Any) -> list[Tool]:
    """Create ExoCortex tools bound to the given ExoCortex instance."""
    tools: list[Tool] = []

    @Tool.define(
        name="search_exocortex",
        description=(
            "Search the ExoCortex knowledge store for research papers and SOTA insights. "
            "Use when you need specific research knowledge about MARL, cognitive architectures, "
            "formal verification, evolutionary computation, or memory systems."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research question to search for"},
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter: marl, cognitive_architectures, formal_verification, evolutionary_computation, memory_systems",
                },
            },
            "required": ["query"],
        },
    )
    async def search_exocortex(query: str, domain: str | None = None) -> str:
        if not exocortex or not exocortex.store_name:
            return "ExoCortex not configured. Set SAGE_EXOCORTEX_STORE environment variable."
        result = await asyncio.to_thread(exocortex.query, query, domain)
        if not result:
            return "No relevant results found in ExoCortex."
        return result

    tools.append(search_exocortex)
    return tools
