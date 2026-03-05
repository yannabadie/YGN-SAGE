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

    @Tool.define(
        name="refresh_knowledge",
        description=(
            "Trigger on-demand knowledge discovery: scan arXiv, Semantic Scholar, "
            "and HuggingFace for new papers, curate them, and ingest into ExoCortex. "
            "Use when you need the latest research on a specific topic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research topic to search for"},
                "domain": {
                    "type": "string",
                    "description": "Optional domain: marl, cognitive_architectures, formal_verification, evolutionary_computation, memory_systems",
                },
            },
            "required": [],
        },
    )
    async def refresh_knowledge(query: str | None = None, domain: str | None = None) -> str:
        try:
            from discover.pipeline import run_pipeline

            domains = [domain] if domain else None
            report = await run_pipeline(
                mode="on-demand" if query else "nightly",
                query=query,
                domains=domains,
                exocortex=exocortex,
            )
            return (
                f"Knowledge refresh complete: "
                f"{report.discovered} discovered, "
                f"{report.curated} curated, "
                f"{report.ingested} ingested."
            )
        except ImportError:
            return "Knowledge pipeline not available. Install sage-discover."
        except Exception as e:
            return f"Knowledge refresh failed: {e}"

    tools.append(refresh_knowledge)
    return tools
