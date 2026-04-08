"""src/discover/mcp.py — FastMCP server exposing 5 discovery tools."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Module-level import so tests can patch `discover.mcp.discover`.
# The local fallback inside tool_discover_papers handles circular-import edge cases.
try:
    from discover.discovery import discover
except ImportError:
    discover = None  # type: ignore[assignment]

_components: dict[str, Any] | None = None


def _get_pipeline_components() -> dict[str, Any]:
    """Lazy-initialize pipeline components."""
    global _components
    if _components is not None:
        return _components

    from discover.store import KnowledgeStore
    from discover.embeddings import EmbeddingPipeline
    from discover.rag import RAGPipeline

    store = KnowledgeStore()
    embedder = EmbeddingPipeline()

    llm = None
    try:
        from sage.llm.google import GoogleProvider
        llm = GoogleProvider()
    except Exception:
        logger.warning("No LLM provider available for MCP tools")

    rag = RAGPipeline(store=store, embedder=embedder, llm=llm)

    _components = {"store": store, "embedder": embedder, "llm": llm, "rag": rag}
    return _components


async def tool_discover_papers(
    query: str,
    domains: list[str] | None = None,
    since: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Discover papers from arXiv + Semantic Scholar + HuggingFace."""
    _discover = discover
    if _discover is None:
        from discover.discovery import discover as _discover  # type: ignore[no-redef]

    since_date = date.fromisoformat(since) if since else date.today() - timedelta(days=7)
    candidates = await _discover(since=since_date, query=query, domains=domains)

    # Persist discovered papers to local store so curate/verify can find them.
    # Without this, discover returns IDs that curate can't resolve.
    try:
        components = _get_pipeline_components()
        store = components.get("store")
        embedder = components.get("embedder")
        if store is not None and embedder is not None:
            for c in candidates[:max_results]:
                try:
                    dense, sparse = embedder.embed_paper(c.title, c.abstract)
                    store.upsert_paper(
                        c.paper_id,
                        dense,
                        sparse,
                        {
                            "title": c.title,
                            "authors": c.authors,
                            "abstract": c.abstract,
                            "source": c.source,
                            "domain": c.domain,
                            "published": c.published.isoformat(),
                            "year": getattr(c.published, "year", None),
                            "citation_count": c.citation_count,
                            "relevance_score": 0.0,
                        },
                    )
                except Exception as exc:
                    logger.warning("Failed to persist discovered paper %s: %s", c.paper_id, exc)
    except Exception as exc:
        logger.warning("Failed to initialize discovery persistence: %s", exc)

    return [
        {
            "paper_id": c.paper_id,
            "title": c.title,
            "authors": c.authors,
            "abstract": c.abstract[:300],
            "source": c.source,
            "domain": c.domain,
            "published": c.published.isoformat(),
            "citation_count": c.citation_count,
        }
        for c in candidates[:max_results]
    ]


async def tool_curate_papers(paper_ids: list[str]) -> list[dict]:
    """Score and filter papers using adaptive curation."""
    components = _get_pipeline_components()
    store = components["store"]
    papers = []
    for pid in paper_ids:
        p = store.get_paper(pid)
        if p:
            papers.append({"paper_id": pid, "title": p.get("title", ""), "relevance_score": p.get("relevance_score", 0)})
    return papers


async def tool_query_knowledge(question: str, top_k: int = 10, domain: str | None = None) -> str:
    """RAG query over the local knowledge store."""
    components = _get_pipeline_components()
    rag = components["rag"]
    return await rag.query(question, top_k=top_k, domain=domain)


async def tool_explore_frontier(domain: str | None = None, generations: int = 5) -> dict:
    """MAP-Elites exploration of the research frontier."""
    from discover.frontier import FrontierExplorer

    components = _get_pipeline_components()
    explorer = FrontierExplorer(store=components["store"], embedder=components["embedder"], llm=components["llm"])
    await explorer.seed()
    report = await explorer.explore(generations=generations)
    return {
        "coverage": round(report.coverage, 4),
        "total_papers": report.total_papers,
        "empty_regions": report.empty_regions,
        "generations_run": report.generations_run,
    }


async def tool_verify_claims(paper_id: str) -> dict:
    """Extract and SMT-verify claims from a paper."""
    from discover.claim_graph import extract_claims_from_text, verify_claim_cluster

    components = _get_pipeline_components()
    store = components["store"]
    llm = components["llm"]

    paper = store.get_paper(paper_id)
    if not paper:
        return {"error": f"Paper {paper_id} not found in store"}

    text = paper.get("abstract", "")
    if not text:
        return {"error": "No text available for claim extraction"}

    if llm is None:
        return {"error": "No LLM available for claim extraction"}

    claims = await extract_claims_from_text(text, paper_id, llm)
    if not claims:
        return {"paper_id": paper_id, "claims": [], "verification": "no_claims"}

    status = verify_claim_cluster(claims)
    return {
        "paper_id": paper_id,
        "claims": [{"statement": c.statement, "type": c.claim_type, "confidence": c.confidence} for c in claims],
        "verification": status,
    }


def create_mcp_server():
    """Create and configure the FastMCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp package not installed. Run: pip install 'mcp[cli]'")
        raise

    mcp = FastMCP("sage-discover")
    mcp.tool()(tool_discover_papers)
    mcp.tool()(tool_curate_papers)
    mcp.tool()(tool_query_knowledge)
    mcp.tool()(tool_explore_frontier)
    mcp.tool()(tool_verify_claims)
    return mcp


if __name__ == "__main__":
    server = create_mcp_server()
    server.run()
