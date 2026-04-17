"""src/discover/pipeline.py -- Knowledge discovery pipeline orchestrator."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from discover.discovery import discover
from discover.curator import curate, heuristic_filter, CuratedPaper
from discover.ingestion import ingest_all, ingest_all_to_store
from discover.migration import migrate_notebooks

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Summary of a pipeline run.

    `ingested` is the MAX of `qdrant_ingested` and `exocortex_ingested`,
    representing "documents that landed in at least one backend" — the
    semantic a casual reader expects. Sum was rejected (advisor 2026-04-17,
    Codex concurred): a 2-paper run hitting both backends would show
    `ingested=4` and look like 4 unique papers to anyone glancing at the
    headline. Per-backend write counts stay in the breakdown fields for
    operational visibility.

    The historical contract (single backend → `ingested == that backend's
    count`) is preserved because max(N, 0) = N.
    """
    discovered: int = 0
    curated: int = 0
    ingested: int = 0
    qdrant_ingested: int = 0
    exocortex_ingested: int = 0


def _try_init_store():
    try:
        from discover.store import KnowledgeStore
        return KnowledgeStore()
    except Exception as e:
        logger.info("KnowledgeStore not available: %s", e)
        return None


def _try_init_embedder():
    try:
        from discover.embeddings import EmbeddingPipeline
        return EmbeddingPipeline()
    except Exception as e:
        logger.info("EmbeddingPipeline not available: %s", e)
        return None


def _try_init_exocortex():
    """Best-effort ExoCortex init. Returns None on import failure or unavailable.

    The class itself defaults `store_name` to the production store
    (remote_rag.DEFAULT_STORE) when SAGE_EXOCORTEX_STORE is unset, so we
    only need to gate on availability — not on the env var being present.
    Previously a missing env var silently disabled ExoCortex even when
    the class could have served queries.
    """
    try:
        from sage.memory.remote_rag import ExoCortex
        ex = ExoCortex()
        return ex if ex.is_available else None
    except Exception:
        return None


def _try_init_llm(llm=None):
    if llm is not None:
        return llm
    try:
        from sage.llm.google import GoogleProvider
        return GoogleProvider()
    except Exception:
        return None


async def run_pipeline(
    mode: str = "nightly",
    query: str | None = None,
    since: date | None = None,
    domains: list[str] | None = None,
    exocortex: Any = None,
    llm: Any = None,
    store: Any = None,
    embedder: Any = None,
) -> PipelineReport:
    """Run the knowledge discovery pipeline."""
    report = PipelineReport()

    if mode == "migrate":
        exocortex = exocortex or _try_init_exocortex()
        if exocortex is None:
            logger.warning("No ExoCortex configured for migration")
            return report
        count = await migrate_notebooks(exocortex)
        report.ingested = count
        return report

    # Initialize components
    llm = _try_init_llm(llm)
    store = store or _try_init_store()
    embedder = embedder or (None if store is None else _try_init_embedder())

    # Discovery
    since = since or (date.today() - timedelta(days=1))
    candidates = await discover(since=since, query=query or "", domains=domains)
    report.discovered = len(candidates)
    logger.info("Discovered %d papers", report.discovered)

    if not candidates:
        return report

    # Curation -- prefer adaptive, fall back to legacy
    if llm:
        try:
            from discover.adaptive_curator import adaptive_curate
            curated = await adaptive_curate(candidates, llm, embedder=embedder)
        except Exception:
            curated = await curate(candidates, llm)
    else:
        filtered = heuristic_filter(candidates)
        curated = [CuratedPaper(candidate=c, relevance_score=5, reason="heuristic") for c in filtered]

    report.curated = len(curated)
    logger.info("Curated %d papers", report.curated)

    if not curated:
        return report

    # Ingestion — write to BOTH stores when available.
    # Qdrant feeds adaptive_curate's local hybrid neighbour search.
    # ExoCortex feeds the runtime `search_exocortex` agent tool (Google
    # File Search store). Pre-fix this was an `if/else` — when Qdrant
    # init succeeded, ExoCortex was silently skipped and the runtime
    # store stayed frozen. See docs/benchmarks/2026-04-17-exocortex-debug.md
    if store and embedder:
        report.qdrant_ingested = await ingest_all_to_store(curated, store, embedder)
    exocortex = exocortex or _try_init_exocortex()
    if exocortex:
        report.exocortex_ingested = await ingest_all(curated, exocortex)
    # Headline = papers that reached at least one backend. See the
    # PipelineReport docstring for the sum-vs-max rejection rationale.
    report.ingested = max(report.qdrant_ingested, report.exocortex_ingested)

    logger.info(
        "Ingested papers: max=%d (qdrant=%d, exocortex=%d)",
        report.ingested, report.qdrant_ingested, report.exocortex_ingested,
    )
    return report
