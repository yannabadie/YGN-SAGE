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
    """Summary of a pipeline run."""
    discovered: int = 0
    curated: int = 0
    ingested: int = 0


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
    store_name = os.environ.get("SAGE_EXOCORTEX_STORE")
    if not store_name:
        return None
    try:
        from sage.memory.remote_rag import ExoCortex
        return ExoCortex()
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

    # Ingestion -- prefer Qdrant store, fall back to ExoCortex
    if store and embedder:
        report.ingested = await ingest_all_to_store(curated, store, embedder)
    else:
        exocortex = exocortex or _try_init_exocortex()
        if exocortex:
            report.ingested = await ingest_all(curated, exocortex)

    logger.info("Ingested %d papers", report.ingested)
    return report
