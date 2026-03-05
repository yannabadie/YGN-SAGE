"""Pipeline orchestrator: ties discovery, curation, ingestion, and migration together.

Supports three modes:
- ``nightly``: scan all domains for recent papers, curate, and ingest.
- ``on-demand``: targeted search with a specific query.
- ``migrate``: bootstrap ExoCortex from NotebookLM markdown exports.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from discover.discovery import discover
from discover.curator import curate, heuristic_filter, CuratedPaper
from discover.ingestion import ingest_all
from discover.migration import migrate_notebooks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineReport:
    """Summary of a pipeline run."""

    discovered: int = 0
    curated: int = 0
    ingested: int = 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    mode: str = "nightly",
    query: str | None = None,
    since: date | None = None,
    domains: list[str] | None = None,
    exocortex: Any = None,
    llm: Any = None,
) -> PipelineReport:
    """Run the knowledge pipeline end-to-end.

    Parameters
    ----------
    mode : str
        One of ``"nightly"``, ``"on-demand"``, ``"migrate"``.
    query : str | None
        Free-text search query (used in on-demand mode).
    since : date | None
        Only include papers published on or after this date.
        Defaults to yesterday for nightly/on-demand modes.
    domains : list[str] | None
        Subset of domain keys to scan. Defaults to all 5 domains.
    exocortex : Any
        ExoCortex instance. If None, attempts to create from env var.
    llm : Any
        LLM provider for curation. If None, attempts to create GoogleProvider.

    Returns
    -------
    PipelineReport
        Summary with discovered, curated, and ingested counts.
    """
    # --- Resolve ExoCortex ---
    if exocortex is None:
        store = os.environ.get("SAGE_EXOCORTEX_STORE")
        if store:
            try:
                from sage.memory.remote_rag import ExoCortex

                exocortex = ExoCortex()
                logger.info("Created ExoCortex from SAGE_EXOCORTEX_STORE=%s", store)
            except (ImportError, Exception) as exc:
                logger.warning("Could not create ExoCortex: %s", exc)
        else:
            logger.warning("No ExoCortex provided and SAGE_EXOCORTEX_STORE not set")

    # --- Migrate mode ---
    if mode == "migrate":
        if exocortex is None:
            logger.error("Cannot migrate without ExoCortex")
            return PipelineReport()
        count = await migrate_notebooks(exocortex)
        return PipelineReport(ingested=count)

    # --- Resolve LLM ---
    if llm is None:
        try:
            from sage.llm.google import GoogleProvider

            llm = GoogleProvider()
            logger.info("Created GoogleProvider for curation")
        except (ImportError, ValueError) as exc:
            logger.warning("Could not create LLM provider: %s", exc)

    # --- Default since ---
    if since is None:
        since = date.today() - timedelta(days=1)

    # --- Discovery ---
    candidates = await discover(since=since, query=query or "", domains=domains)
    report = PipelineReport(discovered=len(candidates))
    logger.info("Discovered %d candidates", report.discovered)

    if not candidates:
        return report

    # --- Curation ---
    if llm is not None:
        curated = await curate(candidates, llm)
    else:
        # Heuristic-only fallback: apply heuristic filter, assign neutral score
        filtered = heuristic_filter(candidates)
        curated = [
            CuratedPaper(
                candidate=c,
                relevance_score=5,
                reason="Heuristic-only (no LLM available)",
            )
            for c in filtered
        ]
    report.curated = len(curated)
    logger.info("Curated %d papers", report.curated)

    if not curated:
        return report

    # --- Ingestion ---
    if exocortex is not None:
        count = await ingest_all(curated, exocortex)
        report.ingested = count
        logger.info("Ingested %d papers", report.ingested)
    else:
        logger.warning("No ExoCortex available -- skipping ingestion")

    return report
