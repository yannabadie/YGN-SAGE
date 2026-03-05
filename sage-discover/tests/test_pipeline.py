"""Tests for the pipeline orchestrator module."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.pipeline import PipelineReport, run_pipeline
from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_candidates() -> list[PaperCandidate]:
    """Two PaperCandidates for discovery mock."""
    return [
        PaperCandidate(
            paper_id="arxiv-2025-001",
            title="MAP-Elites for Neural Architecture Search",
            authors=["Alice"],
            abstract="A" * 200,
            source="arxiv",
            domain="evolutionary_computation",
            published=date.today() - timedelta(days=3),
            pdf_url="https://arxiv.org/pdf/2025.001.pdf",
            citation_count=10,
        ),
        PaperCandidate(
            paper_id="s2-2025-002",
            title="Cooperative MARL with Shared Memory",
            authors=["Bob"],
            abstract="B" * 200,
            source="s2",
            domain="marl",
            published=date.today() - timedelta(days=1),
            pdf_url=None,
            citation_count=5,
        ),
    ]


@pytest.fixture
def sample_curated(sample_candidates: list[PaperCandidate]) -> list[CuratedPaper]:
    """Two CuratedPapers wrapping sample candidates."""
    return [
        CuratedPaper(
            candidate=sample_candidates[0],
            relevance_score=8,
            reason="Relevant to MAP-Elites pillar",
            key_insights=["novel QD approach"],
        ),
        CuratedPaper(
            candidate=sample_candidates[1],
            relevance_score=7,
            reason="Good MARL paper",
            key_insights=["shared memory"],
        ),
    ]


# ---------------------------------------------------------------------------
# PipelineReport structure
# ---------------------------------------------------------------------------


def test_pipeline_report_structure():
    """PipelineReport dataclass has discovered, curated, ingested fields with defaults."""
    report = PipelineReport()
    assert report.discovered == 0
    assert report.curated == 0
    assert report.ingested == 0

    report2 = PipelineReport(discovered=10, curated=5, ingested=3)
    assert report2.discovered == 10
    assert report2.curated == 5
    assert report2.ingested == 3


# ---------------------------------------------------------------------------
# Nightly mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("discover.pipeline.ingest_all", new_callable=AsyncMock, return_value=2)
@patch("discover.pipeline.curate", new_callable=AsyncMock)
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_nightly_mode(
    mock_discover: AsyncMock,
    mock_curate: AsyncMock,
    mock_ingest_all: AsyncMock,
    sample_candidates: list[PaperCandidate],
    sample_curated: list[CuratedPaper],
):
    """Nightly mode calls discover -> curate -> ingest_all."""
    mock_discover.return_value = sample_candidates
    mock_curate.return_value = sample_curated

    exocortex = MagicMock()
    llm = MagicMock()

    report = await run_pipeline(
        mode="nightly",
        query=None,
        since=None,
        domains=None,
        exocortex=exocortex,
        llm=llm,
    )

    mock_discover.assert_called_once()
    mock_curate.assert_called_once()
    mock_ingest_all.assert_called_once()
    assert report.discovered == 2
    assert report.curated == 2
    assert report.ingested == 2


# ---------------------------------------------------------------------------
# On-demand mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("discover.pipeline.ingest_all", new_callable=AsyncMock, return_value=1)
@patch("discover.pipeline.curate", new_callable=AsyncMock)
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_on_demand_mode(
    mock_discover: AsyncMock,
    mock_curate: AsyncMock,
    mock_ingest_all: AsyncMock,
    sample_candidates: list[PaperCandidate],
    sample_curated: list[CuratedPaper],
):
    """On-demand mode with query passes query and domains to discover."""
    mock_discover.return_value = sample_candidates[:1]
    mock_curate.return_value = sample_curated[:1]

    exocortex = MagicMock()
    llm = MagicMock()

    report = await run_pipeline(
        mode="on-demand",
        query="MAP-Elites quality diversity",
        since=date.today() - timedelta(days=7),
        domains=["evolutionary_computation"],
        exocortex=exocortex,
        llm=llm,
    )

    # Verify discover was called with the right args
    call_kwargs = mock_discover.call_args
    assert call_kwargs[1].get("query") == "MAP-Elites quality diversity" or call_kwargs[0][1] == "MAP-Elites quality diversity"
    assert report.discovered == 1
    assert report.curated == 1
    assert report.ingested == 1


# ---------------------------------------------------------------------------
# Migrate mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("discover.pipeline.migrate_notebooks", new_callable=AsyncMock, return_value=5)
async def test_pipeline_migrate_mode(mock_migrate: AsyncMock):
    """Migrate mode calls migrate_notebooks and returns count as ingested."""
    exocortex = MagicMock()

    report = await run_pipeline(
        mode="migrate",
        query=None,
        since=None,
        domains=None,
        exocortex=exocortex,
    )

    mock_migrate.assert_called_once_with(exocortex)
    assert report.ingested == 5
    assert report.discovered == 0
    assert report.curated == 0


# ---------------------------------------------------------------------------
# No-LLM fallback (heuristic-only curation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("discover.pipeline.ingest_all", new_callable=AsyncMock, return_value=2)
@patch("discover.pipeline.curate", new_callable=AsyncMock)
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_no_llm_heuristic_fallback(
    mock_discover: AsyncMock,
    mock_curate: AsyncMock,
    mock_ingest_all: AsyncMock,
    sample_candidates: list[PaperCandidate],
):
    """When llm is None and import fails, pipeline still runs with heuristic fallback."""
    mock_discover.return_value = sample_candidates
    # curate should NOT be called when no llm available

    exocortex = MagicMock()

    report = await run_pipeline(
        mode="nightly",
        query=None,
        since=None,
        domains=None,
        exocortex=exocortex,
        llm=None,
    )

    mock_discover.assert_called_once()
    # When no LLM, curate should not be called (heuristic fallback instead)
    assert report.discovered == 2
