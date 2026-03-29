"""tests/test_pipeline.py -- Updated pipeline tests."""
import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.pipeline import PipelineReport, run_pipeline
from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper


@pytest.fixture
def sample_candidates():
    return [
        PaperCandidate(
            paper_id="arxiv-2025-001", title="Paper A", authors=["Auth1"],
            abstract="A " * 60, source="arxiv", domain="evolutionary_computation",
            published=date.today(), pdf_url="https://arxiv.org/pdf/2025.001.pdf",
            citation_count=10,
        ),
        PaperCandidate(
            paper_id="s2-2025-002", title="Paper B", authors=["Auth2"],
            abstract="B " * 60, source="s2", domain="marl",
            published=date.today(), pdf_url=None, citation_count=5,
        ),
    ]


@pytest.fixture
def sample_curated(sample_candidates):
    return [
        CuratedPaper(candidate=sample_candidates[0], relevance_score=8, reason="Relevant",
                     key_insights=["insight1"]),
        CuratedPaper(candidate=sample_candidates[1], relevance_score=7, reason="Useful",
                     key_insights=["insight2"]),
    ]


def test_pipeline_report_structure():
    r = PipelineReport()
    assert r.discovered == 0 and r.curated == 0 and r.ingested == 0
    r2 = PipelineReport(discovered=5, curated=3, ingested=2)
    assert r2.discovered == 5


@pytest.mark.asyncio
@patch("discover.pipeline.ingest_all_to_store", new_callable=AsyncMock, return_value=2)
@patch("discover.pipeline.curate", new_callable=AsyncMock)
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_nightly_with_store(mock_discover, mock_curate, mock_ingest, sample_candidates, sample_curated):
    mock_discover.return_value = sample_candidates
    mock_curate.return_value = sample_curated
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    report = await run_pipeline(mode="nightly", llm=MagicMock(), store=mock_store, embedder=mock_embedder)
    assert report.discovered == 2
    assert report.ingested == 2
    mock_discover.assert_called_once()


@pytest.mark.asyncio
@patch("discover.pipeline.heuristic_filter")
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_no_llm_fallback(mock_discover, mock_filter, sample_candidates):
    mock_discover.return_value = sample_candidates
    mock_filter.return_value = sample_candidates
    report = await run_pipeline(mode="nightly", llm=None, store=None, embedder=None)
    assert report.discovered == 2
    assert report.curated == 2


@pytest.mark.asyncio
@patch("discover.pipeline.migrate_notebooks", new_callable=AsyncMock, return_value=3)
async def test_pipeline_migrate(mock_migrate):
    mock_exo = MagicMock()
    report = await run_pipeline(mode="migrate", exocortex=mock_exo)
    assert report.ingested == 3
