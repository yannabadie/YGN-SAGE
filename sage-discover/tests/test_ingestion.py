"""Tests for the ingestion module: ExoCortex upload + manifest tracking."""
from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper
from discover.ingestion import (
    Manifest,
    load_manifest,
    save_manifest,
    is_already_ingested,
    download_pdf,
    ingest,
    ingest_all,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PAPERS_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_candidate() -> PaperCandidate:
    """A PaperCandidate with sensible defaults."""
    return PaperCandidate(
        paper_id="arxiv-2025-001",
        title="MAP-Elites for Neural Architecture Search",
        authors=["Alice", "Bob"],
        abstract="A" * 200,
        source="arxiv",
        domain="evolutionary_computation",
        published=date.today() - timedelta(days=5),
        pdf_url="https://arxiv.org/pdf/2025.001.pdf",
        citation_count=10,
    )


@pytest.fixture
def sample_curated(sample_candidate: PaperCandidate) -> CuratedPaper:
    """A CuratedPaper wrapping the sample candidate."""
    return CuratedPaper(
        candidate=sample_candidate,
        relevance_score=8,
        reason="Highly relevant to MAP-Elites pillar",
        key_insights=["novel QD approach", "scalable to large search spaces"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_manifest_structure():
    """Manifest dataclass has store_name and papers fields with correct defaults."""
    m = Manifest()
    assert m.store_name == ""
    assert m.papers == {}
    assert isinstance(m.papers, dict)


def test_load_manifest_missing_file(tmp_path: Path):
    """Missing manifest file returns empty Manifest."""
    path = tmp_path / "nonexistent" / "manifest.json"
    m = load_manifest(path)
    assert isinstance(m, Manifest)
    assert m.store_name == ""
    assert m.papers == {}


def test_save_and_load_manifest(tmp_path: Path):
    """Round-trip: save then load should return identical Manifest."""
    path = tmp_path / "sub" / "manifest.json"
    m = Manifest(
        store_name="projects/test-store",
        papers={
            "paper-001": {
                "title": "Test Paper",
                "domain": "marl",
                "source": "arxiv",
                "relevance_score": 9,
                "ingested_at": "2026-03-06T10:00:00",
            }
        },
    )
    save_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded.store_name == m.store_name
    assert loaded.papers == m.papers


def test_is_already_ingested():
    """True if paper_id is in manifest.papers, False otherwise."""
    m = Manifest(papers={"paper-001": {"title": "Existing"}})
    assert is_already_ingested("paper-001", m) is True
    assert is_already_ingested("paper-999", m) is False


@pytest.mark.asyncio
async def test_ingest_skips_already_ingested(
    sample_curated: CuratedPaper, tmp_path: Path
):
    """If paper_id is already in manifest, ingest() returns False without uploading."""
    manifest_path = tmp_path / "manifest.json"
    # Pre-populate manifest with the paper
    m = Manifest(papers={sample_curated.candidate.paper_id: {"title": "Already there"}})
    save_manifest(m, manifest_path)

    exocortex = AsyncMock()
    result = await ingest(sample_curated, exocortex, manifest_path)

    assert result is False
    exocortex.upload.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_uploads_and_updates_manifest(
    sample_curated: CuratedPaper, tmp_path: Path
):
    """Successful ingest uploads to exocortex and records in manifest."""
    manifest_path = tmp_path / "manifest.json"
    # Create a fake PDF file so download is skipped
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    fake_pdf = pdf_dir / "arxiv-2025-001.pdf"
    fake_pdf.write_text("fake pdf content")
    sample_curated.pdf_path = fake_pdf

    exocortex = AsyncMock()
    result = await ingest(sample_curated, exocortex, manifest_path)

    assert result is True
    exocortex.upload.assert_called_once()
    # Verify manifest was updated
    loaded = load_manifest(manifest_path)
    assert sample_curated.candidate.paper_id in loaded.papers
    entry = loaded.papers[sample_curated.candidate.paper_id]
    assert entry["title"] == sample_curated.candidate.title
    assert entry["domain"] == sample_curated.candidate.domain
    assert entry["relevance_score"] == sample_curated.relevance_score


@pytest.mark.asyncio
async def test_ingest_all_returns_count(
    sample_curated: CuratedPaper, tmp_path: Path
):
    """ingest_all returns the count of newly ingested papers."""
    manifest_path = tmp_path / "manifest.json"

    # Create a second curated paper
    candidate2 = PaperCandidate(
        paper_id="s2-2025-002",
        title="Cooperative MARL with Shared Memory",
        authors=["Charlie"],
        abstract="B" * 200,
        source="s2",
        domain="marl",
        published=date.today() - timedelta(days=3),
        pdf_url=None,
        citation_count=5,
    )
    curated2 = CuratedPaper(
        candidate=candidate2,
        relevance_score=7,
        reason="Good MARL paper",
    )

    # Give both papers a pdf_path so download is skipped
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf1 = pdf_dir / "arxiv-2025-001.pdf"
    pdf1.write_text("fake")
    sample_curated.pdf_path = pdf1

    pdf2 = pdf_dir / "s2-2025-002.pdf"
    pdf2.write_text("fake")
    curated2.pdf_path = pdf2

    exocortex = AsyncMock()
    count = await ingest_all(
        [sample_curated, curated2], exocortex, manifest_path
    )

    assert count == 2
    assert exocortex.upload.call_count == 2


def test_default_paths():
    """DEFAULT_MANIFEST_PATH and DEFAULT_PAPERS_DIR are under ~/.sage."""
    assert DEFAULT_MANIFEST_PATH == Path.home() / ".sage" / "manifest.json"
    assert DEFAULT_PAPERS_DIR == Path.home() / ".sage" / "papers"
