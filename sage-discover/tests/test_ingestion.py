"""tests/test_ingestion.py -- Updated ingestion tests."""
import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.ingestion import (
    ingest_to_store,
    ingest_all_to_store,
    download_pdf,
    Manifest,
    load_manifest,
    save_manifest,
    is_already_ingested,
)
from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper


@pytest.fixture
def sample_curated():
    candidate = PaperCandidate(
        paper_id="test-001", title="Test Paper", authors=["A"],
        abstract="Abstract text", source="arxiv", domain="marl",
        published=date.today(), pdf_url=None, citation_count=5,
    )
    return CuratedPaper(candidate=candidate, relevance_score=8, reason="good")


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_paper.return_value = None
    store.upsert_paper.return_value = None
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [1], "values": [0.5]})
    return emb


@pytest.mark.asyncio
async def test_ingest_to_store(sample_curated, mock_store, mock_embedder):
    result = await ingest_to_store(sample_curated, mock_store, mock_embedder)
    assert result is True
    mock_store.upsert_paper.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_to_store_skips_existing(sample_curated, mock_store, mock_embedder):
    mock_store.get_paper.return_value = {"title": "Already there"}
    result = await ingest_to_store(sample_curated, mock_store, mock_embedder)
    assert result is False


@pytest.mark.asyncio
async def test_ingest_all_to_store(sample_curated, mock_store, mock_embedder):
    count = await ingest_all_to_store([sample_curated], mock_store, mock_embedder)
    assert count == 1


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    m = Manifest(store_name="test", papers={"p1": {"title": "A"}})
    save_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded.store_name == "test"
    assert "p1" in loaded.papers


def test_is_already_ingested():
    m = Manifest(papers={"p1": {}})
    assert is_already_ingested("p1", m) is True
    assert is_already_ingested("p2", m) is False


@pytest.mark.asyncio
async def test_ingest_stamps_store_name_in_manifest(tmp_path, sample_curated):
    """Regression: pre-2026-04-17 the manifest's store_name field was
    never populated even though `ingest()` had access to a live ExoCortex
    client. Multi-store setups couldn't tell which Google File Search
    store a given run had written to.
    """
    from discover.ingestion import ingest

    manifest_path = tmp_path / "manifest.json"
    # PaperCandidate has no pdf_url → ingest() takes the markdown fallback.
    fake_exo = MagicMock()
    fake_exo.upload = AsyncMock()
    fake_exo.store_name = "fileSearchStores/my-target-store"

    ok = await ingest(sample_curated, fake_exo, manifest_path)
    assert ok is True
    fake_exo.upload.assert_awaited_once()

    saved = load_manifest(manifest_path)
    assert saved.store_name == "fileSearchStores/my-target-store"
    assert "test-001" in saved.papers
