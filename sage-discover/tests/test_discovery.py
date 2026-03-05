"""Tests for the discovery module: arXiv, Semantic Scholar, HuggingFace scanning."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.discovery import (
    DOMAINS,
    PaperCandidate,
    _normalize_title,
    deduplicate,
    discover,
    discover_arxiv,
    discover_hf,
    discover_semantic_scholar,
)


# --- DOMAINS ---

def test_domains_has_five_entries():
    assert len(DOMAINS) == 5
    expected_keys = {
        "marl",
        "cognitive_architectures",
        "formal_verification",
        "evolutionary_computation",
        "memory_systems",
    }
    assert set(DOMAINS.keys()) == expected_keys
    for key, val in DOMAINS.items():
        assert "arxiv_categories" in val
        assert "keywords" in val


# --- PaperCandidate ---

def test_paper_candidate_dataclass():
    p = PaperCandidate(
        paper_id="2401.00001",
        title="Test Paper",
        authors=["Alice", "Bob"],
        abstract="An abstract.",
        source="arxiv",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url="https://arxiv.org/pdf/2401.00001",
        citation_count=42,
    )
    assert p.paper_id == "2401.00001"
    assert p.title == "Test Paper"
    assert p.authors == ["Alice", "Bob"]
    assert p.abstract == "An abstract."
    assert p.source == "arxiv"
    assert p.domain == "marl"
    assert p.published == date(2025, 1, 1)
    assert p.pdf_url == "https://arxiv.org/pdf/2401.00001"
    assert p.citation_count == 42


# --- Deduplication ---

def test_deduplicate_merges_by_title():
    """Two papers with same title (different case/punctuation) -> 1 result with highest citation."""
    p1 = PaperCandidate(
        paper_id="a1",
        title="Multi-Agent Reinforcement Learning: A Survey",
        authors=["Alice"],
        abstract="abs1",
        source="arxiv",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url=None,
        citation_count=10,
    )
    p2 = PaperCandidate(
        paper_id="b2",
        title="multi-agent reinforcement learning a survey",
        authors=["Bob"],
        abstract="abs2",
        source="s2",
        domain="marl",
        published=date(2025, 2, 1),
        pdf_url="http://example.com",
        citation_count=50,
    )
    result = deduplicate([p1, p2])
    assert len(result) == 1
    assert result[0].citation_count == 50
    assert result[0].paper_id == "b2"


def test_deduplicate_preserves_different():
    """Two different papers -> both preserved."""
    p1 = PaperCandidate(
        paper_id="a1",
        title="Paper Alpha",
        authors=["Alice"],
        abstract="abs1",
        source="arxiv",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url=None,
        citation_count=5,
    )
    p2 = PaperCandidate(
        paper_id="b2",
        title="Paper Beta",
        authors=["Bob"],
        abstract="abs2",
        source="s2",
        domain="marl",
        published=date(2025, 2, 1),
        pdf_url=None,
        citation_count=10,
    )
    result = deduplicate([p1, p2])
    assert len(result) == 2


# --- discover_arxiv ---

@pytest.mark.asyncio
async def test_discover_arxiv_returns_candidates():
    """Mock arxiv module, verify returns PaperCandidate list with source='arxiv'."""
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/2401.00001v1"
    mock_result.title = "MARL for Cooperative Games"
    mock_result.authors = [MagicMock(name="Alice"), MagicMock(name="Bob")]
    # arxiv authors have a .name attribute
    mock_result.authors[0].name = "Alice"
    mock_result.authors[1].name = "Bob"
    mock_result.summary = "We study MARL in cooperative settings."
    mock_result.published = MagicMock()
    mock_result.published.date.return_value = date(2025, 6, 1)
    mock_result.pdf_url = "https://arxiv.org/pdf/2401.00001"

    mock_client = MagicMock()
    mock_search_cls = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.results.return_value = [mock_result]

    with patch.dict("sys.modules", {"arxiv": MagicMock()}):
        import sys
        arxiv_mock = sys.modules["arxiv"]
        arxiv_mock.Client.return_value = mock_client_instance
        arxiv_mock.Search.return_value = MagicMock()
        arxiv_mock.SortCriterion.SubmittedDate = "submitted_date"

        candidates = await discover_arxiv("marl", since=date(2025, 1, 1), max_results=5)

    assert isinstance(candidates, list)
    for c in candidates:
        assert isinstance(c, PaperCandidate)
        assert c.source == "arxiv"


# --- discover (end-to-end with mocks) ---

@pytest.mark.asyncio
async def test_discover_returns_deduplicated():
    """Mock all 3 sources, verify dedup works end-to-end."""
    shared_title = "Shared Paper Title"

    paper_arxiv = PaperCandidate(
        paper_id="arxiv-1",
        title=shared_title,
        authors=["A"],
        abstract="abs",
        source="arxiv",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url=None,
        citation_count=5,
    )
    paper_s2 = PaperCandidate(
        paper_id="s2-1",
        title=shared_title,
        authors=["B"],
        abstract="abs",
        source="s2",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url=None,
        citation_count=20,
    )
    paper_hf = PaperCandidate(
        paper_id="hf-1",
        title="Unique HF Paper",
        authors=["C"],
        abstract="abs",
        source="hf",
        domain="marl",
        published=date(2025, 1, 1),
        pdf_url=None,
        citation_count=0,
    )

    with (
        patch("discover.discovery.discover_arxiv", new_callable=AsyncMock, return_value=[paper_arxiv]),
        patch("discover.discovery.discover_semantic_scholar", new_callable=AsyncMock, return_value=[paper_s2]),
        patch("discover.discovery.discover_hf", new_callable=AsyncMock, return_value=[paper_hf]),
    ):
        results = await discover(
            since=date(2025, 1, 1),
            query="multi-agent",
            domains=["marl"],
        )

    # shared_title appears from both arxiv and s2 -> deduplicated to 1 (highest citation=20)
    # plus unique HF paper -> total 2
    assert len(results) == 2
    titles = {r.title for r in results}
    assert shared_title in titles or "shared paper title" in {_normalize_title(t) for t in titles}
    assert any(r.title == "Unique HF Paper" for r in results)
    # The surviving shared paper should be the one with citation_count=20
    shared = [r for r in results if _normalize_title(r.title) == _normalize_title(shared_title)]
    assert len(shared) == 1
    assert shared[0].citation_count == 20
