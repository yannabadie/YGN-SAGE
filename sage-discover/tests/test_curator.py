"""Tests for the curator module: heuristic filter + LLM relevance scoring."""
from __future__ import annotations

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from discover.discovery import PaperCandidate
from discover.curator import (
    CuratedPaper,
    RELEVANCE_THRESHOLD,
    heuristic_filter,
    llm_score,
    curate,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_candidate(**overrides) -> PaperCandidate:
    """Create a PaperCandidate with sensible defaults, overridable."""
    defaults = dict(
        paper_id="test-001",
        title="Neural Architecture Search via MAP-Elites",
        authors=["Alice", "Bob"],
        abstract="A" * 200,  # 200 chars, well above MIN_ABSTRACT_LENGTH
        source="arxiv",
        domain="evolutionary_computation",
        published=date.today() - timedelta(days=10),  # recent
        pdf_url="https://arxiv.org/pdf/test-001",
        citation_count=5,
    )
    defaults.update(overrides)
    return PaperCandidate(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_relevance_threshold_is_six():
    assert RELEVANCE_THRESHOLD == 6


def test_heuristic_rejects_short_abstract():
    """Abstract shorter than MIN_ABSTRACT_LENGTH should be rejected."""
    short = _make_candidate(abstract="Too short")
    result = heuristic_filter([short])
    assert result == []


def test_heuristic_rejects_stale_uncited():
    """Paper older than STALE_DAYS with 0 citations should be rejected."""
    old_uncited = _make_candidate(
        published=date.today() - timedelta(days=180),
        citation_count=0,
    )
    result = heuristic_filter([old_uncited])
    assert result == []


def test_heuristic_keeps_recent_paper():
    """A recent paper (within STALE_DAYS) should be kept even with 0 citations."""
    recent = _make_candidate(
        published=date.today() - timedelta(days=10),
        citation_count=0,
    )
    result = heuristic_filter([recent])
    assert len(result) == 1
    assert result[0].paper_id == "test-001"


def test_heuristic_keeps_old_but_cited():
    """An old paper with citations should be kept."""
    old_cited = _make_candidate(
        published=date.today() - timedelta(days=180),
        citation_count=50,
    )
    result = heuristic_filter([old_cited])
    assert len(result) == 1


def test_heuristic_rejects_blocklist_title():
    """A paper whose title matches a blocklist pattern should be rejected."""
    erratum = _make_candidate(title="Erratum for our previous work")
    survey_of_surveys = _make_candidate(title="A Survey of Surveys on LLMs")
    correction = _make_candidate(title="Correction to: Deep RL Results")

    result = heuristic_filter([erratum, survey_of_surveys, correction])
    assert result == []


@pytest.mark.asyncio
async def test_llm_score_returns_curated_papers():
    """Mock LLM returning valid JSON; verify CuratedPaper fields."""
    candidate = _make_candidate()
    llm = AsyncMock()
    llm.generate.return_value = MagicMock(
        content='[{"score": 8, "reason": "Relevant to MAP-Elites", "key_insights": ["novel QD approach"]}]'
    )

    result = await llm_score([candidate], llm)

    assert len(result) == 1
    cp = result[0]
    assert isinstance(cp, CuratedPaper)
    assert cp.candidate is candidate
    assert cp.relevance_score == 8
    assert cp.reason == "Relevant to MAP-Elites"
    assert cp.key_insights == ["novel QD approach"]


@pytest.mark.asyncio
async def test_llm_score_handles_parse_failure():
    """When LLM returns unparseable JSON, assign neutral score 5."""
    candidate = _make_candidate()
    llm = AsyncMock()
    llm.generate.return_value = MagicMock(content="NOT VALID JSON AT ALL")

    result = await llm_score([candidate], llm)

    assert len(result) == 1
    cp = result[0]
    assert cp.relevance_score == 5
    assert cp.reason == "LLM scoring failed"


@pytest.mark.asyncio
async def test_curate_full_pipeline():
    """One candidate passes heuristic, one fails. Only the passing one returns."""
    good = _make_candidate(paper_id="good-001")
    bad_short = _make_candidate(paper_id="bad-002", abstract="short")

    llm = AsyncMock()
    llm.generate.return_value = MagicMock(
        content='[{"score": 9, "reason": "Excellent", "key_insights": ["breakthrough"]}]'
    )

    result = await curate([good, bad_short], llm)

    # bad_short is removed by heuristic; good passes heuristic + LLM score 9 >= 6
    assert len(result) == 1
    assert result[0].candidate.paper_id == "good-001"
    assert result[0].relevance_score == 9
