"""tests/test_adaptive_curator.py — Adaptive curation tests."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from discover.adaptive_curator import (
    CurationSignals,
    CurationBandit,
    KnnCurator,
    adaptive_curate,
)


def test_curation_signals_dataclass():
    s = CurationSignals(knn_score=0.8, llm_score=0.7, heuristic_score=0.6)
    assert s.knn_score == 0.8


def test_bandit_initial_decision():
    bandit = CurationBandit()
    signals = CurationSignals(knn_score=0.8, llm_score=0.9, heuristic_score=0.7)
    accept, confidence = bandit.decide(signals)
    assert isinstance(accept, bool)
    assert isinstance(confidence, float)


def test_bandit_update_shifts_weights():
    bandit = CurationBandit()
    theta_before = bandit.theta.copy()
    signals = CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0)
    bandit.update(signals, reward=1.0)
    assert not np.array_equal(bandit.theta, theta_before)


def test_bandit_learns_from_feedback():
    bandit = CurationBandit()
    for _ in range(20):
        bandit.update(CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0), reward=1.0)
        bandit.update(CurationSignals(knn_score=0.0, llm_score=1.0, heuristic_score=0.0), reward=0.0)
    accept, _ = bandit.decide(CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0))
    assert accept is True


def test_knn_curator_score():
    embeddings = np.random.rand(3, 768).astype(np.float32)
    labels = np.array([1, 1, 0])
    curator = KnnCurator(exemplar_embeddings=embeddings, exemplar_labels=labels)
    query = embeddings[0]
    score = curator.score(query, k=3)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_knn_curator_empty_returns_neutral():
    curator = KnnCurator(exemplar_embeddings=np.array([]).reshape(0, 768), exemplar_labels=np.array([]))
    score = curator.score(np.random.rand(768).astype(np.float32))
    assert score == 0.5


@pytest.mark.asyncio
async def test_adaptive_curate_returns_curated():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='[{"score": 8, "reason": "Relevant", "key_insights": ["insight"]}]'
    )
    from discover.discovery import PaperCandidate

    candidate = PaperCandidate(
        paper_id="p1", title="Test Paper", authors=["A"],
        abstract="A " * 60,
        source="arxiv", domain="marl",
        published=date.today(), pdf_url=None, citation_count=10,
    )

    mock_embedder = MagicMock()
    mock_embedder.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [], "values": []})

    results = await adaptive_curate(
        candidates=[candidate],
        llm=mock_llm,
        embedder=mock_embedder,
    )
    assert isinstance(results, list)
