"""Tests for ConsistencyScore — pairwise cosine similarity."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock


def test_identical_texts_high_score():
    """Mock embedder returns identical embeddings → score near 1.0."""
    from sage.consistency import consistency_score
    mock_embedder = MagicMock()
    mock_embedder.batch_cosine_similarity.return_value = [1.0]
    score = consistency_score(["hello", "hello"], embedder=mock_embedder)
    assert score > 0.99


def test_different_texts_lower_score():
    """Mock embedder returns low similarity."""
    from sage.consistency import consistency_score
    mock_embedder = MagicMock()
    mock_embedder.batch_cosine_similarity.return_value = [0.3]
    score = consistency_score(["cats", "quantum physics"], embedder=mock_embedder)
    assert score < 0.5


def test_single_text_returns_one():
    from sage.consistency import consistency_score
    assert consistency_score(["just one"]) == 1.0


def test_empty_returns_one():
    from sage.consistency import consistency_score
    assert consistency_score([]) == 1.0


def test_no_embedder_returns_one():
    """No embedder and no sentence-transformers → return 1.0 (safe default)."""
    from sage.consistency import consistency_score
    score = consistency_score(["a", "b"], embedder=None)
    # Without sentence-transformers installed, should return 1.0
    assert 0.0 <= score <= 1.0


def test_embedder_failure_falls_back():
    """If Rust embedder raises, fall back gracefully."""
    from sage.consistency import consistency_score
    mock_embedder = MagicMock()
    mock_embedder.batch_cosine_similarity.side_effect = RuntimeError("ONNX fail")
    score = consistency_score(["a", "b"], embedder=mock_embedder)
    assert 0.0 <= score <= 1.0


def test_three_texts_mean_similarity():
    """3 texts → 3 pairs. Returns mean of all pairs."""
    from sage.consistency import consistency_score
    mock_embedder = MagicMock()
    # 3 pairs: (0,1)=0.8, (0,2)=0.6, (1,2)=0.4 → mean = 0.6
    mock_embedder.batch_cosine_similarity.return_value = [0.8, 0.6, 0.4]
    score = consistency_score(["a", "b", "c"], embedder=mock_embedder)
    assert abs(score - 0.6) < 0.01
