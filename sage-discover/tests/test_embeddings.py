"""tests/test_embeddings.py — EmbeddingPipeline tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from discover.embeddings import EmbeddingPipeline, reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    list_a = [{"id": "p1", "score": 0.9}, {"id": "p2", "score": 0.8}]
    list_b = [{"id": "p2", "score": 0.95}, {"id": "p3", "score": 0.7}]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)
    ids = [r["id"] for r in fused]
    assert "p2" in ids  # appears in both, should rank high
    assert len(fused) == 3  # p1, p2, p3


def test_rrf_single_list():
    items = [{"id": "a", "score": 1.0}, {"id": "b", "score": 0.5}]
    fused = reciprocal_rank_fusion(items, k=60)
    assert fused[0]["id"] == "a"


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder", None)
def test_embed_paper_dense_only(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
    mock_st_cls.return_value = mock_model
    pipeline = EmbeddingPipeline()
    dense, sparse = pipeline.embed_paper("Title", "Abstract")
    assert dense.shape == (768,)
    assert "indices" in sparse
    assert "values" in sparse


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder", None)
def test_embed_text(mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
    mock_st_cls.return_value = mock_model
    pipeline = EmbeddingPipeline()
    result = pipeline.embed_text("some query text")
    assert result.shape == (768,)


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder", None)
@patch("discover.embeddings.CrossEncoder")
def test_rerank_returns_sorted(mock_ce_cls, mock_st_cls):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(768).astype(np.float32)
    mock_st_cls.return_value = mock_model
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.1, 0.9, 0.5])
    mock_ce_cls.return_value = mock_ce
    pipeline = EmbeddingPipeline()
    candidates = [
        {"title": "A", "abstract": "a"},
        {"title": "B", "abstract": "b"},
        {"title": "C", "abstract": "c"},
    ]
    ranked = pipeline.rerank("query", candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["title"] == "B"  # highest score
