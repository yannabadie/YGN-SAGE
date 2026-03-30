"""tests/test_rag.py — RAG pipeline tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from discover.rag import RAGPipeline


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search_dense.return_value = [
        {"id": "p1", "score": 0.9, "payload": {"title": "Paper A", "abstract": "Abstract A", "domain": "marl"}},
        {"id": "p2", "score": 0.8, "payload": {"title": "Paper B", "abstract": "Abstract B", "domain": "marl"}},
    ]
    store.search_sparse.return_value = [
        {"id": "p2", "score": 0.95, "payload": {"title": "Paper B", "abstract": "Abstract B", "domain": "marl"}},
        {"id": "p3", "score": 0.7, "payload": {"title": "Paper C", "abstract": "Abstract C", "domain": "marl"}},
    ]
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_text.return_value = np.random.rand(768).astype(np.float32)
    emb.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [1], "values": [0.5]})
    emb.rerank.side_effect = lambda q, candidates, top_k: candidates[:top_k]
    return emb


def test_hybrid_search(mock_store, mock_embedder):
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder)
    results = rag.hybrid_search("multi-agent RL", top_k=3)
    assert len(results) <= 3
    ids = [r["id"] for r in results]
    assert "p2" in ids


@pytest.mark.asyncio
async def test_query_returns_answer(mock_store, mock_embedder):
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="Based on the papers, multi-agent RL is...")
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder, llm=mock_llm)
    answer = await rag.query("What is multi-agent RL?")
    assert len(answer) > 0


@pytest.mark.asyncio
async def test_query_without_llm_returns_summaries(mock_store, mock_embedder):
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder, llm=None)
    answer = await rag.query("test query")
    assert "Paper A" in answer or "Paper B" in answer
