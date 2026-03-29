"""tests/test_integration.py — End-to-end integration tests."""
from __future__ import annotations

import tempfile
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.pipeline import run_pipeline


@pytest.mark.asyncio
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_full_nightly_pipeline(mock_discover):
    """Test complete nightly flow: discover -> curate -> embed -> ingest to Qdrant."""
    from discover.discovery import PaperCandidate

    mock_discover.return_value = [
        PaperCandidate(
            paper_id="integration-001",
            title="Integration Test Paper on Multi-Agent Systems",
            authors=["Test Author"],
            abstract="We propose a novel multi-agent reinforcement learning approach " * 10,
            source="arxiv",
            domain="marl",
            published=date.today(),
            pdf_url=None,
            citation_count=15,
        ),
    ]

    from discover.store import KnowledgeStore
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = KnowledgeStore(path=tmpdir)
        try:
            mock_embedder = MagicMock()
            mock_embedder.embed_paper.return_value = (
                np.random.rand(768).astype(np.float32),
                {"indices": [1, 5, 10], "values": [0.8, 0.5, 0.3]},
            )

            mock_llm = AsyncMock()
            mock_llm.generate.return_value = MagicMock(
                content='[{"score": 8, "reason": "Relevant to MARL", "key_insights": ["novel approach"]}]'
            )

            report = await run_pipeline(
                mode="nightly",
                llm=mock_llm,
                store=store,
                embedder=mock_embedder,
            )

            assert report.discovered == 1
            assert report.curated >= 1
            assert report.ingested >= 1

            paper = store.get_paper("integration-001")
            assert paper is not None
            assert paper["title"] == "Integration Test Paper on Multi-Agent Systems"
        finally:
            store.close()


@pytest.mark.asyncio
async def test_store_to_rag_flow():
    """Test: ingest paper -> search -> RAG answer."""
    from discover.store import KnowledgeStore
    from discover.rag import RAGPipeline

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = KnowledgeStore(path=tmpdir)
        try:
            dense = np.random.rand(768).astype(np.float32)
            sparse = {"indices": [1, 5], "values": [0.8, 0.5]}
            store.upsert_paper("p1", dense, sparse, {
                "title": "Multi-Agent RL Survey",
                "abstract": "A comprehensive survey of MARL techniques.",
                "domain": "marl",
            })

            mock_embedder = MagicMock()
            mock_embedder.embed_text.return_value = dense
            mock_embedder.embed_paper.return_value = (dense, sparse)
            mock_embedder.rerank.side_effect = lambda q, c, top_k: c[:top_k]

            rag = RAGPipeline(store=store, embedder=mock_embedder, llm=None)
            answer = await rag.query("multi-agent RL")
            assert "Multi-Agent RL Survey" in answer
        finally:
            store.close()
