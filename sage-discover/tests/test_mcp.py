"""tests/test_mcp.py — MCP server tool tests."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.mcp import (
    tool_discover_papers,
    tool_curate_papers,
    tool_query_knowledge,
    tool_verify_claims,
)


def _candidate(**overrides):
    data = {
        "paper_id": "p1",
        "title": "Test Paper",
        "authors": ["Alice"],
        "abstract": "Test abstract",
        "source": "arxiv",
        "domain": "marl",
        "published": date(2026, 4, 8),
        "citation_count": 7,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class _FakeStore:
    def __init__(self) -> None:
        self.papers: dict[str, dict] = {}

    def upsert_paper(self, paper_id, dense_vector, sparse_vector, payload) -> None:
        self.papers[paper_id] = {
            "_paper_id": paper_id,
            "_dense": dense_vector,
            "_sparse": sparse_vector,
            **payload,
        }

    def get_paper(self, paper_id):
        return self.papers.get(paper_id)


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_discover_papers(mock_components):
    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_paper.return_value = ([0.1, 0.2], {"indices": [1], "values": [0.3]})
    mock_components.return_value = {
        "store": mock_store,
        "embedder": mock_embedder,
        "llm": None,
        "rag": MagicMock(),
    }
    mock_discover = AsyncMock(return_value=[_candidate()])
    with patch("discover.mcp.discover", mock_discover):
        result = await tool_discover_papers(query="multi-agent RL", domains=None, since=None, max_results=10)
    assert result["ok"] is True
    assert result["papers"][0]["paper_id"] == "p1"
    assert result["persisted_paper_ids"] == ["p1"]
    mock_embedder.embed_paper.assert_called_once_with("Test Paper", "Test abstract")
    mock_store.upsert_paper.assert_called_once()


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_discover_then_curate_uses_persisted_store(mock_components):
    fake_store = _FakeStore()
    mock_embedder = MagicMock()
    mock_embedder.embed_paper.return_value = ([0.1, 0.2], {"indices": [1], "values": [0.3]})
    mock_components.return_value = {
        "store": fake_store,
        "embedder": mock_embedder,
        "llm": None,
        "rag": MagicMock(),
    }

    with patch("discover.mcp.discover", AsyncMock(return_value=[_candidate()])):
        discovered = await tool_discover_papers(query="multi-agent RL", domains=None, since=None, max_results=10)
    curated = await tool_curate_papers(discovered["persisted_paper_ids"])

    assert curated == {
        "ok": True,
        "requested_paper_ids": ["p1"],
        "found": [{"paper_id": "p1", "title": "Test Paper", "relevance_score": 0.0}],
        "missing_paper_ids": [],
        "found_count": 1,
        "missing_count": 0,
    }


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_discover_papers_surfaces_persistence_failures(mock_components):
    mock_store = MagicMock()
    mock_store.upsert_paper.side_effect = RuntimeError("disk full")
    mock_embedder = MagicMock()
    mock_embedder.embed_paper.return_value = ([0.1, 0.2], {"indices": [1], "values": [0.3]})
    mock_components.return_value = {
        "store": mock_store,
        "embedder": mock_embedder,
        "llm": None,
        "rag": MagicMock(),
    }

    with patch("discover.mcp.discover", AsyncMock(return_value=[_candidate()])):
        result = await tool_discover_papers(query="multi-agent RL", domains=None, since=None, max_results=10)

    assert result["ok"] is False
    assert result["error"]["code"] == "persistence_failed"
    assert result["persisted_paper_ids"] == []
    assert result["papers"][0]["paper_id"] == "p1"


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_curate_papers_reports_missing_ids(mock_components):
    fake_store = _FakeStore()
    fake_store.papers["p1"] = {"title": "Stored Paper", "relevance_score": 0.5}
    mock_components.return_value = {"store": fake_store}

    result = await tool_curate_papers(["p1", "p2"])

    assert result == {
        "ok": False,
        "requested_paper_ids": ["p1", "p2"],
        "found": [{"paper_id": "p1", "title": "Stored Paper", "relevance_score": 0.5}],
        "missing_paper_ids": ["p2"],
        "found_count": 1,
        "missing_count": 1,
    }


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_query_knowledge(mock_components):
    mock_rag = MagicMock()
    mock_rag.query = AsyncMock(return_value="Answer about multi-agent RL")
    mock_components.return_value = {"rag": mock_rag}
    result = await tool_query_knowledge(question="What is MARL?", top_k=5, domain=None)
    assert len(result) > 0


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_verify_claims(mock_components):
    mock_store = MagicMock()
    mock_store.get_paper.return_value = {"title": "Test", "abstract": "Test abstract about method achieving 90% accuracy"}
    mock_store.get_claims_for_paper.return_value = []
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content='[{"statement": "achieves 90%", "type": "finding", "confidence": 0.9}]')
    mock_embedder = MagicMock()
    mock_components.return_value = {"store": mock_store, "llm": mock_llm, "embedder": mock_embedder}
    result = await tool_verify_claims(paper_id="p1")
    assert "paper_id" in result or "error" in result
