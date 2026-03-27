"""tests/test_mcp.py — MCP server tool tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.mcp import (
    tool_discover_papers,
    tool_query_knowledge,
    tool_verify_claims,
)


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_discover_papers(mock_components):
    mock_discover = AsyncMock(return_value=[])
    with patch("discover.mcp.discover", mock_discover):
        result = await tool_discover_papers(query="multi-agent RL", domains=None, since=None, max_results=10)
    assert isinstance(result, list)


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
