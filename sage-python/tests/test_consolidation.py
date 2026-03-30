"""Tests for the inter-tier memory consolidation pipeline.

Verifies that:
1. Unconsolidated episodic entries are processed
2. Already-consolidated entries are skipped
3. Entities flow from episodic -> semantic
4. Sequential steps create causal edges
5. Empty/short content is skipped
6. Failures are gracefully handled
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.memory.causal import CausalMemory
from sage.memory.consolidator import MemoryConsolidator, ConsolidationResult


@dataclass
class MockExtraction:
    entities: list[str]
    relationships: list[list[str]]
    summary: str = ""


class TestConsolidation:
    """Test the MemoryConsolidator."""

    @pytest.fixture
    def mock_episodic(self):
        ep = AsyncMock()
        ep.list_all = AsyncMock(return_value=[
            {"key": "step-1", "content": "The Parser component processes the input AST correctly.", "metadata": {}},
            {"key": "step-2", "content": "The Optimizer reduced the intermediate representation size.", "metadata": {}},
            {"key": "step-3", "content": "Already done.", "metadata": {"consolidated": True}},
        ])
        ep.update = AsyncMock(return_value=True)
        return ep

    @pytest.fixture
    def mock_semantic(self):
        sem = MagicMock()
        sem.add_extraction = MagicMock()
        return sem

    @pytest.fixture
    def mock_memory_agent(self):
        agent = AsyncMock()
        agent.extract = AsyncMock(side_effect=[
            MockExtraction(entities=["Parser", "AST"], relationships=[["Parser", "processes", "AST"]], summary="Parser processes AST"),
            MockExtraction(entities=["Optimizer", "IR"], relationships=[["Optimizer", "reduced", "IR"]], summary="Optimizer reduces IR"),
        ])
        return agent

    @pytest.mark.asyncio
    async def test_consolidation_processes_unconsolidated(self, mock_episodic, mock_semantic, mock_memory_agent):
        cm = CausalMemory()
        consolidator = MemoryConsolidator(
            episodic=mock_episodic,
            semantic=mock_semantic,
            causal=cm,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()

        assert result.processed == 2  # step-1 and step-2
        assert result.skipped_already_consolidated == 1  # step-3
        assert result.entities_added == 4  # Parser, AST, Optimizer, IR

    @pytest.mark.asyncio
    async def test_consolidation_feeds_semantic(self, mock_episodic, mock_semantic, mock_memory_agent):
        consolidator = MemoryConsolidator(
            episodic=mock_episodic,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        await consolidator.consolidate()

        assert mock_semantic.add_extraction.call_count == 2

    @pytest.mark.asyncio
    async def test_consolidation_creates_causal_edges(self, mock_episodic, mock_semantic, mock_memory_agent):
        cm = CausalMemory()
        consolidator = MemoryConsolidator(
            episodic=mock_episodic,
            semantic=mock_semantic,
            causal=cm,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()

        # step-1 entities [Parser, AST], step-2 entities [Optimizer, IR]
        # Causal edge: AST (last of step-1) -> Optimizer (first of step-2)
        assert result.causal_edges_added == 1
        chain = cm.get_causal_chain("AST")
        assert "Optimizer" in chain

    @pytest.mark.asyncio
    async def test_consolidation_marks_as_consolidated(self, mock_episodic, mock_semantic, mock_memory_agent):
        consolidator = MemoryConsolidator(
            episodic=mock_episodic,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        await consolidator.consolidate()

        # update() should be called for each processed entry
        assert mock_episodic.update.call_count == 2
        # Check that consolidated=True is in the metadata
        for call in mock_episodic.update.call_args_list:
            meta = call.kwargs.get("metadata") or call.args[1] if len(call.args) > 1 else {}
            if "metadata" in call.kwargs:
                meta = call.kwargs["metadata"]
            assert meta.get("consolidated") is True

    @pytest.mark.asyncio
    async def test_consolidation_idempotent(self, mock_semantic, mock_memory_agent):
        """Already-consolidated entries are skipped on re-run."""
        ep = AsyncMock()
        ep.list_all = AsyncMock(return_value=[
            {"key": "step-1", "content": "Some content here for testing.", "metadata": {"consolidated": True}},
            {"key": "step-2", "content": "More content for testing purposes.", "metadata": {"consolidated": True}},
        ])

        consolidator = MemoryConsolidator(
            episodic=ep,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()

        assert result.processed == 0
        assert result.skipped_already_consolidated == 2
        mock_memory_agent.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidation_skips_short_content(self, mock_semantic, mock_memory_agent):
        ep = AsyncMock()
        ep.list_all = AsyncMock(return_value=[
            {"key": "step-1", "content": "short", "metadata": {}},  # < 20 chars
        ])

        consolidator = MemoryConsolidator(
            episodic=ep,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()
        assert result.processed == 0
        mock_memory_agent.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidation_handles_extraction_failure(self, mock_semantic):
        ep = AsyncMock()
        ep.list_all = AsyncMock(return_value=[
            {"key": "step-1", "content": "Some valid content that should be processed.", "metadata": {}},
        ])
        ep.update = AsyncMock(return_value=True)

        agent = AsyncMock()
        agent.extract = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        consolidator = MemoryConsolidator(
            episodic=ep,
            semantic=mock_semantic,
            causal=None,
            memory_agent=agent,
        )

        result = await consolidator.consolidate()
        assert result.processed == 0  # Failed extraction -> not processed
        mock_semantic.add_extraction.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidation_without_causal(self, mock_episodic, mock_semantic, mock_memory_agent):
        """Consolidation works without causal memory."""
        consolidator = MemoryConsolidator(
            episodic=mock_episodic,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()
        assert result.processed == 2
        assert result.causal_edges_added == 0  # No causal memory -> no edges

    @pytest.mark.asyncio
    async def test_empty_episodic(self, mock_semantic, mock_memory_agent):
        ep = AsyncMock()
        ep.list_all = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            episodic=ep,
            semantic=mock_semantic,
            causal=None,
            memory_agent=mock_memory_agent,
        )

        result = await consolidator.consolidate()
        assert result.processed == 0
        assert result.entities_added == 0
