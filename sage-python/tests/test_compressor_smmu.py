"""Tests for compressor -> S-MMU wiring (Task 2).

Verifies that MemoryCompressor.step() calls compact_to_arrow_with_meta
after compression, extracts keywords from the summary, and passes
embeddings from the Embedder.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sage.memory.compressor import MemoryCompressor
from sage.memory.embedder import EMBEDDING_DIM, Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm(summary_text: str = "SUMMARY: The agent solved a complex routing problem.\nDISCOVERIES:\n- Found bug in parser"):
    """Create a mock LLM that returns a fixed summary."""
    llm = AsyncMock()
    response = MagicMock()
    response.content = summary_text
    llm.generate = AsyncMock(return_value=response)
    return llm


def _make_working_memory(event_count: int = 25):
    """Create a mock WorkingMemory with enough events to trigger compression."""
    wm = MagicMock()
    wm.agent_id = "test-agent"
    wm.event_count.return_value = event_count

    events = [
        {"type": "PERCEIVE", "content": f"Event {i}"}
        for i in range(event_count)
    ]
    wm.recent_events.return_value = events
    wm.compress = MagicMock()
    wm.compact_to_arrow_with_meta = MagicMock(return_value=1)
    return wm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompressorCallsCompactAfterCompression:
    """Verify compact_to_arrow_with_meta is called after compress."""

    @pytest.mark.asyncio
    async def test_compact_called_after_compress(self):
        llm = _make_mock_llm()
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)

        result = await compressor.step(wm)

        assert result is True
        # compress must be called first
        wm.compress.assert_called_once()
        # compact_to_arrow_with_meta must be called after compress
        wm.compact_to_arrow_with_meta.assert_called_once()

    @pytest.mark.asyncio
    async def test_compact_not_called_below_threshold(self):
        llm = _make_mock_llm()
        compressor = MemoryCompressor(llm=llm, compression_threshold=30)
        wm = _make_working_memory(event_count=10)

        result = await compressor.step(wm)

        assert result is False
        wm.compact_to_arrow_with_meta.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_failure_does_not_crash(self):
        """compact_to_arrow_with_meta failure is best-effort — should not raise."""
        llm = _make_mock_llm()
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)
        wm.compact_to_arrow_with_meta.side_effect = RuntimeError("Arrow error")

        # Should not raise
        result = await compressor.step(wm)
        assert result is True  # compression itself still succeeded


class TestCompressorExtractsKeywords:
    """Verify keywords are extracted from the LLM summary."""

    @pytest.mark.asyncio
    async def test_keywords_from_summary(self):
        llm = _make_mock_llm("SUMMARY: The agent solved routing problem efficiently.\nDISCOVERIES:\n- Found optimization")
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)

        await compressor.step(wm)

        call_args = wm.compact_to_arrow_with_meta.call_args
        keywords = call_args[1].get("keywords") or call_args[0][0] if call_args[0] else call_args[1]["keywords"]
        # Keywords should be words > 3 chars from the summary
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert len(keywords) <= 10
        # All keywords should be > 3 chars
        for kw in keywords:
            assert len(kw) > 3, f"keyword '{kw}' should be > 3 chars"

    @pytest.mark.asyncio
    async def test_keywords_max_10(self):
        """Even with a long summary, keywords should be capped at 10."""
        long_summary = "SUMMARY: " + " ".join([f"word{i}longtext" for i in range(50)])
        llm = _make_mock_llm(long_summary)
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)

        await compressor.step(wm)

        call_args = wm.compact_to_arrow_with_meta.call_args
        keywords = call_args[1].get("keywords") or call_args[0][0] if call_args[0] else call_args[1]["keywords"]
        assert len(keywords) <= 10


class TestCompressorPassesEmbedding:
    """Verify embedder.embed() is called and embedding is passed."""

    @pytest.mark.asyncio
    async def test_embedding_passed_to_compact(self):
        llm = _make_mock_llm()
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)

        await compressor.step(wm)

        call_args = wm.compact_to_arrow_with_meta.call_args
        embedding = call_args[1].get("embedding")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_embedding_from_summary_text(self):
        """The embedding should be computed from the summary text."""
        summary = "SUMMARY: Important discovery about memory systems."
        llm = _make_mock_llm(summary)
        embedder = Embedder(force_hash=True)
        compressor = MemoryCompressor(llm=llm)
        compressor.embedder = embedder

        wm = _make_working_memory(event_count=25)
        await compressor.step(wm)

        # The embedding passed should match what the embedder produces for the summary
        call_args = wm.compact_to_arrow_with_meta.call_args
        embedding = call_args[1].get("embedding")
        expected = embedder.embed("Important discovery about memory systems.")
        assert embedding == expected

    @pytest.mark.asyncio
    async def test_summary_passed_to_compact(self):
        """The LLM summary text should be passed to compact_to_arrow_with_meta."""
        llm = _make_mock_llm("SUMMARY: Agent discovered critical bug.\nDISCOVERIES:\n- Bug found")
        compressor = MemoryCompressor(llm=llm)
        wm = _make_working_memory(event_count=25)

        await compressor.step(wm)

        call_args = wm.compact_to_arrow_with_meta.call_args
        summary = call_args[1].get("summary")
        assert summary is not None
        assert "Agent discovered critical bug" in summary
