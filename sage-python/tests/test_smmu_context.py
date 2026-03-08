"""Tests for S-MMU context retrieval helper (Task 5).

Verifies that retrieve_smmu_context() correctly queries the S-MMU graph
and formats results as injectable context strings.
"""
from __future__ import annotations

import pytest

from sage.memory.working import WorkingMemory
from sage.memory.smmu_context import retrieve_smmu_context


class TestRetrieveReturnsEmptyWhenNoChunks:
    """When the S-MMU has no chunks, retrieval should return empty string."""

    def test_retrieve_returns_empty_when_no_chunks(self):
        wm = WorkingMemory(agent_id="test-empty")
        result = retrieve_smmu_context(wm)
        assert result == ""

    def test_retrieve_returns_string_type(self):
        wm = WorkingMemory(agent_id="test-type")
        result = retrieve_smmu_context(wm)
        assert isinstance(result, str)


class TestRetrieveAfterCompaction:
    """After compaction (mock mode), retrieval should not error."""

    def test_retrieve_returns_string_after_compaction(self):
        """In mock mode, S-MMU is not populated so result is '', but must not error."""
        wm = WorkingMemory(agent_id="test-compacted")
        # Add some events and compact (mock returns chunk_id=0, smmu_chunk_count=0)
        for i in range(5):
            wm.add_event("TEST", f"event {i}")
        wm.compact_to_arrow()

        result = retrieve_smmu_context(wm)
        # Mock S-MMU returns 0 chunks, so result should be ""
        assert isinstance(result, str)
        # Should not raise


class TestRetrieveWithCustomParams:
    """Verify custom parameters are accepted without error."""

    def test_custom_max_hops(self):
        wm = WorkingMemory(agent_id="test-hops")
        result = retrieve_smmu_context(wm, max_hops=5)
        assert isinstance(result, str)

    def test_custom_top_k(self):
        wm = WorkingMemory(agent_id="test-topk")
        result = retrieve_smmu_context(wm, top_k=10)
        assert isinstance(result, str)

    def test_custom_weights(self):
        wm = WorkingMemory(agent_id="test-weights")
        result = retrieve_smmu_context(wm, weights=(0.5, 3.0, 2.0, 0.5))
        assert isinstance(result, str)


class TestRetrieveWithMockedChunks:
    """Test formatting logic with a mocked working memory that returns chunks."""

    def test_formats_results_as_bullet_list(self):
        """When chunks exist, output should have a header and bullet items."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 3
        wm.retrieve_relevant_chunks.return_value = [
            (2, 0.95),
            (1, 0.72),
            (0, 0.55),
        ]
        # No summary available — falls back to bare chunk ID format
        wm.get_chunk_summary.return_value = ""

        result = retrieve_smmu_context(wm, top_k=5)
        assert "S-MMU" in result or "graph" in result.lower() or "chunk" in result.lower()
        # Should mention chunk IDs in fallback format (all scores >= 0.5 min_score)
        assert "Chunk 2" in result
        assert "Chunk 1" in result
        assert "Chunk 0" in result

    def test_respects_top_k(self):
        """Only top_k results should appear."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 10
        wm.retrieve_relevant_chunks.return_value = [
            (i, 1.0 - i * 0.1) for i in range(10)
        ]
        wm.get_chunk_summary.return_value = ""

        result = retrieve_smmu_context(wm, top_k=3)
        # Should only show 3 items
        lines = [l for l in result.strip().split("\n") if l.strip().startswith("-")]
        assert len(lines) == 3

    def test_returns_empty_when_no_hits(self):
        """If retrieve_relevant_chunks returns empty list, result is ''."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 5
        wm.retrieve_relevant_chunks.return_value = []

        result = retrieve_smmu_context(wm)
        assert result == ""

    def test_exception_returns_empty(self):
        """Any exception in retrieval should return '' (best-effort)."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.side_effect = RuntimeError("Rust panic")

        result = retrieve_smmu_context(wm)
        assert result == ""


class TestRetrieveWithChunkSummaries:
    """Test that chunk summaries are included when available."""

    def test_summary_included_with_score(self):
        """When get_chunk_summary returns text, it should appear with score prefix."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 3
        wm.retrieve_relevant_chunks.return_value = [
            (2, 0.95),
            (1, 0.72),
        ]
        wm.get_chunk_summary.side_effect = lambda cid: {
            2: "User asked about Python async patterns",
            1: "Agent resolved import error in main.py",
        }.get(cid, "")

        result = retrieve_smmu_context(wm, top_k=5)
        assert "[0.95] User asked about Python async patterns" in result
        assert "[0.72] Agent resolved import error in main.py" in result
        # Should NOT contain bare "Chunk N" format for these
        assert "Chunk 2" not in result
        assert "Chunk 1" not in result

    def test_mixed_summary_and_fallback(self):
        """Chunks with summaries show text; chunks without fall back to ID format."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 3
        wm.retrieve_relevant_chunks.return_value = [
            (2, 0.90),
            (1, 0.60),
            (0, 0.52),
        ]
        wm.get_chunk_summary.side_effect = lambda cid: {
            2: "Discussed memory architecture",
            1: "",   # no summary
            0: "",   # no summary
        }.get(cid, "")

        result = retrieve_smmu_context(wm, top_k=5)
        assert "[0.90] Discussed memory architecture" in result
        assert "Chunk 1 (relevance: 0.60)" in result
        assert "Chunk 0 (relevance: 0.52)" in result

    def test_fallback_when_get_chunk_summary_missing(self):
        """If working memory has no get_chunk_summary, fall back gracefully."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 2
        wm.retrieve_relevant_chunks.return_value = [
            (1, 0.80),
        ]
        # Simulate missing method
        wm.get_chunk_summary.side_effect = AttributeError("no such method")

        result = retrieve_smmu_context(wm, top_k=5)
        # Should fall back to bare ID format
        assert "Chunk 1 (relevance: 0.80)" in result

    def test_header_line_present(self):
        """The header line should always be present when there are results."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 1
        wm.retrieve_relevant_chunks.return_value = [(0, 0.50)]
        wm.get_chunk_summary.return_value = "Test summary"

        result = retrieve_smmu_context(wm, top_k=5)
        assert result.startswith("[S-MMU Graph Memory]")


class TestMinScoreFiltering:
    """Task 8: min_score=0.5 filters low-relevance chunks."""

    def test_chunks_below_min_score_filtered(self):
        """Chunks with score < 0.5 should be excluded."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 4
        wm.retrieve_relevant_chunks.return_value = [
            (3, 0.95),
            (2, 0.60),
            (1, 0.49),  # below threshold
            (0, 0.30),  # below threshold
        ]
        wm.get_chunk_summary.return_value = ""

        result = retrieve_smmu_context(wm, top_k=5)
        lines = [l for l in result.strip().split("\n") if l.strip().startswith("-")]
        assert len(lines) == 2  # Only 0.95 and 0.60

    def test_all_chunks_below_min_score_returns_empty(self):
        """If all chunks score below min_score, return empty string."""
        from unittest.mock import MagicMock

        wm = MagicMock()
        wm.smmu_chunk_count.return_value = 2
        wm.retrieve_relevant_chunks.return_value = [
            (1, 0.40),
            (0, 0.20),
        ]

        result = retrieve_smmu_context(wm)
        assert result == ""
