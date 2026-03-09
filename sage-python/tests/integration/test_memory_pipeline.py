"""Integration tests for the memory pipeline (no mocks).

Every test uses real implementations with in-memory backends.
No API keys required, no mocks, no patches.
"""
import pytest

from sage.memory.episodic import EpisodicMemory
from sage.memory.semantic import SemanticMemory
from sage.memory.causal import CausalMemory
from sage.memory.relevance_gate import RelevanceGate
from sage.memory.write_gate import WriteGate
from sage.memory.memory_agent import ExtractionResult


# ---------------------------------------------------------------------------
# Episodic Memory (async, in-memory backend)
# ---------------------------------------------------------------------------

class TestEpisodicMemoryIntegration:
    """Test episodic memory with real in-memory backend."""

    async def test_store_and_retrieve(self):
        mem = EpisodicMemory()  # in-memory by default (db_path=None)
        await mem.store("question-1", "What is Python?", {"type": "question"})
        await mem.store("answer-1", "A programming language", {"type": "answer"})
        results = await mem.search("Python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    async def test_cross_key_search(self):
        mem = EpisodicMemory()
        await mem.store("deploy-aws", "Deploy to AWS", {})
        await mem.store("deploy-gcp", "Deploy to GCP", {})
        results = await mem.search("Deploy")
        assert len(results) >= 2

    async def test_store_overwrite(self):
        mem = EpisodicMemory()
        await mem.store("k1", "original content")
        await mem.store("k1", "updated content")
        count = await mem.count()
        assert count == 1
        results = await mem.search("updated")
        assert len(results) == 1
        assert results[0]["content"] == "updated content"

    async def test_delete_entry(self):
        mem = EpisodicMemory()
        await mem.store("k1", "some content")
        assert await mem.count() == 1
        deleted = await mem.delete("k1")
        assert deleted is True
        assert await mem.count() == 0

    async def test_delete_nonexistent(self):
        mem = EpisodicMemory()
        deleted = await mem.delete("nonexistent")
        assert deleted is False

    async def test_search_no_match(self):
        mem = EpisodicMemory()
        await mem.store("k1", "Python programming")
        results = await mem.search("quantum")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Semantic Memory (sync, in-memory entity graph)
# ---------------------------------------------------------------------------

class TestSemanticMemoryIntegration:
    """Test semantic memory with real entity extraction."""

    def test_add_and_retrieve_entities(self):
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["Python", "Rust"],
            relationships=[("Python", "competes_with", "Rust")],
        )
        sem.add_extraction(extraction)
        context = sem.get_context_for("Tell me about Python")
        assert "Python" in context
        assert "competes_with" in context

    def test_empty_context_for_unknown_topic(self):
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["Python"],
            relationships=[],
        )
        sem.add_extraction(extraction)
        context = sem.get_context_for("Tell me about quantum physics")
        # "quantum physics" is not an entity, so no context
        assert isinstance(context, str)
        assert context == ""

    def test_multi_hop_query(self):
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["Python", "Rust", "LLVM"],
            relationships=[
                ("Python", "competes_with", "Rust"),
                ("Rust", "compiles_via", "LLVM"),
            ],
        )
        sem.add_extraction(extraction)
        # 1-hop from Python should reach Rust
        rels = sem.query_entities("Python", hops=1)
        assert len(rels) >= 1
        # 2-hops should reach LLVM
        rels = sem.query_entities("Python", hops=2)
        assert len(rels) >= 2

    def test_entity_count(self):
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["A", "B", "C"],
            relationships=[],
        )
        sem.add_extraction(extraction)
        assert sem.entity_count() == 3

    def test_dedup_relations(self):
        sem = SemanticMemory()
        extraction = ExtractionResult(
            entities=["A", "B"],
            relationships=[("A", "r", "B"), ("A", "r", "B")],
        )
        sem.add_extraction(extraction)
        rels = sem.query_entities("A")
        assert len(rels) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Causal Memory (sync, in-memory causal graph)
# ---------------------------------------------------------------------------

class TestCausalMemoryIntegration:
    """Test causal memory with real entity-relation graph."""

    def test_add_and_query_causal_chain(self):
        mem = CausalMemory()
        mem.add_entity("test", {"type": "action"})
        mem.add_entity("deploy", {"type": "action"})
        mem.add_causal_edge("test", "deploy", cause_type="precedes")
        chain = mem.get_causal_chain("test")
        assert "test" in chain
        assert "deploy" in chain

    def test_causal_ancestors(self):
        mem = CausalMemory()
        mem.add_entity("design")
        mem.add_entity("code")
        mem.add_entity("test")
        mem.add_causal_edge("design", "code", cause_type="enables")
        mem.add_causal_edge("code", "test", cause_type="enables")
        ancestors = mem.get_causal_ancestors("test")
        assert "code" in ancestors
        assert "design" in ancestors

    def test_temporal_ordering(self):
        mem = CausalMemory()
        mem.add_entity("first")
        mem.add_entity("second")
        mem.add_entity("third")
        order = mem.temporal_order()
        assert order == ["first", "second", "third"]

    def test_context_generation(self):
        mem = CausalMemory()
        mem.add_entity("deploy")
        mem.add_entity("rollback")
        mem.add_relation("deploy", "followed_by", "rollback")
        context = mem.get_context_for("What happened after deploy?")
        assert "deploy" in context
        assert "rollback" in context

    def test_entity_eviction(self):
        mem = CausalMemory(max_entities=2)
        mem.add_entity("a")
        mem.add_entity("b")
        mem.add_entity("c")  # should evict "a"
        assert mem.entity_count() == 2
        assert not mem.has_entity("a")
        assert mem.has_entity("b")
        assert mem.has_entity("c")


# ---------------------------------------------------------------------------
# Relevance Gate (sync, no LLM)
# ---------------------------------------------------------------------------

class TestRelevanceGateIntegration:
    """Test relevance gate with real scoring."""

    def test_relevant_context_passes(self):
        gate = RelevanceGate(threshold=0.3)
        score = gate.score("Write a Python function for sorting", "Python programming functions sorting algorithms")
        assert score >= 0.3
        assert gate.is_relevant("Write a Python function for sorting", "Python programming functions sorting algorithms")

    def test_irrelevant_context_blocked(self):
        gate = RelevanceGate(threshold=0.3)
        score = gate.score("What is the weather?", "Python programming functions")
        assert isinstance(score, float)
        assert score >= 0.0
        # "weather" has no overlap with "Python programming functions"
        assert not gate.is_relevant("What is the weather today?", "Python programming functions")

    def test_empty_input(self):
        gate = RelevanceGate(threshold=0.3)
        assert gate.score("", "some context") == 0.0
        assert gate.score("some task", "") == 0.0
        assert not gate.is_relevant("task", "")

    def test_identical_text_scores_high(self):
        gate = RelevanceGate(threshold=0.3)
        text = "implement binary search algorithm in Python"
        score = gate.score(text, text)
        assert score >= 0.5  # high overlap when same text


# ---------------------------------------------------------------------------
# Write Gate (sync, no LLM)
# ---------------------------------------------------------------------------

class TestWriteGateIntegration:
    """Test write gate deduplication and confidence thresholding."""

    def test_duplicate_detection(self):
        gate = WriteGate(threshold=0.5)
        first = gate.evaluate("new content", confidence=0.8)
        assert first.allowed is True
        second = gate.evaluate("new content", confidence=0.8)
        assert second.allowed is False
        assert "duplicate" in second.reason.lower()

    def test_confidence_threshold(self):
        gate = WriteGate(threshold=0.5)
        low = gate.evaluate("low confidence content", confidence=0.2)
        assert low.allowed is False
        assert "confidence" in low.reason.lower()

    def test_empty_content_blocked(self):
        gate = WriteGate(threshold=0.5)
        result = gate.evaluate("", confidence=0.9)
        assert result.allowed is False
        assert "empty" in result.reason.lower()

    def test_stats_tracking(self):
        gate = WriteGate(threshold=0.5)
        gate.evaluate("good content", confidence=0.8)  # allowed
        gate.evaluate("low", confidence=0.1)  # blocked: confidence
        gate.evaluate("good content", confidence=0.8)  # blocked: duplicate
        stats = gate.stats()
        assert stats["writes"] == 1
        assert stats["abstentions"] == 2
        assert stats["abstention_rate"] > 0.0

    def test_bounded_dedup_eviction(self):
        gate = WriteGate(threshold=0.0, max_dedup_size=3)
        for i in range(5):
            gate.evaluate(f"content-{i}", confidence=0.9)
        # Oldest entries should have been evicted from dedup set
        # content-0 and content-1 were evicted, so re-storing them should succeed
        result = gate.evaluate("content-0", confidence=0.9)
        assert result.allowed is True
