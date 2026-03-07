"""Tests for the Embedder adapter (hash fallback + sentence-transformers)."""
import math

from sage.memory.embedder import EMBEDDING_DIM, Embedder


class TestEmbedderReturnsVector:
    """test_embedder_returns_vector: returns list[float] of correct length."""

    def test_returns_list_of_floats(self):
        emb = Embedder(force_hash=True)
        vec = emb.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_no_nan_or_inf(self):
        emb = Embedder(force_hash=True)
        vec = emb.embed("test string with special chars !@#$%")
        for v in vec:
            assert not math.isnan(v), "embedding contains NaN"
            assert not math.isinf(v), "embedding contains Inf"

    def test_values_in_reasonable_range(self):
        emb = Embedder(force_hash=True)
        vec = emb.embed("some text")
        # Hash embedder should produce values in [-1, 1] range
        for v in vec:
            assert -1.0 <= v <= 1.0, f"value {v} out of [-1, 1] range"


class TestEmbedderDeterministic:
    """test_embedder_deterministic: same input -> same output."""

    def test_same_input_same_output(self):
        emb = Embedder(force_hash=True)
        v1 = emb.embed("deterministic test")
        v2 = emb.embed("deterministic test")
        assert v1 == v2

    def test_different_input_different_output(self):
        emb = Embedder(force_hash=True)
        v1 = emb.embed("input A")
        v2 = emb.embed("input B")
        assert v1 != v2

    def test_deterministic_across_instances(self):
        emb1 = Embedder(force_hash=True)
        emb2 = Embedder(force_hash=True)
        v1 = emb1.embed("cross-instance test")
        v2 = emb2.embed("cross-instance test")
        assert v1 == v2


class TestEmbedderBatch:
    """test_embedder_batch: batch returns correct number of vectors."""

    def test_batch_count(self):
        emb = Embedder(force_hash=True)
        texts = ["alpha", "beta", "gamma"]
        results = emb.embed_batch(texts)
        assert len(results) == 3

    def test_batch_dimensions(self):
        emb = Embedder(force_hash=True)
        texts = ["one", "two", "three", "four"]
        results = emb.embed_batch(texts)
        for vec in results:
            assert len(vec) == EMBEDDING_DIM

    def test_batch_matches_single(self):
        emb = Embedder(force_hash=True)
        texts = ["hello", "world"]
        batch_results = emb.embed_batch(texts)
        single_results = [emb.embed(t) for t in texts]
        assert batch_results == single_results

    def test_batch_empty(self):
        emb = Embedder(force_hash=True)
        results = emb.embed_batch([])
        assert results == []


class TestHashEmbedderForced:
    """test_hash_embedder_forced: force_hash=True works."""

    def test_force_hash_creates_hash_embedder(self):
        emb = Embedder(force_hash=True)
        assert not emb.is_semantic

    def test_force_hash_still_embeds(self):
        emb = Embedder(force_hash=True)
        vec = emb.embed("force hash test")
        assert len(vec) == EMBEDDING_DIM

    def test_empty_string(self):
        emb = Embedder(force_hash=True)
        vec = emb.embed("")
        assert len(vec) == EMBEDDING_DIM
        # Even empty string should produce valid floats
        for v in vec:
            assert not math.isnan(v)
            assert not math.isinf(v)


class TestIsSemanticProperty:
    """test_is_semantic_property: is_semantic is False for hash mode."""

    def test_hash_mode_not_semantic(self):
        emb = Embedder(force_hash=True)
        assert emb.is_semantic is False

    def test_is_semantic_is_bool(self):
        emb = Embedder(force_hash=True)
        assert isinstance(emb.is_semantic, bool)
