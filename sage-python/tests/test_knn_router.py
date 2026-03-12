"""Tests for kNN routing (arXiv 2505.12601)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from sage.strategy.knn_router import KnnRouter, KnnRoutingResult

GT_PATH = Path(__file__).parent.parent / "config" / "routing_ground_truth.json"


# -- Fixtures -----------------------------------------------------------------

def _make_mock_embedder():
    """Create a mock embedder with keyword-based clustering."""
    emb = MagicMock()
    emb.is_semantic = True
    emb.is_hash_fallback = False
    emb._backend = "mock"

    rng_base = np.random.default_rng(42)
    s1_center = rng_base.normal(0, 0.005, 768).astype(np.float32)
    s1_center[:10] = 1.0
    s1_center /= np.linalg.norm(s1_center)
    s2_center = rng_base.normal(0, 0.005, 768).astype(np.float32)
    s2_center[10:20] = 1.0
    s2_center /= np.linalg.norm(s2_center)
    s3_center = rng_base.normal(0, 0.005, 768).astype(np.float32)
    s3_center[20:30] = 1.0
    s3_center /= np.linalg.norm(s3_center)

    s3_kw = ["prove", "verify", "formal", "induction", "invariant", "z3",
             "correctness", "safety", "liveness", "temporal", "ltl",
             "undecidable", "riemann", "cantor", "turing-complete",
             "ramsey", "linearizability", "preservation"]
    s2_kw = ["write", "implement", "code", "debug", "build", "create",
             "refactor", "optimize", "design", "api", "function", "class",
             "test", "fix", "parse", "migration", "deploy", "pipeline",
             "scraper", "decorator", "concurrent", "neural", "sql",
             "caching", "bloom", "comparison", "dynamic programming",
             "paginated", "endpoint"]

    def mock_embed(text):
        lower = text.lower()
        if any(kw in lower for kw in s3_kw):
            center = s3_center
        elif any(kw in lower for kw in s2_kw):
            center = s2_center
        else:
            center = s1_center
        seed = int(abs(hash(text))) % (2**31)
        noise = np.random.default_rng(seed).normal(0, 0.01, 768).astype(np.float32)
        vec = center + noise
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    emb.embed = mock_embed
    emb.embed_batch = lambda ts: [mock_embed(t) for t in ts]
    return emb


@pytest.fixture
def mock_embedder():
    return _make_mock_embedder()


@pytest.fixture
def mock_hash_embedder():
    emb = MagicMock()
    emb.is_semantic = False
    emb.is_hash_fallback = True
    emb._backend = "hash"
    emb.embed = lambda t: [0.0] * 768
    emb.embed_batch = lambda ts: [[0.0] * 768 for _ in ts]
    return emb


@pytest.fixture
def ready_router(mock_embedder):
    """A KnnRouter pre-built from ground truth with mock embedder."""
    router = KnnRouter(embedder=mock_embedder, k=5)
    assert router.build_from_ground_truth(GT_PATH)
    return router


@pytest.fixture
def exemplar_npz(mock_embedder, tmp_path):
    """Create a temporary .npz from ground truth."""
    router = KnnRouter(embedder=mock_embedder, k=5)
    save_path = tmp_path / "routing_exemplars.npz"
    router.build_from_ground_truth(GT_PATH, save_path=save_path)
    return save_path


# -- Unit tests ---------------------------------------------------------------

class TestKnnRouterInit:
    def test_no_file_not_ready(self, mock_embedder):
        router = KnnRouter(
            exemplar_path=Path("/nonexistent/file.npz"),
            embedder=mock_embedder,
        )
        assert not router.is_ready

    def test_loads_npz(self, exemplar_npz, mock_embedder):
        router = KnnRouter(exemplar_path=exemplar_npz, embedder=mock_embedder)
        assert router.is_ready
        assert router.exemplar_count == 50

    def test_corrupt_npz_graceful(self, tmp_path, mock_embedder):
        bad_path = tmp_path / "bad.npz"
        bad_path.write_bytes(b"not a real npz file")
        router = KnnRouter(exemplar_path=bad_path, embedder=mock_embedder)
        assert not router.is_ready

    def test_wrong_dimensions(self, tmp_path, mock_embedder):
        npz_path = tmp_path / "wrong_dim.npz"
        np.savez_compressed(
            npz_path,
            embeddings=np.zeros((10, 256), dtype=np.float32),
            labels=np.ones(10, dtype=np.int32),
        )
        router = KnnRouter(exemplar_path=npz_path, embedder=mock_embedder)
        assert not router.is_ready


class TestKnnRouting:
    def test_routes_s1_task(self, ready_router):
        result = ready_router.route("What is the capital of France?")
        assert result is not None
        assert result.system in (1, 2, 3)
        assert result.method == "knn"
        # Nearest neighbor should be exact match (task 1 in ground truth)
        assert result.nearest_distance > 0.99

    def test_routes_s2_task(self, ready_router):
        result = ready_router.route("Write a Python function to sort a list")
        assert result is not None
        assert result.system == 2

    def test_routes_s3_task(self, ready_router):
        result = ready_router.route("Prove that sqrt(2) is irrational by contradiction")
        assert result is not None
        assert result.system == 3

    def test_refuses_hash_embedder(self, mock_hash_embedder):
        router = KnnRouter(embedder=mock_hash_embedder)
        router.build_from_ground_truth(GT_PATH)
        # build_from_ground_truth refuses hash embedder, so not ready
        assert not router.is_ready

    def test_route_returns_none_on_hash(self, exemplar_npz, mock_hash_embedder):
        """Even with loaded exemplars, hash embedder is rejected at route time."""
        router = KnnRouter(exemplar_path=exemplar_npz, embedder=mock_hash_embedder)
        result = router.route("What is 2+2?")
        assert result is None

    def test_confidence_bounds(self, ready_router):
        result = ready_router.route("What is the capital of France?")
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_k_neighbors_returned(self, ready_router):
        result = ready_router.route("Write a function in Python")
        assert result is not None
        assert len(result.k_distances) == 5
        assert len(result.k_labels) == 5

    def test_distances_sorted_descending(self, ready_router):
        result = ready_router.route("What is TCP?")
        assert result is not None
        for i in range(len(result.k_distances) - 1):
            assert result.k_distances[i] >= result.k_distances[i + 1]

    def test_not_ready_returns_none(self, mock_embedder):
        router = KnnRouter(
            exemplar_path=Path("/nonexistent.npz"),
            embedder=mock_embedder,
        )
        assert router.route("anything") is None


class TestKnnBuildFromGroundTruth:
    def test_build_in_memory(self, mock_embedder):
        # Use nonexistent path to prevent auto-loading from config/
        router = KnnRouter(exemplar_path=Path("/nonexistent.npz"), embedder=mock_embedder)
        assert not router.is_ready
        ok = router.build_from_ground_truth(GT_PATH)
        assert ok
        assert router.is_ready
        assert router.exemplar_count == 50

    def test_build_and_save(self, mock_embedder, tmp_path):
        router = KnnRouter(embedder=mock_embedder)
        save_path = tmp_path / "saved.npz"
        ok = router.build_from_ground_truth(GT_PATH, save_path=save_path)
        assert ok
        assert save_path.exists()

        # Reload from saved file
        router2 = KnnRouter(exemplar_path=save_path, embedder=mock_embedder)
        assert router2.is_ready
        assert router2.exemplar_count == 50

    def test_build_refuses_hash(self, mock_hash_embedder):
        router = KnnRouter(embedder=mock_hash_embedder)
        ok = router.build_from_ground_truth(GT_PATH)
        assert not ok
        assert not router.is_ready


class TestKnnNpzRoundTrip:
    def test_npz_round_trip_routing(self, exemplar_npz, mock_embedder):
        """Load from .npz and verify routing still works."""
        router = KnnRouter(exemplar_path=exemplar_npz, embedder=mock_embedder, k=5)
        assert router.is_ready

        # Should produce a result (exact system may vary with mock)
        result = router.route("Prove this invariant holds")
        assert result is not None
        assert result.system in (1, 2, 3)
        assert 0.0 <= result.confidence <= 1.0


class TestKnnAdaptiveIntegration:
    def test_adaptive_router_uses_knn(self, ready_router):
        from sage.strategy.adaptive_router import AdaptiveRouter

        router = AdaptiveRouter(knn_router=ready_router)
        assert router.has_knn
        result = router.route_adaptive("Prove sqrt(2) is irrational")
        assert result.method == "knn"
        assert result.decision.system == 3

    def test_adaptive_assess_complexity_knn(self, ready_router):
        from sage.strategy.adaptive_router import AdaptiveRouter

        router = AdaptiveRouter(knn_router=ready_router)
        profile = router.assess_complexity("What is 2+2?")
        assert profile.reasoning.startswith("knn_")

    def test_adaptive_fallback_without_knn(self):
        from sage.strategy.adaptive_router import AdaptiveRouter

        router = AdaptiveRouter()
        assert not router.has_knn
        result = router.route_adaptive("What is 2+2?")
        assert result.method in ("heuristic", "rust_s0")


class TestKnnGroundTruthAccuracy:
    """Test kNN accuracy on the ground truth dataset with mock embedder."""

    def test_loo_accuracy_above_80pct(self, mock_embedder):
        """Leave-one-out cross-validation should exceed 80%."""
        with open(GT_PATH) as f:
            gt = json.load(f)

        tasks = gt["tasks"]
        texts = [t["task"] for t in tasks]
        labels = np.array([t["expected_system"] for t in tasks], dtype=np.int32)

        vectors = mock_embedder.embed_batch(texts)
        embeddings = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings /= norms

        correct = 0
        for i in range(len(embeddings)):
            bank = np.delete(embeddings, i, axis=0)
            bank_labels = np.delete(labels, i, axis=0)
            sims = bank @ embeddings[i]
            top_k = np.argpartition(sims, -5)[-5:]

            votes: dict[int, float] = {}
            for idx in top_k:
                label = int(bank_labels[idx])
                votes[label] = votes.get(label, 0.0) + float(sims[idx])

            predicted = max(votes, key=lambda s: votes[s])
            if predicted == labels[i]:
                correct += 1

        accuracy = correct / len(embeddings)
        assert accuracy >= 0.80, f"LOO-CV accuracy {accuracy:.1%} < 80%"
