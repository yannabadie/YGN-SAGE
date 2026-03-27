"""tests/test_store.py — KnowledgeStore (Qdrant local wrapper)."""
from __future__ import annotations

import numpy as np
import pytest

from discover.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    """In-memory Qdrant store for tests."""
    return KnowledgeStore(path=str(tmp_path / "test_store"))


@pytest.fixture
def sample_paper():
    return {
        "id": "arxiv-2505.12601",
        "title": "kNN Routing for LLMs",
        "authors": ["Author A"],
        "abstract": "We propose kNN-based routing for LLM selection.",
        "domain": "marl",
        "source": "arxiv",
        "year": 2025,
        "citation_count": 42,
    }


@pytest.fixture
def sample_dense():
    return np.random.rand(768).astype(np.float32)


@pytest.fixture
def sample_sparse():
    return {"indices": [1, 5, 100, 500], "values": [0.8, 1.2, 0.5, 0.3]}


def test_store_initializes_collections(store):
    collections = store.list_collections()
    assert "papers" in collections
    assert "claims" in collections


def test_upsert_and_get_paper(store, sample_paper, sample_dense, sample_sparse):
    store.upsert_paper(
        paper_id=sample_paper["id"],
        dense_vector=sample_dense,
        sparse_vector=sample_sparse,
        payload=sample_paper,
    )
    result = store.get_paper(sample_paper["id"])
    assert result is not None
    assert result["title"] == "kNN Routing for LLMs"


def test_upsert_deduplicates(store, sample_paper, sample_dense, sample_sparse):
    store.upsert_paper(sample_paper["id"], sample_dense, sample_sparse, sample_paper)
    updated = {**sample_paper, "citation_count": 99}
    store.upsert_paper(sample_paper["id"], sample_dense, sample_sparse, updated)
    result = store.get_paper(sample_paper["id"])
    assert result["citation_count"] == 99


def test_search_dense(store, sample_paper, sample_dense, sample_sparse):
    store.upsert_paper(sample_paper["id"], sample_dense, sample_sparse, sample_paper)
    results = store.search_dense(sample_dense, limit=5)
    assert len(results) >= 1
    assert results[0]["id"] == sample_paper["id"]


def test_search_with_domain_filter(store, sample_dense, sample_sparse):
    store.upsert_paper("p1", sample_dense, sample_sparse, {"domain": "marl", "title": "A"})
    other_dense = np.random.rand(768).astype(np.float32)
    store.upsert_paper("p2", other_dense, sample_sparse, {"domain": "memory_systems", "title": "B"})
    results = store.search_dense(sample_dense, limit=10, domain="marl")
    assert all(r["payload"]["domain"] == "marl" for r in results)


def test_paper_count(store, sample_paper, sample_dense, sample_sparse):
    assert store.paper_count() == 0
    store.upsert_paper(sample_paper["id"], sample_dense, sample_sparse, sample_paper)
    assert store.paper_count() == 1


def test_upsert_claim(store):
    claim_dense = np.random.rand(768).astype(np.float32)
    store.upsert_claim(
        claim_id="claim-001",
        dense_vector=claim_dense,
        payload={
            "statement": "kNN achieves 92% accuracy",
            "paper_id": "arxiv-2505.12601",
            "claim_type": "finding",
            "smt_verified": False,
            "smt_status": "not_checked",
        },
    )
    assert store.claim_count() == 1


def test_get_claims_for_paper(store):
    dense = np.random.rand(768).astype(np.float32)
    store.upsert_claim("c1", dense, {"paper_id": "p1", "statement": "A", "smt_status": "not_checked"})
    store.upsert_claim("c2", dense, {"paper_id": "p1", "statement": "B", "smt_status": "not_checked"})
    store.upsert_claim("c3", dense, {"paper_id": "p2", "statement": "C", "smt_status": "not_checked"})
    claims = store.get_claims_for_paper("p1")
    assert len(claims) == 2
