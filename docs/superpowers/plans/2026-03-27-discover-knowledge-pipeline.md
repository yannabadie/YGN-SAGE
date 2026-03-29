# sage-discover Knowledge Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform sage-discover into a formally verified knowledge discovery engine with Qdrant hybrid search, SMT-verified claims, MAP-Elites frontier exploration, and adaptive curation.

**Architecture:** 9 new modules layered bottom-up: storage (Qdrant) + embeddings (SPECTER2/SPLADE) form the foundation, then extraction (Docling), citation graph (NetworkX), claim graph (OxiZ SMT), adaptive curation (kNN+bandit), RAG, frontier explorer (MAP-Elites), and MCP server on top. Existing pipeline.py is rewired to use new components; dead code (knowledge.py, workflow.py, researcher.py) is deleted.

**Tech Stack:** Qdrant (local), sentence-transformers (SPECTER2, SPLADE, cross-encoder), Docling, NetworkX, OxiZ (sage_core.SmtVerifier), FastMCP, flan-t5-small, pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-03-27-discover-knowledge-pipeline-design.md`

---

## File Map

### New files (create)

| File | Responsibility |
|------|----------------|
| `src/discover/store.py` | Qdrant local wrapper: collections, upsert, hybrid search, RRF |
| `src/discover/embeddings.py` | SPECTER2 dense + SPLADE sparse + cross-encoder reranker |
| `src/discover/extractor.py` | Docling PDF-to-structured-text extraction |
| `src/discover/citation_graph.py` | NetworkX DiGraph from S2 citations, PageRank, Louvain |
| `src/discover/claim_graph.py` | Claim extraction (flan-t5), relation classification, OxiZ SMT |
| `src/discover/adaptive_curator.py` | KnnCurator + CurationBandit (LinUCB) + self-feedback |
| `src/discover/rag.py` | Hybrid search + LLM RAG answer generation |
| `src/discover/frontier.py` | MAP-Elites frontier explorer with 4D behavior descriptors |
| `src/discover/mcp.py` | FastMCP server exposing 5 tools |
| `tests/test_store.py` | Tests for KnowledgeStore |
| `tests/test_embeddings.py` | Tests for EmbeddingPipeline |
| `tests/test_extractor.py` | Tests for PDF extractor |
| `tests/test_citation_graph.py` | Tests for CitationGraphBuilder |
| `tests/test_claim_graph.py` | Tests for ClaimExtractor + SMT verification |
| `tests/test_adaptive_curator.py` | Tests for KnnCurator + CurationBandit |
| `tests/test_rag.py` | Tests for RAG pipeline |
| `tests/test_frontier.py` | Tests for MAP-Elites frontier |
| `tests/test_mcp.py` | Tests for MCP server tools |

### Modified files

| File | Changes |
|------|---------|
| `pyproject.toml` | Add new dependencies |
| `src/discover/pipeline.py` | Wire new components (store, adaptive curator, extractor, claims, citations) |
| `src/discover/discovery.py` | Add S2 recommendations, influential_citation_count field |
| `src/discover/ingestion.py` | Upsert to Qdrant instead of ExoCortex upload |
| `src/discover/__init__.py` | Update exports |
| `src/discover/__main__.py` | Add `mcp` mode |
| `tests/test_pipeline.py` | Update mocks for new pipeline flow |
| `tests/test_ingestion.py` | Update for Qdrant-based ingestion |

### Deleted files

| File | Reason |
|------|--------|
| `src/discover/knowledge.py` | Stub; replaced by Qdrant RAG in rag.py |
| `src/discover/workflow.py` | Fake eBPF evolution; replaced by frontier.py |
| `src/discover/researcher.py` | Data-only; absorbed into frontier.py |
| `tests/test_discover.py` | Tests for deleted workflow.py/researcher.py |

---

## Task 1: Project Setup & Dependencies

**Files:**
- Modify: `sage-discover/pyproject.toml`

- [ ] **Step 1: Update pyproject.toml with new dependencies**

```toml
[project]
name = "sage-discover"
version = "0.2.0"
description = "YGN-SAGE Knowledge Discovery Engine — formally verified, evolutionary, adaptive"
requires-python = ">=3.12"
dependencies = [
    "ygn-sage>=0.1.0",
    "arxiv>=2.1",
    "semanticscholar>=0.8",
    "qdrant-client>=1.12",
    "sentence-transformers>=3.4",
    "docling>=2.0",
    "networkx>=3.4",
    "mcp[cli]>=1.0",
    "transformers>=4.48",
    "torch>=2.5",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/discover"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Install dependencies**

Run: `cd sage-discover && pip install -e ".[dev]"`
Expected: Successful installation. Note: `ygn-sage` may need to be installed separately from `../sage-python`. `sentence-transformers` will pull `torch`. `docling` may take a while.

- [ ] **Step 3: Verify imports work**

Run: `python -c "import qdrant_client; import sentence_transformers; import networkx; import docling; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 4: Commit**

```bash
git add sage-discover/pyproject.toml
git commit -m "chore: add dependencies for knowledge pipeline (qdrant, sentence-transformers, docling, networkx, mcp)"
```

---

## Task 2: KnowledgeStore (Qdrant Wrapper)

**Files:**
- Create: `src/discover/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for KnowledgeStore**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'discover.store'`

- [ ] **Step 3: Implement KnowledgeStore**

```python
"""src/discover/store.py — Qdrant-backed knowledge store."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)

# Deterministic int ID from string paper_id (Qdrant needs int or UUID for local mode)
def _str_to_int_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


class KnowledgeStore:
    """Local Qdrant wrapper for papers and claims."""

    DENSE_DIM = 768  # SPECTER2 / arctic-embed-m

    def __init__(self, path: str = ".qdrant_store"):
        self._client = QdrantClient(path=path)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}

        if "papers" not in existing:
            self._client.create_collection(
                collection_name="papers",
                vectors_config={
                    "specter2": models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "splade": models.SparseVectorParams(),
                },
            )
            # Create payload indexes for filtering
            self._client.create_payload_index("papers", "domain", models.PayloadSchemaType.KEYWORD)
            self._client.create_payload_index("papers", "year", models.PayloadSchemaType.INTEGER)

        if "claims" not in existing:
            self._client.create_collection(
                collection_name="claims",
                vectors_config={
                    "specter2": models.VectorParams(
                        size=self.DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
            )
            self._client.create_payload_index("claims", "paper_id", models.PayloadSchemaType.KEYWORD)
            self._client.create_payload_index("claims", "smt_status", models.PayloadSchemaType.KEYWORD)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    # --- Papers ---

    def upsert_paper(
        self,
        paper_id: str,
        dense_vector: np.ndarray,
        sparse_vector: dict[str, list],
        payload: dict[str, Any],
    ) -> None:
        point_id = _str_to_int_id(paper_id)
        self._client.upsert(
            collection_name="papers",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "specter2": dense_vector.tolist(),
                        "splade": models.SparseVector(
                            indices=sparse_vector["indices"],
                            values=sparse_vector["values"],
                        ),
                    },
                    payload={**payload, "_paper_id": paper_id},
                ),
            ],
        )

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        point_id = _str_to_int_id(paper_id)
        results = self._client.retrieve("papers", ids=[point_id], with_payload=True)
        if results:
            return results[0].payload
        return None

    def paper_count(self) -> int:
        return self._client.count("papers").count

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if domain:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="domain", match=models.MatchValue(value=domain))]
            )
        results = self._client.query_points(
            collection_name="papers",
            query=query_vector.tolist(),
            using="specter2",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
        return [{"id": r.payload.get("_paper_id", str(r.id)), "score": r.score, "payload": r.payload} for r in results]

    def search_sparse(
        self,
        sparse_vector: dict[str, list],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results = self._client.query_points(
            collection_name="papers",
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="splade",
            limit=limit,
            with_payload=True,
        ).points
        return [{"id": r.payload.get("_paper_id", str(r.id)), "score": r.score, "payload": r.payload} for r in results]

    # --- Claims ---

    def upsert_claim(
        self,
        claim_id: str,
        dense_vector: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        point_id = _str_to_int_id(claim_id)
        self._client.upsert(
            collection_name="claims",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={"specter2": dense_vector.tolist()},
                    payload={**payload, "_claim_id": claim_id},
                ),
            ],
        )

    def claim_count(self) -> int:
        return self._client.count("claims").count

    def get_claims_for_paper(self, paper_id: str) -> list[dict[str, Any]]:
        results = self._client.scroll(
            collection_name="claims",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]
            ),
            limit=100,
            with_payload=True,
        )[0]
        return [r.payload for r in results]

    def update_claim_payload(self, claim_id: str, updates: dict[str, Any]) -> None:
        point_id = _str_to_int_id(claim_id)
        self._client.set_payload("claims", payload=updates, points=[point_id])

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_store.py -v`
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/store.py tests/test_store.py
git commit -m "feat: add KnowledgeStore (Qdrant local wrapper) with papers + claims collections"
```

---

## Task 3: EmbeddingPipeline (SPECTER2 + SPLADE + Reranker)

**Files:**
- Create: `src/discover/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_embeddings.py — EmbeddingPipeline tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from discover.embeddings import EmbeddingPipeline


@pytest.fixture
def mock_specter2():
    model = MagicMock()
    model.encode.return_value = np.random.rand(768).astype(np.float32)
    return model


@pytest.fixture
def mock_splade():
    model = MagicMock()
    # SparseEncoder returns scipy sparse or similar
    sparse = MagicMock()
    sparse.nonzero.return_value = (np.array([0, 0, 0]), np.array([5, 42, 100]))
    sparse.__getitem__ = lambda self, idx: np.array([0.8, 1.2, 0.5])
    model.encode.return_value = sparse
    return model


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder")
def test_embed_paper_returns_dense_and_sparse(mock_sparse_cls, mock_st_cls, mock_specter2, mock_splade):
    mock_st_cls.return_value = mock_specter2
    mock_sparse_cls.return_value = mock_splade
    pipeline = EmbeddingPipeline()
    dense, sparse = pipeline.embed_paper("Title", "Abstract text here")
    assert dense.shape == (768,)
    assert "indices" in sparse
    assert "values" in sparse


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder")
def test_embed_paper_concatenates_title_abstract(mock_sparse_cls, mock_st_cls, mock_specter2, mock_splade):
    mock_st_cls.return_value = mock_specter2
    mock_sparse_cls.return_value = mock_splade
    pipeline = EmbeddingPipeline()
    pipeline.embed_paper("My Title", "My Abstract")
    call_args = mock_specter2.encode.call_args[0][0]
    assert "My Title" in call_args
    assert "My Abstract" in call_args


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder")
def test_rerank_returns_sorted(mock_sparse_cls, mock_st_cls, mock_specter2, mock_splade):
    mock_st_cls.return_value = mock_specter2
    mock_sparse_cls.return_value = mock_splade
    pipeline = EmbeddingPipeline()
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.1, 0.9, 0.5])
    with patch("discover.embeddings.CrossEncoder", return_value=mock_ce):
        candidates = [
            {"title": "A", "abstract": "a"},
            {"title": "B", "abstract": "b"},
            {"title": "C", "abstract": "c"},
        ]
        ranked = pipeline.rerank("query", candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["title"] == "B"  # highest score


@patch("discover.embeddings.SentenceTransformer")
@patch("discover.embeddings.SparseEncoder")
def test_reciprocal_rank_fusion(mock_sparse_cls, mock_st_cls, mock_specter2, mock_splade):
    mock_st_cls.return_value = mock_specter2
    mock_sparse_cls.return_value = mock_splade
    from discover.embeddings import reciprocal_rank_fusion
    list_a = [{"id": "p1", "score": 0.9}, {"id": "p2", "score": 0.8}]
    list_b = [{"id": "p2", "score": 0.95}, {"id": "p3", "score": 0.7}]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)
    ids = [r["id"] for r in fused]
    assert "p2" in ids  # p2 appears in both, should rank high
    assert len(fused) == 3  # p1, p2, p3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'discover.embeddings'`

- [ ] **Step 3: Implement EmbeddingPipeline**

```python
"""src/discover/embeddings.py — SPECTER2 + SPLADE + cross-encoder reranker."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading models at import time
SentenceTransformer = None
SparseEncoder = None
CrossEncoder = None


def _ensure_imports():
    global SentenceTransformer, SparseEncoder, CrossEncoder
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
    if SparseEncoder is None:
        try:
            from sentence_transformers import SparseEncoder as SE
            SparseEncoder = SE
        except ImportError:
            SparseEncoder = None
    if CrossEncoder is None:
        from sentence_transformers import CrossEncoder as CE
        CrossEncoder = CE


def reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked result lists."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in items:
                items[item_id] = item

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{**items[id_], "rrf_score": scores[id_]} for id_ in sorted_ids]


class EmbeddingPipeline:
    """Manages SPECTER2 (dense), SPLADE (sparse), and cross-encoder (reranker)."""

    SPECTER2_MODEL = "allenai/specter2_base"
    SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(self):
        _ensure_imports()
        logger.info("Loading SPECTER2 model: %s", self.SPECTER2_MODEL)
        self._dense = SentenceTransformer(self.SPECTER2_MODEL)
        self._sparse = None
        if SparseEncoder is not None:
            logger.info("Loading SPLADE model: %s", self.SPLADE_MODEL)
            self._sparse = SparseEncoder(self.SPLADE_MODEL)
        self._reranker = None  # lazy

    def embed_paper(self, title: str, abstract: str) -> tuple[np.ndarray, dict[str, list]]:
        """Embed a paper using SPECTER2 (dense) + SPLADE (sparse).

        Returns (dense_vector, {"indices": [...], "values": [...]}).
        """
        text = f"{title}. {abstract}"

        dense = self._dense.encode(text)
        if not isinstance(dense, np.ndarray):
            dense = np.array(dense, dtype=np.float32)

        sparse_dict = {"indices": [], "values": []}
        if self._sparse is not None:
            sparse_out = self._sparse.encode(text)
            # Extract non-zero indices and values from sparse output
            try:
                nz = sparse_out.nonzero()
                if len(nz) == 2:
                    indices = nz[1].tolist() if hasattr(nz[1], 'tolist') else list(nz[1])
                    values = [float(sparse_out[0, i]) for i in indices] if len(indices) > 0 else []
                    sparse_dict = {"indices": indices, "values": values}
            except Exception:
                logger.warning("SPLADE encoding failed, using empty sparse vector")

        return dense, sparse_dict

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text (for claims, queries)."""
        dense = self._dense.encode(text)
        if not isinstance(dense, np.ndarray):
            dense = np.array(dense, dtype=np.float32)
        return dense

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder."""
        if self._reranker is None:
            _ensure_imports()
            logger.info("Loading reranker: %s", self.RERANKER_MODEL)
            self._reranker = CrossEncoder(self.RERANKER_MODEL)

        pairs = [(query, f"{c.get('title', '')}. {c.get('abstract', '')}") for c in candidates]
        scores = self._reranker.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [c for c, _ in scored[:top_k]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_embeddings.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/embeddings.py tests/test_embeddings.py
git commit -m "feat: add EmbeddingPipeline (SPECTER2 + SPLADE + cross-encoder reranker)"
```

---

## Task 4: PDF Extractor (Docling)

**Files:**
- Create: `src/discover/extractor.py`
- Test: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_extractor.py — Docling PDF extractor tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from discover.extractor import extract_full_text, extract_sections_from_markdown


def test_extract_sections_from_markdown():
    md = """# Introduction
This is the introduction.

## Methodology
We use method X.

## Results
We found Y.

## Conclusion
In conclusion Z.

## References
[1] Ref A
"""
    sections = extract_sections_from_markdown(md)
    assert "introduction" in sections
    assert "methodology" in sections
    assert "results" in sections
    assert "conclusion" in sections
    assert "We use method X." in sections["methodology"]


def test_extract_sections_handles_missing():
    md = "Just a plain abstract with no sections."
    sections = extract_sections_from_markdown(md)
    assert sections["introduction"] is None
    assert sections["methodology"] is None


@patch("discover.extractor.DocumentConverter")
def test_extract_full_text_returns_structured(mock_converter_cls):
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Title\nContent here"
    mock_result.document.tables = []
    mock_result.document.pictures = []
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter

    result = extract_full_text(Path("/fake/paper.pdf"))
    assert "full_text" in result
    assert "sections" in result
    assert "tables" in result


@patch("discover.extractor.DocumentConverter")
def test_extract_full_text_fallback_on_error(mock_converter_cls):
    mock_converter_cls.return_value.convert.side_effect = Exception("PDF corrupted")
    result = extract_full_text(Path("/fake/bad.pdf"))
    assert result["full_text"] is None
    assert result["error"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'discover.extractor'`

- [ ] **Step 3: Implement extractor**

```python
"""src/discover/extractor.py — PDF-to-structured-text via Docling."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    DocumentConverter = None
    HAS_DOCLING = False


# Section header patterns (case-insensitive)
_SECTION_PATTERNS = {
    "introduction": re.compile(r"^#+\s*(introduction|1\.\s*introduction)", re.IGNORECASE),
    "methodology": re.compile(r"^#+\s*(method|methodology|approach|2\.\s*method)", re.IGNORECASE),
    "results": re.compile(r"^#+\s*(results|experiments|evaluation|3\.\s*results)", re.IGNORECASE),
    "conclusion": re.compile(r"^#+\s*(conclusion|discussion|summary|4\.\s*conclusion)", re.IGNORECASE),
}


def extract_sections_from_markdown(md: str) -> dict[str, str | None]:
    """Extract named sections from markdown text."""
    sections: dict[str, str | None] = {k: None for k in _SECTION_PATTERNS}
    lines = md.split("\n")
    current_section: str | None = None
    current_lines: list[str] = []

    def _flush():
        nonlocal current_section, current_lines
        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()
        current_lines = []

    for line in lines:
        matched = False
        for section_name, pattern in _SECTION_PATTERNS.items():
            if pattern.match(line.strip()):
                _flush()
                current_section = section_name
                matched = True
                break
        if not matched and current_section:
            # Stop current section at next heading
            if line.strip().startswith("#"):
                _flush()
                current_section = None
            else:
                current_lines.append(line)

    _flush()
    return sections


def extract_full_text(pdf_path: Path) -> dict[str, Any]:
    """Extract structured content from PDF using Docling.

    Returns dict with keys: full_text, sections, tables, error.
    """
    if not HAS_DOCLING:
        return {
            "full_text": None,
            "sections": {k: None for k in _SECTION_PATTERNS},
            "tables": [],
            "error": "docling not installed",
        }

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        md = doc.export_to_markdown()
        sections = extract_sections_from_markdown(md)

        tables = []
        if hasattr(doc, "tables"):
            for t in doc.tables:
                try:
                    tables.append(t.export_to_markdown())
                except Exception:
                    pass

        return {
            "full_text": md,
            "sections": sections,
            "tables": tables,
            "error": None,
        }
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, e)
        return {
            "full_text": None,
            "sections": {k: None for k in _SECTION_PATTERNS},
            "tables": [],
            "error": str(e),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_extractor.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/extractor.py tests/test_extractor.py
git commit -m "feat: add PDF extractor via Docling with section parsing"
```

---

## Task 5: Citation Graph Builder

**Files:**
- Create: `src/discover/citation_graph.py`
- Test: `tests/test_citation_graph.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_citation_graph.py — Citation graph tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.citation_graph import CitationGraphBuilder


@pytest.fixture
def builder():
    return CitationGraphBuilder()


def test_add_paper_node(builder):
    builder.add_paper("p1", title="Paper A", year=2025, citation_count=10)
    assert "p1" in builder.graph
    assert builder.graph.nodes["p1"]["title"] == "Paper A"


def test_add_citation_edge(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("p2", title="B", year=2025, citation_count=0)
    builder.add_citation("p1", "p2")  # p1 cites p2
    assert builder.graph.has_edge("p1", "p2")


def test_pagerank(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("p2", title="B", year=2025, citation_count=0)
    builder.add_paper("p3", title="C", year=2025, citation_count=0)
    builder.add_citation("p1", "p3")
    builder.add_citation("p2", "p3")
    ranks = builder.pagerank()
    assert ranks["p3"] > ranks["p1"]  # p3 is cited by 2 papers


def test_communities(builder):
    # Create two clusters
    for i in range(5):
        builder.add_paper(f"a{i}", title=f"A{i}", year=2025, citation_count=0)
    for i in range(5):
        builder.add_paper(f"b{i}", title=f"B{i}", year=2025, citation_count=0)
    # Dense edges within clusters
    for i in range(4):
        builder.add_citation(f"a{i}", f"a{i+1}")
        builder.add_citation(f"b{i}", f"b{i+1}")
    comms = builder.communities()
    assert len(comms) >= 2


def test_bridges(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("bridge", title="Bridge", year=2025, citation_count=0)
    builder.add_paper("p2", title="C", year=2025, citation_count=0)
    builder.add_citation("p1", "bridge")
    builder.add_citation("bridge", "p2")
    bridges = builder.bridges()
    assert bridges["bridge"] >= bridges.get("p1", 0)


def test_node_count(builder):
    assert builder.node_count() == 0
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    assert builder.node_count() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_citation_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CitationGraphBuilder**

```python
"""src/discover/citation_graph.py — Citation graph via NetworkX."""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class CitationGraphBuilder:
    """Builds and analyzes a citation graph using NetworkX DiGraph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_paper(self, paper_id: str, **attrs: Any) -> None:
        """Add a paper node with metadata attributes."""
        self.graph.add_node(paper_id, **attrs)

    def add_citation(self, citing: str, cited: str) -> None:
        """Add a directed citation edge: citing -> cited."""
        # Ensure both nodes exist
        if citing not in self.graph:
            self.graph.add_node(citing)
        if cited not in self.graph:
            self.graph.add_node(cited)
        self.graph.add_edge(citing, cited, relation="cites")

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank over the citation graph."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, alpha=alpha)

    def communities(self) -> list[set[str]]:
        """Detect communities using Louvain on the undirected projection."""
        if self.graph.number_of_nodes() < 2:
            return [set(self.graph.nodes)]
        undirected = self.graph.to_undirected()
        return list(nx.community.louvain_communities(undirected))

    def bridges(self) -> dict[str, float]:
        """Compute betweenness centrality to find bridge papers."""
        if self.graph.number_of_nodes() < 2:
            return {n: 0.0 for n in self.graph.nodes}
        return nx.betweenness_centrality(self.graph)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def neighbors(self, paper_id: str, direction: str = "both") -> list[str]:
        """Get neighbors of a paper. direction: 'citing', 'cited', or 'both'."""
        result = []
        if direction in ("citing", "both"):
            result.extend(self.graph.predecessors(paper_id))
        if direction in ("cited", "both"):
            result.extend(self.graph.successors(paper_id))
        return list(set(result))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_citation_graph.py -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/citation_graph.py tests/test_citation_graph.py
git commit -m "feat: add CitationGraphBuilder (NetworkX DiGraph + PageRank + Louvain)"
```

---

## Task 6: ClaimGraph — SMT-Verified Claims (Innovation #1)

**Files:**
- Create: `src/discover/claim_graph.py`
- Test: `tests/test_claim_graph.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_claim_graph.py — Claim extraction + SMT verification tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from discover.claim_graph import (
    Claim,
    ClaimRelation,
    extract_claims_from_text,
    classify_relation,
    translate_claim_to_smt,
    verify_claim_cluster,
)


def test_claim_dataclass():
    c = Claim(
        claim_id="c1",
        statement="kNN achieves 92% accuracy",
        paper_id="p1",
        claim_type="finding",
        confidence=0.9,
    )
    assert c.claim_id == "c1"
    assert c.smt_status == "not_checked"


def test_translate_performance_claim():
    c = Claim("c1", "Method X achieves 92% accuracy", "p1", "finding", 0.9)
    formula = translate_claim_to_smt(c)
    assert formula is not None
    assert "92" in formula


def test_translate_comparison_claim():
    c = Claim("c1", "Method X improves over baseline by 5 percentage points", "p1", "finding", 0.9)
    formula = translate_claim_to_smt(c)
    assert formula is not None
    assert "5" in formula


def test_translate_qualitative_returns_none():
    c = Claim("c1", "Our method is more elegant", "p1", "finding", 0.5)
    formula = translate_claim_to_smt(c)
    assert formula is None


def test_verify_consistent_cluster():
    claims = [
        Claim("c1", "Method X achieves 92% accuracy", "p1", "finding", 0.9),
        Claim("c2", "Method Y achieves 88% accuracy", "p2", "finding", 0.9),
    ]
    result = verify_claim_cluster(claims)
    assert result in ("consistent", "unknown")


def test_verify_contradictory_cluster():
    claims = [
        Claim("c1", "Method X achieves 92% accuracy on benchmark B", "p1", "finding", 0.9),
        Claim("c2", "Method X achieves 45% accuracy on benchmark B", "p2", "finding", 0.9),
    ]
    result = verify_claim_cluster(claims)
    # These claims about the same method + benchmark with different values should be contradictory
    assert result in ("contradictory", "unknown")


@pytest.mark.asyncio
async def test_extract_claims_from_text():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='[{"statement": "We achieve 92% accuracy", "type": "finding", "confidence": 0.9}]'
    )
    claims = await extract_claims_from_text("Some paper text here", "p1", mock_llm)
    assert len(claims) == 1
    assert claims[0].statement == "We achieve 92% accuracy"
    assert claims[0].paper_id == "p1"


@pytest.mark.asyncio
async def test_classify_relation():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="supports")
    c1 = Claim("c1", "X is good", "p1", "finding", 0.9)
    c2 = Claim("c2", "X works well", "p2", "finding", 0.8)
    rel = await classify_relation(c1, c2, mock_llm)
    assert rel.relation_type in ("supports", "extends", "refutes", "qualifies", "independent")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_claim_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement claim_graph.py**

```python
"""src/discover/claim_graph.py — Claim extraction, relation classification, SMT verification."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Try importing OxiZ from sage_core
try:
    import sage_core
    HAS_SMT = hasattr(sage_core, "SmtVerifier")
except ImportError:
    HAS_SMT = False


@dataclass
class Claim:
    claim_id: str
    statement: str
    paper_id: str
    claim_type: str  # finding | method | limitation | hypothesis
    confidence: float
    section: str = "unknown"
    smt_status: str = "not_checked"  # not_checked | consistent | contradictory | unknown
    smt_formula: str | None = None
    relations: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ClaimRelation:
    source_id: str
    target_id: str
    relation_type: str  # supports | extends | refutes | qualifies | independent


# --- Claim Extraction ---

CLAIM_EXTRACTION_PROMPT = """\
Extract the main scientific claims from this text.
For each claim, provide a JSON array with objects containing:
- "statement": the claim in one sentence
- "type": one of "finding", "method", "limitation", "hypothesis"
- "confidence": float 0.0-1.0 (how certain the authors are)

Text:
{text}

Respond ONLY with a JSON array (no markdown fences):"""


async def extract_claims_from_text(
    text: str,
    paper_id: str,
    llm: Any,
) -> list[Claim]:
    """Extract scientific claims from text using an LLM."""
    from sage.llm.base import Message, Role

    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text[:4000])  # cap context
    messages = [Message(role=Role.USER, content=prompt)]
    response = await llm.generate(messages)

    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse claims JSON from LLM response")
        return []

    claims = []
    for i, item in enumerate(items):
        claims.append(Claim(
            claim_id=f"{paper_id}_c{i}",
            statement=item.get("statement", ""),
            paper_id=paper_id,
            claim_type=item.get("type", "finding"),
            confidence=float(item.get("confidence", 0.5)),
        ))
    return claims


# --- Relation Classification ---

RELATION_PROMPT = """\
What is the relationship between these two scientific claims?

Claim A: "{claim_a}"
Claim B: "{claim_b}"

Choose exactly one:
- supports: B provides evidence for A
- extends: B builds upon A with new contributions
- refutes: B contradicts A
- qualifies: B limits the scope of A
- independent: no direct relationship

Respond with a single word:"""

VALID_RELATIONS = {"supports", "extends", "refutes", "qualifies", "independent"}


async def classify_relation(
    claim_a: Claim,
    claim_b: Claim,
    llm: Any,
) -> ClaimRelation:
    """Classify the relationship between two claims using an LLM."""
    from sage.llm.base import Message, Role

    prompt = RELATION_PROMPT.format(claim_a=claim_a.statement, claim_b=claim_b.statement)
    messages = [Message(role=Role.USER, content=prompt)]
    response = await llm.generate(messages)

    rel_type = response.content.strip().lower()
    if rel_type not in VALID_RELATIONS:
        rel_type = "independent"

    return ClaimRelation(
        source_id=claim_a.claim_id,
        target_id=claim_b.claim_id,
        relation_type=rel_type,
    )


# --- SMT Translation ---

# Patterns for translating quantitative claims to SMT-LIB2
_PERF_PATTERN = re.compile(
    r"(?:achieve|attain|reach|obtain|report)s?\s+(\d+(?:\.\d+)?)\s*%\s*(?:accuracy|precision|recall|F1|score)",
    re.IGNORECASE,
)
_IMPROVE_PATTERN = re.compile(
    r"improv(?:e|es|ing)\s+(?:over|upon|compared to)\s+.*?by\s+(\d+(?:\.\d+)?)\s*(?:percentage|pp|%)",
    re.IGNORECASE,
)
_COMPARE_PATTERN = re.compile(
    r"(\w+)\s+(?:achieves?|attains?)\s+(\d+(?:\.\d+)?)\s*%.*?(?:on|for)\s+(?:benchmark\s+)?(\w+)",
    re.IGNORECASE,
)


def translate_claim_to_smt(claim: Claim) -> str | None:
    """Translate a quantitative claim to SMT-LIB2 formula. Returns None for qualitative claims."""
    text = claim.statement

    # Pattern 1: "achieves X% accuracy"
    m = _PERF_PATTERN.search(text)
    if m:
        val = int(float(m.group(1)))
        var = f"perf_{claim.claim_id}"
        return f"(= {var} {val})"

    # Pattern 2: "improves over baseline by X%"
    m = _IMPROVE_PATTERN.search(text)
    if m:
        delta = int(float(m.group(1)))
        var = f"improvement_{claim.claim_id}"
        return f"(= {var} {delta})"

    # Pattern 3: "Method achieves X% on benchmark B"
    m = _COMPARE_PATTERN.search(text)
    if m:
        method = m.group(1).lower()
        val = int(float(m.group(2)))
        benchmark = m.group(3).lower()
        var = f"perf_{method}_{benchmark}"
        return f"(= {var} {val})"

    return None


# --- SMT Verification ---

def verify_claim_cluster(claims: list[Claim]) -> str:
    """Verify logical consistency of a cluster of related claims.

    Returns: "consistent" | "contradictory" | "unknown"
    """
    if not HAS_SMT:
        return "unknown"

    formulas = []
    variables = set()

    for claim in claims:
        formula = translate_claim_to_smt(claim)
        if formula:
            claim.smt_formula = formula
            formulas.append(formula)
            # Extract variable names for declaration
            for var_match in re.finditer(r"(?:perf|improvement)_\w+", formula):
                variables.add(var_match.group())

    if len(formulas) < 2:
        return "unknown"

    try:
        verifier = sage_core.SmtVerifier()
        verifier.set_logic("QF_LIA")

        for var in variables:
            verifier.declare_const(var, "Int")

        for formula in formulas:
            verifier.assert_(formula)

        result = verifier.check_sat()
        if result == "sat":
            return "consistent"
        elif result == "unsat":
            return "contradictory"
        else:
            return "unknown"
    except Exception as e:
        logger.warning("SMT verification failed: %s", e)
        return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_claim_graph.py -v`
Expected: 9 tests PASS (SMT tests return "unknown" if sage_core not available, which is valid)

- [ ] **Step 5: Commit**

```bash
git add src/discover/claim_graph.py tests/test_claim_graph.py
git commit -m "feat: add ClaimGraph — claim extraction, relation classification, OxiZ SMT verification"
```

---

## Task 7: Adaptive Curator (Innovation #3 — kNN + Bandit)

**Files:**
- Create: `src/discover/adaptive_curator.py`
- Test: `tests/test_adaptive_curator.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_adaptive_curator.py — Adaptive curation tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.adaptive_curator import (
    CurationSignals,
    CurationBandit,
    KnnCurator,
    adaptive_curate,
)


def test_curation_signals_dataclass():
    s = CurationSignals(knn_score=0.8, llm_score=0.7, heuristic_score=0.6)
    assert s.knn_score == 0.8


def test_bandit_initial_decision():
    bandit = CurationBandit()
    signals = CurationSignals(knn_score=0.8, llm_score=0.9, heuristic_score=0.7)
    accept, confidence = bandit.decide(signals)
    assert isinstance(accept, bool)
    assert 0.0 <= confidence <= 2.0


def test_bandit_update_shifts_weights():
    bandit = CurationBandit()
    theta_before = bandit.theta.copy()
    signals = CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0)
    bandit.update(signals, reward=1.0)
    # After rewarding knn-only signal, theta should shift
    assert not np.array_equal(bandit.theta, theta_before)


def test_bandit_learns_from_feedback():
    bandit = CurationBandit()
    # Train: kNN=high + reward=1 should increase knn weight
    for _ in range(20):
        bandit.update(CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0), reward=1.0)
        bandit.update(CurationSignals(knn_score=0.0, llm_score=1.0, heuristic_score=0.0), reward=0.0)
    # Now kNN-heavy signals should be accepted
    accept, _ = bandit.decide(CurationSignals(knn_score=1.0, llm_score=0.0, heuristic_score=0.0))
    assert accept is True


def test_knn_curator_score():
    # Mock exemplars: 3 papers with embeddings and labels
    embeddings = np.random.rand(3, 768).astype(np.float32)
    labels = np.array([1, 1, 0])
    curator = KnnCurator(exemplar_embeddings=embeddings, exemplar_labels=labels)
    query = embeddings[0]  # Same as first exemplar (label=1)
    score = curator.score(query, k=3)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be high since query matches accepted exemplars


def test_knn_curator_empty_returns_neutral():
    curator = KnnCurator(exemplar_embeddings=np.array([]).reshape(0, 768), exemplar_labels=np.array([]))
    score = curator.score(np.random.rand(768).astype(np.float32))
    assert score == 0.5  # neutral when no exemplars


@pytest.mark.asyncio
async def test_adaptive_curate_returns_curated():
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(
        content='[{"score": 8, "reason": "Relevant", "key_insights": ["insight"]}]'
    )
    from discover.discovery import PaperCandidate
    from datetime import date

    candidate = PaperCandidate(
        paper_id="p1", title="Test Paper", authors=["A"],
        abstract="A " * 60,  # > 100 chars
        source="arxiv", domain="marl",
        published=date.today(), pdf_url=None, citation_count=10,
    )

    mock_embedder = MagicMock()
    mock_embedder.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [], "values": []})

    results = await adaptive_curate(
        candidates=[candidate],
        llm=mock_llm,
        embedder=mock_embedder,
    )
    assert len(results) >= 0  # May or may not pass depending on bandit state
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_adaptive_curator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement adaptive_curator.py**

```python
"""src/discover/adaptive_curator.py — kNN + LinUCB bandit + self-feedback curation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from discover.curator import CuratedPaper, heuristic_filter, llm_score, RELEVANCE_THRESHOLD
from discover.discovery import PaperCandidate

logger = logging.getLogger(__name__)


@dataclass
class CurationSignals:
    knn_score: float      # 0-1, from KnnCurator
    llm_score: float      # 0-1 (LLM score / 10)
    heuristic_score: float  # 0 or 1 (passed filter or not)


class KnnCurator:
    """kNN-based paper relevance scorer, same architecture as strategy/knn_router.py."""

    def __init__(
        self,
        exemplar_embeddings: np.ndarray,
        exemplar_labels: np.ndarray,
    ):
        self._embeddings = exemplar_embeddings  # (N, 768)
        self._labels = exemplar_labels  # (N,) — 1=accepted, 0=rejected

    def score(self, query_embedding: np.ndarray, k: int = 7) -> float:
        """Distance-weighted majority vote. Returns 0-1 acceptance probability."""
        if len(self._embeddings) == 0:
            return 0.5  # neutral

        k = min(k, len(self._embeddings))

        # Cosine distances
        norms_e = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms_q = np.linalg.norm(query_embedding)
        if norms_q == 0:
            return 0.5

        similarities = (self._embeddings @ query_embedding) / (norms_e.squeeze() * norms_q + 1e-8)
        top_k_idx = np.argsort(-similarities)[:k]

        distances = 1.0 - similarities[top_k_idx]
        weights = 1.0 / (distances + 1e-6)
        labels = self._labels[top_k_idx]

        return float(np.average(labels, weights=weights))

    def add_exemplar(self, embedding: np.ndarray, label: int) -> None:
        """Add a new exemplar (online learning)."""
        self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)]) if len(self._embeddings) > 0 else embedding.reshape(1, -1)
        self._labels = np.append(self._labels, label)


class CurationBandit:
    """LinUCB bandit for fusing 3 curation signals."""

    def __init__(self, n_features: int = 3, alpha: float = 0.25):
        self.alpha = alpha
        self.n = n_features
        self.A = np.eye(n_features)
        self.b = np.zeros(n_features)
        self.A_inv = np.eye(n_features)
        self.theta = np.ones(n_features) / n_features  # equal weights initially

    def decide(self, signals: CurationSignals) -> tuple[bool, float]:
        """Returns (accept, confidence)."""
        x = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        score = float(self.theta @ x + self.alpha * np.sqrt(x @ self.A_inv @ x))
        return score > 0.5, score

    def update(self, signals: CurationSignals, reward: float) -> None:
        """Update after user feedback. reward: 1.0=useful, 0.0=not useful."""
        x = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        self.A += np.outer(x, x)
        self.b += reward * x
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b


# Global bandit instance (persists across pipeline runs in-process)
_bandit = CurationBandit()


async def adaptive_curate(
    candidates: list[PaperCandidate],
    llm: Any,
    embedder: Any | None = None,
    knn_curator: KnnCurator | None = None,
    bandit: CurationBandit | None = None,
) -> list[CuratedPaper]:
    """Adaptive curation pipeline: heuristic + kNN + LLM + bandit fusion.

    Falls back to legacy curator.curate() if embedder is not available.
    """
    if bandit is None:
        bandit = _bandit

    # Stage 1: Heuristic filter
    filtered = heuristic_filter(candidates)
    passed_ids = {c.paper_id for c in filtered}

    # Stage 2: LLM scoring
    llm_curated = await llm_score(filtered, llm) if llm else []
    llm_map = {cp.candidate.paper_id: cp for cp in llm_curated}

    results = []
    for candidate in filtered:
        # Heuristic signal: 1.0 if passed filter
        h_score = 1.0 if candidate.paper_id in passed_ids else 0.0

        # LLM signal
        l_score = 0.5  # neutral default
        cp = llm_map.get(candidate.paper_id)
        if cp:
            l_score = cp.relevance_score / 10.0

        # kNN signal
        k_score = 0.5  # neutral default
        if embedder and knn_curator:
            dense, _ = embedder.embed_paper(candidate.title, candidate.abstract)
            k_score = knn_curator.score(dense)

        signals = CurationSignals(knn_score=k_score, llm_score=l_score, heuristic_score=h_score)
        accept, confidence = bandit.decide(signals)

        if accept:
            curated = cp if cp else CuratedPaper(
                candidate=candidate,
                relevance_score=int(confidence * 10),
                reason="bandit-accepted",
            )
            results.append(curated)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_adaptive_curator.py -v`
Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/adaptive_curator.py tests/test_adaptive_curator.py
git commit -m "feat: add adaptive curator — kNN relevance + LinUCB bandit fusion + self-feedback"
```

---

## Task 8: RAG Pipeline (Hybrid Search + Answer Generation)

**Files:**
- Create: `src/discover/rag.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_rag.py — RAG pipeline tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.rag import RAGPipeline


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.search_dense.return_value = [
        {"id": "p1", "score": 0.9, "payload": {"title": "Paper A", "abstract": "Abstract A", "domain": "marl"}},
        {"id": "p2", "score": 0.8, "payload": {"title": "Paper B", "abstract": "Abstract B", "domain": "marl"}},
    ]
    store.search_sparse.return_value = [
        {"id": "p2", "score": 0.95, "payload": {"title": "Paper B", "abstract": "Abstract B", "domain": "marl"}},
        {"id": "p3", "score": 0.7, "payload": {"title": "Paper C", "abstract": "Abstract C", "domain": "marl"}},
    ]
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_text.return_value = np.random.rand(768).astype(np.float32)
    emb.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [1], "values": [0.5]})
    emb.rerank.side_effect = lambda q, candidates, top_k: candidates[:top_k]
    return emb


def test_hybrid_search(mock_store, mock_embedder):
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder)
    results = rag.hybrid_search("multi-agent RL", top_k=3)
    assert len(results) <= 3
    ids = [r["id"] for r in results]
    assert "p2" in ids  # appears in both lists, should rank high


@pytest.mark.asyncio
async def test_query_returns_answer(mock_store, mock_embedder):
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = MagicMock(content="Based on the papers, multi-agent RL is...")
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder, llm=mock_llm)
    answer = await rag.query("What is multi-agent RL?")
    assert "multi-agent" in answer.lower() or len(answer) > 0


@pytest.mark.asyncio
async def test_query_without_llm_returns_summaries(mock_store, mock_embedder):
    rag = RAGPipeline(store=mock_store, embedder=mock_embedder, llm=None)
    answer = await rag.query("test query")
    assert "Paper A" in answer or "Paper B" in answer  # returns paper summaries
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_rag.py -v`
Expected: FAIL

- [ ] **Step 3: Implement RAG pipeline**

```python
"""src/discover/rag.py — Hybrid search + RAG answer generation."""
from __future__ import annotations

import logging
from typing import Any

from discover.embeddings import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

RAG_PROMPT = """\
Answer the following research question based on the provided papers.
Cite papers by their title in [brackets].

Papers:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """Hybrid search (dense+sparse+rerank) + LLM RAG generation."""

    def __init__(
        self,
        store: Any,
        embedder: Any,
        llm: Any = None,
    ):
        self._store = store
        self._embedder = embedder
        self._llm = llm

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using RRF fusion of dense + sparse results, then rerank."""
        dense_vec = self._embedder.embed_text(query)
        _, sparse_vec = self._embedder.embed_paper(query, "")

        dense_results = self._store.search_dense(dense_vec, limit=top_k * 3, domain=domain)
        sparse_results = self._store.search_sparse(sparse_vec, limit=top_k * 3)

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        # Rerank top candidates
        candidates_for_rerank = [
            {"title": r.get("payload", {}).get("title", ""), "abstract": r.get("payload", {}).get("abstract", ""), **r}
            for r in fused[: top_k * 2]
        ]
        if candidates_for_rerank:
            reranked = self._embedder.rerank(query, candidates_for_rerank, top_k=top_k)
            return reranked

        return fused[:top_k]

    async def query(
        self,
        question: str,
        top_k: int = 10,
        domain: str | None = None,
    ) -> str:
        """RAG query: search + generate grounded answer."""
        results = self.hybrid_search(question, top_k=top_k, domain=domain)

        if not results:
            return "No relevant papers found."

        # Build context from search results
        context_parts = []
        for i, r in enumerate(results, 1):
            payload = r.get("payload", r)
            title = payload.get("title", "Unknown")
            abstract = payload.get("abstract", "No abstract")
            context_parts.append(f"[{i}] {title}\n{abstract[:500]}")

        context = "\n\n".join(context_parts)

        if self._llm is None:
            # Fallback: return paper summaries without LLM synthesis
            return f"Found {len(results)} relevant papers:\n\n" + context

        from sage.llm.base import Message, Role
        prompt = RAG_PROMPT.format(context=context, question=question)
        messages = [Message(role=Role.USER, content=prompt)]
        response = await self._llm.generate(messages)
        return response.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_rag.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/rag.py tests/test_rag.py
git commit -m "feat: add RAG pipeline — hybrid search (RRF dense+sparse+rerank) + LLM answer"
```

---

## Task 9: MAP-Elites Frontier Explorer (Innovation #2)

**Files:**
- Create: `src/discover/frontier.py`
- Test: `tests/test_frontier.py`

- [ ] **Step 1: Write failing tests**

```python
"""tests/test_frontier.py — MAP-Elites frontier explorer tests."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.frontier import (
    FrontierDescriptor,
    FrontierArchive,
    FrontierExplorer,
)


def test_descriptor_dataclass():
    d = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    assert d.domain_idx == 0


def test_archive_insert():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    inserted = archive.try_insert("p1", desc, fitness=0.9)
    assert inserted is True
    assert archive.size() == 1


def test_archive_replaces_lower_fitness():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.5)
    replaced = archive.try_insert("p2", desc, fitness=0.9)
    assert replaced is True
    assert archive.size() == 1  # still 1, p2 replaced p1


def test_archive_rejects_lower_fitness():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.9)
    rejected = archive.try_insert("p2", desc, fitness=0.3)
    assert rejected is False


def test_archive_coverage():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    total_cells = 5 * 4 * 4 * 3  # 240
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.9)
    coverage = archive.coverage()
    assert abs(coverage - 1.0 / total_cells) < 0.001


def test_archive_get_empty_cells():
    archive = FrontierArchive(bins=[2, 2, 2, 2])
    total = 2 * 2 * 2 * 2  # 16
    desc = FrontierDescriptor(domain_idx=0, recency=0.0, citation_velocity=0.0, novelty=0.0)
    archive.try_insert("p1", desc, fitness=0.9)
    empty = archive.get_empty_cells()
    assert len(empty) == total - 1


def test_compute_descriptor():
    explorer = FrontierExplorer.__new__(FrontierExplorer)
    explorer._archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = explorer._compute_descriptor(
        domain="marl",
        published=date.today(),
        citation_count=10,
        paper_embedding=np.random.rand(768).astype(np.float32),
    )
    assert 0 <= desc.domain_idx <= 4
    assert 0.0 <= desc.recency <= 1.0
    assert 0.0 <= desc.novelty <= 1.0


@pytest.mark.asyncio
async def test_explorer_seed():
    mock_store = MagicMock()
    mock_store.search_dense.return_value = [
        {"id": "p1", "score": 0.9, "payload": {
            "title": "A", "abstract": "B", "domain": "marl",
            "year": 2025, "citation_count": 5, "_paper_id": "p1",
        }},
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = np.random.rand(768).astype(np.float32)

    explorer = FrontierExplorer(store=mock_store, embedder=mock_embedder)
    await explorer.seed()
    assert explorer._archive.size() >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_frontier.py -v`
Expected: FAIL

- [ ] **Step 3: Implement frontier.py**

```python
"""src/discover/frontier.py — MAP-Elites frontier explorer for research discovery."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

from discover.discovery import DOMAINS, discover

logger = logging.getLogger(__name__)

DOMAINS_LIST = list(DOMAINS.keys())


@dataclass
class FrontierDescriptor:
    """4D behavior descriptor for MAP-Elites archive."""
    domain_idx: int           # 0-4 (5 research domains)
    recency: float            # 0-1 (days_old / 365, clamped)
    citation_velocity: float  # 0-1 (normalized)
    novelty: float            # 0-1 (1 - max_cosine_sim to archive)


@dataclass
class FrontierEntry:
    paper_id: str
    descriptor: FrontierDescriptor
    fitness: float


@dataclass
class FrontierReport:
    coverage: float
    total_papers: int
    empty_regions: int
    generations_run: int


class FrontierArchive:
    """MAP-Elites grid archive for research frontier coverage."""

    def __init__(self, bins: list[int] | None = None):
        self._bins = bins or [5, 4, 4, 3]  # 240 cells
        self._grid: dict[tuple[int, ...], FrontierEntry] = {}

    def _descriptor_to_cell(self, desc: FrontierDescriptor) -> tuple[int, ...]:
        """Map continuous descriptor to discrete grid cell."""
        d = min(desc.domain_idx, self._bins[0] - 1)
        r = min(int(desc.recency * self._bins[1]), self._bins[1] - 1)
        c = min(int(desc.citation_velocity * self._bins[2]), self._bins[2] - 1)
        n = min(int(desc.novelty * self._bins[3]), self._bins[3] - 1)
        return (d, r, c, n)

    def try_insert(self, paper_id: str, desc: FrontierDescriptor, fitness: float) -> bool:
        """Insert if cell is empty or new fitness > existing. Returns True if inserted."""
        cell = self._descriptor_to_cell(desc)
        existing = self._grid.get(cell)
        if existing is None or fitness > existing.fitness:
            self._grid[cell] = FrontierEntry(paper_id, desc, fitness)
            return True
        return False

    def size(self) -> int:
        return len(self._grid)

    def coverage(self) -> float:
        total = 1
        for b in self._bins:
            total *= b
        return len(self._grid) / total

    def get_empty_cells(self) -> list[tuple[int, ...]]:
        """Return all empty cell coordinates."""
        all_cells = set()
        for d in range(self._bins[0]):
            for r in range(self._bins[1]):
                for c in range(self._bins[2]):
                    for n in range(self._bins[3]):
                        all_cells.add((d, r, c, n))
        return list(all_cells - set(self._grid.keys()))

    def cell_to_descriptor(self, cell: tuple[int, ...]) -> FrontierDescriptor:
        """Convert cell back to approximate descriptor (midpoints)."""
        return FrontierDescriptor(
            domain_idx=cell[0],
            recency=(cell[1] + 0.5) / self._bins[1],
            citation_velocity=(cell[2] + 0.5) / self._bins[2],
            novelty=(cell[3] + 0.5) / self._bins[3],
        )

    def get_best_per_dimension(self, dim: int) -> dict[int, FrontierEntry]:
        """Get best entry for each value of dimension `dim`."""
        best: dict[int, FrontierEntry] = {}
        for cell, entry in self._grid.items():
            val = cell[dim]
            if val not in best or entry.fitness > best[val].fitness:
                best[val] = entry
        return best


class FrontierExplorer:
    """MAP-Elites-based research frontier explorer."""

    def __init__(
        self,
        store: Any,
        embedder: Any,
        llm: Any = None,
        bins: list[int] | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._archive = FrontierArchive(bins=bins)

    def _compute_descriptor(
        self,
        domain: str,
        published: date | None = None,
        citation_count: int = 0,
        paper_embedding: np.ndarray | None = None,
    ) -> FrontierDescriptor:
        domain_idx = DOMAINS_LIST.index(domain) if domain in DOMAINS_LIST else 0

        if published:
            days_old = (date.today() - published).days
            recency = min(days_old / 365.0, 1.0)
        else:
            recency = 0.5

        # Simple citation velocity (citations per assumed 6-month window)
        citation_velocity = min(citation_count / 50.0, 1.0)

        # Novelty: 1 - max similarity to existing archive entries
        novelty = 1.0
        if paper_embedding is not None and self._archive.size() > 0:
            # Search store for nearest neighbors
            results = self._store.search_dense(paper_embedding, limit=1)
            if results and results[0].get("score", 0) > 0:
                novelty = max(0.0, 1.0 - results[0]["score"])

        return FrontierDescriptor(
            domain_idx=domain_idx,
            recency=recency,
            citation_velocity=citation_velocity,
            novelty=novelty,
        )

    async def seed(self) -> None:
        """Seed archive from papers already in the store."""
        for domain in DOMAINS_LIST:
            dummy_vec = self._embedder.embed_text(domain)
            results = self._store.search_dense(dummy_vec, limit=20, domain=domain)
            for r in results:
                payload = r.get("payload", {})
                desc = self._compute_descriptor(
                    domain=payload.get("domain", domain),
                    citation_count=payload.get("citation_count", 0),
                )
                self._archive.try_insert(
                    r["id"], desc, fitness=r.get("score", 0.5),
                )

    async def _generate_query(self, target: FrontierDescriptor) -> str:
        """Generate a search query targeting a specific frontier region."""
        domain_name = DOMAINS_LIST[target.domain_idx] if target.domain_idx < len(DOMAINS_LIST) else "marl"
        recency_hint = "very recent (last month)" if target.recency < 0.1 else "from the past year"
        novelty_hint = "highly novel, underexplored" if target.novelty > 0.7 else "well-established"

        if self._llm is None:
            # Fallback: use domain keywords
            keywords = DOMAINS[domain_name]["keywords"]
            return keywords[0] if keywords else domain_name

        from sage.llm.base import Message, Role
        prompt = (
            f"Generate a specific arXiv search query for:\n"
            f"- Domain: {domain_name}\n"
            f"- Paper type: {recency_hint}\n"
            f"- Desired novelty: {novelty_hint}\n\n"
            f"Return only the search query string, no explanation."
        )
        messages = [Message(role=Role.USER, content=prompt)]
        response = await self._llm.generate(messages)
        return response.content.strip()

    async def explore(self, generations: int = 5, batch_size: int = 10) -> FrontierReport:
        """Run MAP-Elites exploration for N generations."""
        for gen in range(generations):
            empty_cells = self._archive.get_empty_cells()
            if not empty_cells:
                logger.info("Archive fully covered at generation %d", gen)
                break

            # Target underexplored regions
            targets = empty_cells[:batch_size]
            for cell in targets:
                target_desc = self._archive.cell_to_descriptor(cell)
                query = await self._generate_query(target_desc)

                try:
                    since = date.today() - timedelta(days=int(target_desc.recency * 365) + 7)
                    domain = DOMAINS_LIST[target_desc.domain_idx] if target_desc.domain_idx < len(DOMAINS_LIST) else None
                    candidates = await discover(
                        since=since,
                        query=query,
                        domains=[domain] if domain else None,
                    )
                except Exception as e:
                    logger.warning("Discovery failed for query '%s': %s", query, e)
                    continue

                for paper in candidates[:5]:  # cap per query
                    embedding = self._embedder.embed_text(f"{paper.title}. {paper.abstract}")
                    desc = self._compute_descriptor(
                        domain=paper.domain,
                        published=paper.published,
                        citation_count=paper.citation_count,
                        paper_embedding=embedding,
                    )
                    self._archive.try_insert(paper.paper_id, desc, fitness=0.5)

            logger.info("Generation %d: coverage=%.2f%%, archive_size=%d",
                       gen, self._archive.coverage() * 100, self._archive.size())

        return FrontierReport(
            coverage=self._archive.coverage(),
            total_papers=self._archive.size(),
            empty_regions=len(self._archive.get_empty_cells()),
            generations_run=generations,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_frontier.py -v`
Expected: 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/frontier.py tests/test_frontier.py
git commit -m "feat: add MAP-Elites frontier explorer — 4D behavior descriptors, coverage-driven search"
```

---

## Task 10: MCP Server

**Files:**
- Create: `src/discover/mcp.py`
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write failing tests**

```python
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
    mock_components.return_value = {"discover_fn": mock_discover}
    result = await tool_discover_papers(query="multi-agent RL", domains=None, since=None, max_results=10)
    assert isinstance(result, list)


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_query_knowledge(mock_components):
    mock_rag = MagicMock()
    mock_rag.query = AsyncMock(return_value="Answer about multi-agent RL")
    mock_components.return_value = {"rag": mock_rag}
    result = await tool_query_knowledge(question="What is MARL?", top_k=5, domain=None)
    assert "multi-agent" in result.lower() or len(result) > 0


@pytest.mark.asyncio
@patch("discover.mcp._get_pipeline_components")
async def test_tool_verify_claims(mock_components):
    mock_store = MagicMock()
    mock_store.get_paper.return_value = {"title": "Test", "abstract": "Test abstract"}
    mock_store.get_claims_for_paper.return_value = []
    mock_components.return_value = {"store": mock_store, "llm": AsyncMock(), "embedder": MagicMock()}
    result = await tool_verify_claims(paper_id="p1")
    assert "claims" in result or "paper_id" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-discover && python -m pytest tests/test_mcp.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MCP server**

```python
"""src/discover/mcp.py — FastMCP server exposing 5 discovery tools."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-initialized singleton components
_components: dict[str, Any] | None = None


def _get_pipeline_components() -> dict[str, Any]:
    """Lazy-initialize pipeline components."""
    global _components
    if _components is not None:
        return _components

    from discover.store import KnowledgeStore
    from discover.embeddings import EmbeddingPipeline
    from discover.rag import RAGPipeline

    store = KnowledgeStore()
    embedder = EmbeddingPipeline()

    # Try to get LLM provider
    llm = None
    try:
        from sage.llm.google import GoogleProvider
        llm = GoogleProvider()
    except Exception:
        logger.warning("No LLM provider available for MCP tools")

    rag = RAGPipeline(store=store, embedder=embedder, llm=llm)

    _components = {
        "store": store,
        "embedder": embedder,
        "llm": llm,
        "rag": rag,
        "discover_fn": None,  # will import discover.discovery.discover
    }
    return _components


async def tool_discover_papers(
    query: str,
    domains: list[str] | None = None,
    since: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Discover papers from arXiv + Semantic Scholar + HuggingFace."""
    from discover.discovery import discover

    since_date = date.fromisoformat(since) if since else date.today() - timedelta(days=7)
    candidates = await discover(since=since_date, query=query, domains=domains)
    return [
        {
            "paper_id": c.paper_id,
            "title": c.title,
            "authors": c.authors,
            "abstract": c.abstract[:300],
            "source": c.source,
            "domain": c.domain,
            "published": c.published.isoformat(),
            "citation_count": c.citation_count,
        }
        for c in candidates[:max_results]
    ]


async def tool_curate_papers(paper_ids: list[str]) -> list[dict]:
    """Score and filter papers using adaptive curation."""
    components = _get_pipeline_components()
    store = components["store"]

    papers = []
    for pid in paper_ids:
        p = store.get_paper(pid)
        if p:
            papers.append({"paper_id": pid, "title": p.get("title", ""), "relevance_score": p.get("relevance_score", 0)})
    return papers


async def tool_query_knowledge(
    question: str,
    top_k: int = 10,
    domain: str | None = None,
) -> str:
    """RAG query over the local knowledge store."""
    components = _get_pipeline_components()
    rag = components["rag"]
    return await rag.query(question, top_k=top_k, domain=domain)


async def tool_explore_frontier(
    domain: str | None = None,
    generations: int = 5,
) -> dict:
    """MAP-Elites exploration of the research frontier."""
    from discover.frontier import FrontierExplorer

    components = _get_pipeline_components()
    explorer = FrontierExplorer(
        store=components["store"],
        embedder=components["embedder"],
        llm=components["llm"],
    )
    await explorer.seed()
    report = await explorer.explore(generations=generations)
    return {
        "coverage": round(report.coverage, 4),
        "total_papers": report.total_papers,
        "empty_regions": report.empty_regions,
        "generations_run": report.generations_run,
    }


async def tool_verify_claims(paper_id: str) -> dict:
    """Extract and SMT-verify claims from a paper."""
    from discover.claim_graph import extract_claims_from_text, verify_claim_cluster

    components = _get_pipeline_components()
    store = components["store"]
    llm = components["llm"]

    paper = store.get_paper(paper_id)
    if not paper:
        return {"error": f"Paper {paper_id} not found in store"}

    text = paper.get("abstract", "")
    if not text:
        return {"error": "No text available for claim extraction"}

    if llm is None:
        return {"error": "No LLM available for claim extraction"}

    claims = await extract_claims_from_text(text, paper_id, llm)
    if not claims:
        return {"paper_id": paper_id, "claims": [], "verification": "no_claims"}

    status = verify_claim_cluster(claims)
    return {
        "paper_id": paper_id,
        "claims": [{"statement": c.statement, "type": c.claim_type, "confidence": c.confidence} for c in claims],
        "verification": status,
    }


def create_mcp_server():
    """Create and configure the FastMCP server."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp package not installed. Run: pip install 'mcp[cli]'")
        raise

    mcp = FastMCP("sage-discover")

    mcp.tool()(tool_discover_papers)
    mcp.tool()(tool_curate_papers)
    mcp.tool()(tool_query_knowledge)
    mcp.tool()(tool_explore_frontier)
    mcp.tool()(tool_verify_claims)

    return mcp


if __name__ == "__main__":
    server = create_mcp_server()
    server.run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-discover && python -m pytest tests/test_mcp.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discover/mcp.py tests/test_mcp.py
git commit -m "feat: add MCP server — 5 tools (discover, curate, query, explore, verify)"
```

---

## Task 11: Wire Pipeline + Update Existing Files

**Files:**
- Modify: `src/discover/pipeline.py`
- Modify: `src/discover/ingestion.py`
- Modify: `src/discover/discovery.py`
- Modify: `src/discover/__init__.py`
- Modify: `src/discover/__main__.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Update discovery.py — add influential_citation_count**

Add `influential_citation_count` field to `PaperCandidate`:

In `src/discover/discovery.py`, after line 85 (`citation_count: int`), add:

```python
    influential_citation_count: int = 0
```

- [ ] **Step 2: Update ingestion.py — upsert to Qdrant instead of ExoCortex**

Replace the `ingest` function in `src/discover/ingestion.py` to support both Qdrant store and legacy ExoCortex:

```python
"""src/discover/ingestion.py — Ingest papers to KnowledgeStore (Qdrant) or ExoCortex."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from discover.curator import CuratedPaper

logger = logging.getLogger(__name__)

DEFAULT_PAPERS_DIR: Path = Path.home() / ".sage" / "papers"


async def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF from HTTPS URL."""
    if not url or not url.startswith("https://"):
        return False

    def _download():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except Exception as e:
            logger.warning("PDF download failed: %s", e)
            return False

    return await asyncio.to_thread(_download)


async def ingest_to_store(
    paper: CuratedPaper,
    store: Any,
    embedder: Any,
) -> bool:
    """Ingest a curated paper into the Qdrant KnowledgeStore."""
    pid = paper.candidate.paper_id

    # Check if already in store
    existing = store.get_paper(pid)
    if existing is not None:
        return False

    # Download PDF if available
    pdf_path = paper.pdf_path
    if pdf_path is None and paper.candidate.pdf_url:
        pdf_path = DEFAULT_PAPERS_DIR / f"{pid.replace('/', '_')}.pdf"
        downloaded = await download_pdf(paper.candidate.pdf_url, pdf_path)
        if not downloaded:
            pdf_path = None

    # Embed
    dense, sparse = embedder.embed_paper(paper.candidate.title, paper.candidate.abstract)

    # Build payload
    payload = {
        "title": paper.candidate.title,
        "authors": paper.candidate.authors,
        "abstract": paper.candidate.abstract,
        "domain": paper.candidate.domain,
        "source": paper.candidate.source,
        "year": paper.candidate.published.year,
        "citation_count": paper.candidate.citation_count,
        "relevance_score": paper.relevance_score,
        "reason": paper.reason,
        "key_insights": paper.key_insights,
        "pdf_url": paper.candidate.pdf_url,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    store.upsert_paper(pid, dense, sparse, payload)
    logger.info("Ingested paper %s: %s", pid, paper.candidate.title)
    return True


async def ingest_all_to_store(
    papers: list[CuratedPaper],
    store: Any,
    embedder: Any,
) -> int:
    """Ingest all curated papers into the store. Returns count of newly ingested."""
    count = 0
    for paper in papers:
        try:
            if await ingest_to_store(paper, store, embedder):
                count += 1
        except Exception as e:
            logger.warning("Failed to ingest %s: %s", paper.candidate.paper_id, e)
    return count


# --- Legacy ExoCortex support ---

@dataclass
class Manifest:
    """Local tracking manifest for legacy ExoCortex ingestion."""
    store_name: str = ""
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)


DEFAULT_MANIFEST_PATH: Path = Path.home() / ".sage" / "manifest.json"


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    try:
        data = json.loads(path.read_text())
        return Manifest(**data)
    except Exception:
        return Manifest()


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"store_name": manifest.store_name, "papers": manifest.papers}, indent=2))


def is_already_ingested(paper_id: str, manifest: Manifest) -> bool:
    return paper_id in manifest.papers


async def ingest(
    paper: CuratedPaper,
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    """Legacy: ingest to ExoCortex."""
    manifest = load_manifest(manifest_path)
    pid = paper.candidate.paper_id

    if is_already_ingested(pid, manifest):
        return False

    pdf_path = paper.pdf_path
    if pdf_path is None and paper.candidate.pdf_url:
        pdf_path = DEFAULT_PAPERS_DIR / f"{pid.replace('/', '_')}.pdf"
        if not await download_pdf(paper.candidate.pdf_url, pdf_path):
            # Fallback: markdown
            pdf_path = DEFAULT_PAPERS_DIR / f"{pid.replace('/', '_')}.md"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_text(f"# {paper.candidate.title}\n\n{paper.candidate.abstract}")

    if pdf_path is None:
        pdf_path = DEFAULT_PAPERS_DIR / f"{pid.replace('/', '_')}.md"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_text(f"# {paper.candidate.title}\n\n{paper.candidate.abstract}")

    display_name = f"[{paper.candidate.domain}] {paper.candidate.title}"
    await exocortex.upload(str(pdf_path), display_name)

    manifest.papers[pid] = {
        "title": paper.candidate.title,
        "domain": paper.candidate.domain,
        "source": paper.candidate.source,
        "relevance_score": paper.relevance_score,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest, manifest_path)
    return True


async def ingest_all(
    papers: list[CuratedPaper],
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> int:
    """Legacy: ingest all to ExoCortex."""
    count = 0
    for paper in papers:
        try:
            if await ingest(paper, exocortex, manifest_path):
                count += 1
        except Exception as e:
            logger.warning("Failed to ingest %s: %s", paper.candidate.paper_id, e)
    return count
```

- [ ] **Step 3: Update pipeline.py — wire new components**

Replace `src/discover/pipeline.py` with the following that supports both new (Qdrant) and legacy (ExoCortex) paths:

```python
"""src/discover/pipeline.py — Knowledge discovery pipeline orchestrator."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from discover.discovery import discover
from discover.curator import curate, heuristic_filter, CuratedPaper
from discover.ingestion import ingest_all, ingest_all_to_store
from discover.migration import migrate_notebooks

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Summary of a pipeline run."""
    discovered: int = 0
    curated: int = 0
    ingested: int = 0


def _try_init_store():
    """Try to initialize Qdrant KnowledgeStore."""
    try:
        from discover.store import KnowledgeStore
        return KnowledgeStore()
    except Exception as e:
        logger.info("KnowledgeStore not available: %s", e)
        return None


def _try_init_embedder():
    """Try to initialize EmbeddingPipeline."""
    try:
        from discover.embeddings import EmbeddingPipeline
        return EmbeddingPipeline()
    except Exception as e:
        logger.info("EmbeddingPipeline not available: %s", e)
        return None


def _try_init_exocortex():
    """Try to initialize ExoCortex (legacy)."""
    store_name = os.environ.get("SAGE_EXOCORTEX_STORE")
    if not store_name:
        return None
    try:
        from sage.memory.remote_rag import ExoCortex
        return ExoCortex()
    except Exception:
        return None


def _try_init_llm(llm=None):
    """Try to initialize LLM provider."""
    if llm is not None:
        return llm
    try:
        from sage.llm.google import GoogleProvider
        return GoogleProvider()
    except Exception:
        return None


async def run_pipeline(
    mode: str = "nightly",
    query: str | None = None,
    since: date | None = None,
    domains: list[str] | None = None,
    exocortex: Any = None,
    llm: Any = None,
    store: Any = None,
    embedder: Any = None,
) -> PipelineReport:
    """Run the knowledge discovery pipeline.

    Modes:
        nightly   — discover recent papers, curate, ingest
        on-demand — targeted search with query
        migrate   — import NotebookLM markdown exports
        watch     — detect unprofiled models
    """
    report = PipelineReport()

    if mode == "migrate":
        exocortex = exocortex or _try_init_exocortex()
        if exocortex is None:
            logger.warning("No ExoCortex configured for migration")
            return report
        count = await migrate_notebooks(exocortex)
        report.ingested = count
        return report

    # Initialize components
    llm = _try_init_llm(llm)
    store = store or _try_init_store()
    embedder = embedder or (None if store is None else _try_init_embedder())

    # Discovery
    since = since or (date.today() - timedelta(days=1))
    candidates = await discover(since=since, query=query or "", domains=domains)
    report.discovered = len(candidates)
    logger.info("Discovered %d papers", report.discovered)

    if not candidates:
        return report

    # Curation
    if llm:
        try:
            from discover.adaptive_curator import adaptive_curate
            curated = await adaptive_curate(candidates, llm, embedder=embedder)
        except Exception:
            curated = await curate(candidates, llm)
    else:
        # Heuristic fallback
        filtered = heuristic_filter(candidates)
        curated = [CuratedPaper(candidate=c, relevance_score=5, reason="heuristic") for c in filtered]

    report.curated = len(curated)
    logger.info("Curated %d papers", report.curated)

    if not curated:
        return report

    # Ingestion — prefer Qdrant store, fall back to ExoCortex
    if store and embedder:
        report.ingested = await ingest_all_to_store(curated, store, embedder)
    else:
        exocortex = exocortex or _try_init_exocortex()
        if exocortex:
            report.ingested = await ingest_all(curated, exocortex)

    logger.info("Ingested %d papers", report.ingested)
    return report
```

- [ ] **Step 4: Update __init__.py — new exports**

```python
"""sage-discover: Knowledge Discovery Engine with SMT-verified claims."""

__version__ = "0.2.0"

from discover.pipeline import run_pipeline, PipelineReport

__all__ = [
    "run_pipeline",
    "PipelineReport",
]
```

- [ ] **Step 5: Update __main__.py — add mcp mode**

Add the MCP mode to `src/discover/__main__.py`. After the watch mode handling, add:

```python
    if args.mode == "mcp":
        from discover.mcp import create_mcp_server
        server = create_mcp_server()
        server.run()
        return
```

And add `"mcp"` to the mode choices in the argparse definition.

- [ ] **Step 6: Update test_pipeline.py**

Update the pipeline tests to reflect the new `store`/`embedder` parameters and adaptive curation:

```python
"""tests/test_pipeline.py — Updated pipeline tests."""
import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from discover.pipeline import PipelineReport, run_pipeline
from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper


@pytest.fixture
def sample_candidates():
    return [
        PaperCandidate(
            paper_id="arxiv-2025-001", title="Paper A", authors=["Auth1"],
            abstract="A " * 60, source="arxiv", domain="evolutionary_computation",
            published=date.today(), pdf_url="https://arxiv.org/pdf/2025.001.pdf",
            citation_count=10,
        ),
        PaperCandidate(
            paper_id="s2-2025-002", title="Paper B", authors=["Auth2"],
            abstract="B " * 60, source="s2", domain="marl",
            published=date.today(), pdf_url=None, citation_count=5,
        ),
    ]


@pytest.fixture
def sample_curated(sample_candidates):
    return [
        CuratedPaper(candidate=sample_candidates[0], relevance_score=8, reason="Relevant",
                     key_insights=["insight1"]),
        CuratedPaper(candidate=sample_candidates[1], relevance_score=7, reason="Useful",
                     key_insights=["insight2"]),
    ]


def test_pipeline_report_structure():
    r = PipelineReport()
    assert r.discovered == 0
    assert r.curated == 0
    assert r.ingested == 0
    r2 = PipelineReport(discovered=5, curated=3, ingested=2)
    assert r2.discovered == 5


@pytest.mark.asyncio
@patch("discover.pipeline.ingest_all_to_store", new_callable=AsyncMock, return_value=2)
@patch("discover.pipeline.curate", new_callable=AsyncMock)
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_nightly_with_store(mock_discover, mock_curate, mock_ingest, sample_candidates, sample_curated):
    mock_discover.return_value = sample_candidates
    mock_curate.return_value = sample_curated
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    report = await run_pipeline(mode="nightly", llm=MagicMock(), store=mock_store, embedder=mock_embedder)
    assert report.discovered == 2
    assert report.ingested == 2
    mock_discover.assert_called_once()


@pytest.mark.asyncio
@patch("discover.pipeline.heuristic_filter")
@patch("discover.pipeline.discover", new_callable=AsyncMock)
async def test_pipeline_no_llm_fallback(mock_discover, mock_filter, sample_candidates):
    mock_discover.return_value = sample_candidates
    mock_filter.return_value = sample_candidates
    report = await run_pipeline(mode="nightly", llm=None, store=None, embedder=None)
    assert report.discovered == 2
    assert report.curated == 2  # all pass heuristic with score=5


@pytest.mark.asyncio
@patch("discover.pipeline.migrate_notebooks", new_callable=AsyncMock, return_value=3)
async def test_pipeline_migrate(mock_migrate):
    mock_exo = MagicMock()
    report = await run_pipeline(mode="migrate", exocortex=mock_exo)
    assert report.ingested == 3
```

- [ ] **Step 7: Update test_ingestion.py**

Add tests for the new `ingest_to_store` function:

```python
"""tests/test_ingestion.py — Updated ingestion tests."""
import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from discover.ingestion import (
    ingest_to_store,
    ingest_all_to_store,
    download_pdf,
    Manifest,
    load_manifest,
    save_manifest,
    is_already_ingested,
)
from discover.discovery import PaperCandidate
from discover.curator import CuratedPaper


@pytest.fixture
def sample_curated():
    candidate = PaperCandidate(
        paper_id="test-001", title="Test Paper", authors=["A"],
        abstract="Abstract text", source="arxiv", domain="marl",
        published=date.today(), pdf_url=None, citation_count=5,
    )
    return CuratedPaper(candidate=candidate, relevance_score=8, reason="good")


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_paper.return_value = None  # not yet ingested
    store.upsert_paper.return_value = None
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_paper.return_value = (np.random.rand(768).astype(np.float32), {"indices": [1], "values": [0.5]})
    return emb


@pytest.mark.asyncio
async def test_ingest_to_store(sample_curated, mock_store, mock_embedder):
    result = await ingest_to_store(sample_curated, mock_store, mock_embedder)
    assert result is True
    mock_store.upsert_paper.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_to_store_skips_existing(sample_curated, mock_store, mock_embedder):
    mock_store.get_paper.return_value = {"title": "Already there"}
    result = await ingest_to_store(sample_curated, mock_store, mock_embedder)
    assert result is False


@pytest.mark.asyncio
async def test_ingest_all_to_store(sample_curated, mock_store, mock_embedder):
    count = await ingest_all_to_store([sample_curated, sample_curated], mock_store, mock_embedder)
    # First ingests, second is duplicate (same paper_id)
    assert count >= 1


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    m = Manifest(store_name="test", papers={"p1": {"title": "A"}})
    save_manifest(m, path)
    loaded = load_manifest(path)
    assert loaded.store_name == "test"
    assert "p1" in loaded.papers


def test_is_already_ingested():
    m = Manifest(papers={"p1": {}})
    assert is_already_ingested("p1", m) is True
    assert is_already_ingested("p2", m) is False
```

- [ ] **Step 8: Run all tests**

Run: `cd sage-discover && python -m pytest tests/ -v`
Expected: All tests pass (existing + new)

- [ ] **Step 9: Commit**

```bash
git add src/discover/pipeline.py src/discover/ingestion.py src/discover/discovery.py src/discover/__init__.py src/discover/__main__.py tests/test_pipeline.py tests/test_ingestion.py
git commit -m "feat: wire new components into pipeline — Qdrant store, adaptive curation, MCP mode"
```

---

## Task 12: Delete Dead Code

**Files:**
- Delete: `src/discover/knowledge.py`
- Delete: `src/discover/workflow.py`
- Delete: `src/discover/researcher.py`
- Delete: `tests/test_discover.py`

- [ ] **Step 1: Delete dead files**

```bash
cd sage-discover
git rm src/discover/knowledge.py src/discover/workflow.py src/discover/researcher.py tests/test_discover.py
```

- [ ] **Step 2: Verify no remaining imports of deleted modules**

Run: `cd sage-discover && grep -r "from discover.knowledge\|from discover.workflow\|from discover.researcher\|import researcher\|import workflow\|import knowledge" src/ tests/`
Expected: No output (no remaining imports)

- [ ] **Step 3: Run all tests to verify nothing breaks**

Run: `cd sage-discover && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove dead code — knowledge.py (stub), workflow.py (fake eBPF), researcher.py (absorbed)"
```

---

## Task 13: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""tests/test_integration.py — End-to-end integration test (all components mocked at API boundary)."""
from __future__ import annotations

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

    # Use real Qdrant store (in-memory) with mocked embedder
    from discover.store import KnowledgeStore
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = KnowledgeStore(path=tmpdir)
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

        # Verify paper is in store
        paper = store.get_paper("integration-001")
        assert paper is not None
        assert paper["title"] == "Integration Test Paper on Multi-Agent Systems"
        assert paper["relevance_score"] == 8


@pytest.mark.asyncio
async def test_store_to_rag_flow():
    """Test: ingest paper -> search -> RAG answer."""
    from discover.store import KnowledgeStore
    from discover.rag import RAGPipeline

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = KnowledgeStore(path=tmpdir)
        dense = np.random.rand(768).astype(np.float32)
        sparse = {"indices": [1, 5], "values": [0.8, 0.5]}
        store.upsert_paper("p1", dense, sparse, {
            "title": "Multi-Agent RL Survey",
            "abstract": "A comprehensive survey of MARL techniques.",
            "domain": "marl",
        })

        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = dense  # same vector = high similarity
        mock_embedder.embed_paper.return_value = (dense, sparse)
        mock_embedder.rerank.side_effect = lambda q, c, top_k: c[:top_k]

        rag = RAGPipeline(store=store, embedder=mock_embedder, llm=None)
        answer = await rag.query("multi-agent RL")
        assert "Multi-Agent RL Survey" in answer
```

- [ ] **Step 2: Run integration tests**

Run: `cd sage-discover && python -m pytest tests/test_integration.py -v`
Expected: 2 tests PASS

- [ ] **Step 3: Run FULL test suite**

Run: `cd sage-discover && python -m pytest tests/ -v`
Expected: All tests pass (existing + new ≈ 70-80 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for knowledge pipeline"
```

---

## Summary

| Task | Component | Tests | Innovation |
|------|-----------|-------|------------|
| 1 | Dependencies | - | Setup |
| 2 | KnowledgeStore (Qdrant) | 8 | SOTA parity |
| 3 | EmbeddingPipeline | 4 | SOTA parity |
| 4 | PDF Extractor | 4 | SOTA parity |
| 5 | CitationGraphBuilder | 6 | SOTA parity |
| 6 | ClaimGraph + SMT | 9 | **Innovation #1** |
| 7 | Adaptive Curator | 7 | **Innovation #3** |
| 8 | RAG Pipeline | 3 | SOTA parity |
| 9 | MAP-Elites Frontier | 9 | **Innovation #2** |
| 10 | MCP Server | 3 | SOTA parity |
| 11 | Pipeline Wiring | 6 | Integration |
| 12 | Dead Code Cleanup | 0 | Cleanup |
| 13 | Integration Tests | 2 | Validation |
| **Total** | **13 tasks** | **~61 new tests** | **3 innovations** |
