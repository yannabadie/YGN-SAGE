"""src/discover/embeddings.py — SPECTER2 + SPLADE + cross-encoder reranker."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports — set to actual classes on first use
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
        """Embed a paper. Returns (dense_768d, {"indices": [...], "values": [...]})."""
        text = f"{title}. {abstract}"
        dense = self._dense.encode(text)
        if not isinstance(dense, np.ndarray):
            dense = np.array(dense, dtype=np.float32)

        sparse_dict = {"indices": [], "values": []}
        if self._sparse is not None:
            try:
                sparse_out = self._sparse.encode(text)
                import torch
                if isinstance(sparse_out, torch.Tensor):
                    # Force CPU + dense — SPLADE on CUDA returns sparse tensor
                    t = sparse_out.cpu()
                    if t.is_sparse:
                        t = t.to_dense()
                    if t.dim() == 1:
                        nz_indices = t.nonzero(as_tuple=True)[0]
                        vals_tensor = t[nz_indices]
                    else:
                        nz_indices = t[0].nonzero(as_tuple=True)[0]
                        vals_tensor = t[0][nz_indices]
                    indices = nz_indices.tolist()
                    values = vals_tensor.tolist()
                    sparse_dict = {"indices": indices, "values": values}
                else:
                    nz = sparse_out.nonzero()
                    if len(nz) == 2:
                        indices = nz[1].tolist() if hasattr(nz[1], 'tolist') else list(nz[1])
                        values = [float(sparse_out[0, i]) for i in indices] if indices else []
                        sparse_dict = {"indices": indices, "values": values}
            except Exception as exc:
                logger.warning("SPLADE encoding failed: %s", exc)

        return dense, sparse_dict

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text (queries, claims)."""
        dense = self._dense.encode(text)
        if not isinstance(dense, np.ndarray):
            dense = np.array(dense, dtype=np.float32)
        return dense

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
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
