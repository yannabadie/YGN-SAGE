"""src/discover/store.py — Qdrant-backed knowledge store."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


def _str_to_int_id(s: str) -> int:
    """Deterministic int ID from string paper_id."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


class KnowledgeStore:
    """Local Qdrant wrapper for papers and claims."""

    DENSE_DIM = 768

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
