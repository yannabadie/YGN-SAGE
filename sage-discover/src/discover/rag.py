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

    def __init__(self, store: Any, embedder: Any, llm: Any = None):
        self._store = store
        self._embedder = embedder
        self._llm = llm

    def hybrid_search(self, query: str, top_k: int = 10, domain: str | None = None) -> list[dict[str, Any]]:
        dense_vec = self._embedder.embed_text(query)
        _, sparse_vec = self._embedder.embed_paper(query, "")

        dense_results = self._store.search_dense(dense_vec, limit=top_k * 3, domain=domain)
        sparse_results = self._store.search_sparse(sparse_vec, limit=top_k * 3)

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)

        candidates_for_rerank = [
            {"title": r.get("payload", {}).get("title", ""), "abstract": r.get("payload", {}).get("abstract", ""), **r}
            for r in fused[:top_k * 2]
        ]
        if candidates_for_rerank:
            reranked = self._embedder.rerank(query, candidates_for_rerank, top_k=top_k)
            return reranked

        return fused[:top_k]

    async def query(self, question: str, top_k: int = 10, domain: str | None = None) -> str:
        results = self.hybrid_search(question, top_k=top_k, domain=domain)

        if not results:
            return "No relevant papers found."

        context_parts = []
        for i, r in enumerate(results, 1):
            payload = r.get("payload", r)
            title = payload.get("title", "Unknown")
            abstract = payload.get("abstract", "No abstract")
            context_parts.append(f"[{i}] {title}\n{abstract[:500]}")

        context = "\n\n".join(context_parts)

        if self._llm is None:
            return f"Found {len(results)} relevant papers:\n\n" + context

        from sage.llm.base import Message, Role
        prompt = RAG_PROMPT.format(context=context, question=question)
        messages = [Message(role=Role.USER, content=prompt)]
        response = await self._llm.generate(messages)
        return response.content
