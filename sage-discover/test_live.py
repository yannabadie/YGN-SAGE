"""Live integration test — real API calls to arXiv + Semantic Scholar + Qdrant + SPECTER2.

Usage: cd sage-discover && python test_live.py
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("live_test")


async def main():
    t0 = time.time()

    # -------------------------------------------------------------------
    # 1. Discovery — real API calls
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Discovery (arXiv + Semantic Scholar + HuggingFace)")
    logger.info("=" * 60)

    from discover.discovery import discover, DOMAINS

    since = date.today() - timedelta(days=14)
    candidates = await discover(since=since, query="multi-agent reinforcement learning", domains=["marl"])

    logger.info("Discovered %d papers in %.1fs", len(candidates), time.time() - t0)
    for c in candidates[:5]:
        logger.info("  [%s] %s (citations=%d, source=%s)", c.domain, c.title[:80], c.citation_count, c.source)

    if not candidates:
        logger.warning("No candidates found — check network/API keys. Exiting.")
        return

    # -------------------------------------------------------------------
    # 2. Embedding — real SPECTER2
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: Embedding (SPECTER2)")
    logger.info("=" * 60)

    t1 = time.time()
    from discover.embeddings import EmbeddingPipeline

    embedder = EmbeddingPipeline()
    logger.info("EmbeddingPipeline loaded in %.1fs", time.time() - t1)

    # Embed first 3 papers
    embeddings = []
    for c in candidates[:3]:
        t2 = time.time()
        dense, sparse = embedder.embed_paper(c.title, c.abstract)
        embeddings.append((c, dense, sparse))
        logger.info("  Embedded '%s' in %.2fs — dense=%s, sparse_nnz=%d",
                     c.title[:60], time.time() - t2, dense.shape, len(sparse["indices"]))

    # -------------------------------------------------------------------
    # 3. Store — real Qdrant (local, tmpdir)
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 3: KnowledgeStore (Qdrant local)")
    logger.info("=" * 60)

    from discover.store import KnowledgeStore

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = KnowledgeStore(path=tmpdir)
        logger.info("Qdrant store initialized at %s", tmpdir)

        for c, dense, sparse in embeddings:
            store.upsert_paper(
                paper_id=c.paper_id,
                dense_vector=dense,
                sparse_vector=sparse,
                payload={
                    "title": c.title,
                    "authors": c.authors,
                    "abstract": c.abstract,
                    "domain": c.domain,
                    "source": c.source,
                    "year": c.published.year,
                    "citation_count": c.citation_count,
                },
            )

        logger.info("Stored %d papers", store.paper_count())

        # -------------------------------------------------------------------
        # 4. Hybrid Search — real vectors
        # -------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("PHASE 4: Hybrid Search")
        logger.info("=" * 60)

        query = "cooperative multi-agent systems"
        query_dense = embedder.embed_text(query)
        results = store.search_dense(query_dense, limit=3)
        logger.info("Search '%s' returned %d results:", query, len(results))
        for r in results:
            logger.info("  score=%.3f title='%s'", r["score"], r["payload"].get("title", "?")[:80])

        # -------------------------------------------------------------------
        # 5. RAG — real search, no LLM (fallback mode)
        # -------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("PHASE 5: RAG Query (no-LLM fallback)")
        logger.info("=" * 60)

        from discover.rag import RAGPipeline

        rag = RAGPipeline(store=store, embedder=embedder, llm=None)
        answer = await rag.query("What are the latest advances in multi-agent RL?")
        logger.info("RAG answer (%d chars):\n%s", len(answer), answer[:500])

        # -------------------------------------------------------------------
        # 6. ClaimGraph — extract claims from first paper
        # -------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("PHASE 6: Claim Extraction + SMT (first paper)")
        logger.info("=" * 60)

        from discover.claim_graph import translate_claim_to_smt, Claim, verify_claim_cluster

        first = candidates[0]
        # Manually create claims from the abstract for SMT testing
        test_claims = [
            Claim("test_c0", f"Method achieves 95% accuracy on benchmark X", first.paper_id, "finding", 0.9),
            Claim("test_c1", f"Method achieves 40% accuracy on benchmark X", first.paper_id, "finding", 0.9),
        ]

        for tc in test_claims:
            formula = translate_claim_to_smt(tc)
            logger.info("  Claim: '%s' -> SMT: %s", tc.statement[:60], formula)

        status = verify_claim_cluster(test_claims)
        logger.info("  SMT verification: %s (expected: contradictory or unknown)", status)

        # -------------------------------------------------------------------
        # 7. Citation Graph — from discovered papers
        # -------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("PHASE 7: Citation Graph (NetworkX)")
        logger.info("=" * 60)

        from discover.citation_graph import CitationGraphBuilder

        graph = CitationGraphBuilder()
        for c in candidates[:10]:
            graph.add_paper(c.paper_id, title=c.title, year=c.published.year)

        # Add synthetic citation edges between consecutive papers
        paper_ids = [c.paper_id for c in candidates[:10]]
        for i in range(len(paper_ids) - 1):
            graph.add_citation(paper_ids[i], paper_ids[i + 1])

        ranks = graph.pagerank()
        top_ranked = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info("  Graph: %d nodes, %d edges", graph.node_count(), graph.edge_count())
        for pid, rank in top_ranked:
            title = graph.graph.nodes[pid].get("title", "?")[:60]
            logger.info("  PageRank: %.4f — %s", rank, title)

        # -------------------------------------------------------------------
        # 8. MAP-Elites Frontier — seed from store
        # -------------------------------------------------------------------
        logger.info("=" * 60)
        logger.info("PHASE 8: MAP-Elites Frontier (seed only)")
        logger.info("=" * 60)

        from discover.frontier import FrontierExplorer

        explorer = FrontierExplorer(store=store, embedder=embedder, llm=None)
        await explorer.seed()
        logger.info("  Archive seeded: %d entries, coverage=%.2f%%",
                     explorer._archive.size(), explorer._archive.coverage() * 100)

        store.close()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("LIVE TEST COMPLETE in %.1fs", elapsed)
    logger.info("  Papers discovered: %d", len(candidates))
    logger.info("  Papers embedded: %d", len(embeddings))
    logger.info("  Papers stored: %d", len(embeddings))
    logger.info("  Search results: %d", len(results))
    logger.info("  RAG answer length: %d chars", len(answer))
    logger.info("  SMT status: %s", status)
    logger.info("  Graph nodes: %d", graph.node_count())
    logger.info("  Frontier coverage: %.2f%%", explorer._archive.coverage() * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
