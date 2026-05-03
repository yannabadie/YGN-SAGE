# sage-discover

Knowledge Discovery Engine for YGN-SAGE. Discovers, curates, and ingests research papers with three innovations:

1. **ClaimGraph + SMT Verification** — extracts scientific claims, classifies relations (supports/extends/refutes), verifies logical consistency via OxiZ
2. **MAP-Elites Frontier Explorer** — evolutionary search maximizing research coverage across 4 dimensions (domain, recency, citation velocity, novelty)
3. **Adaptive Curation** — kNN learned preferences + LinUCB bandit fusion that improves from user feedback

## What's real vs aspirational

**Fully implemented and tested (100 tests, canonical: `docs/status/current.json`):**
- Discovery pipeline: arXiv + Semantic Scholar + HuggingFace APIs (real API calls, rate-limit tolerant)
- SPECTER2 (768d) dense embeddings for scientific papers (downloaded, GPU-accelerated)
- SPLADE sparse embeddings for hybrid search (downloaded, torch-based)
- Cross-encoder reranking (ms-marco-MiniLM, 22.7M params)
- Qdrant local vector store with named vectors (dense + sparse), no server needed
- Reciprocal Rank Fusion (RRF) for merging dense + sparse search results
- Docling PDF-to-structured-text extraction with section parsing
- Citation graph (NetworkX DiGraph) with PageRank, Louvain communities, betweenness centrality
- Claim extraction via LLM prompting with JSON parsing + markdown fence stripping
- SMT claim verification using sage_core.SmtVerifier.verify_arithmetic() — detects contradictions between quantitative claims about the same method+benchmark
- kNN curator (cosine similarity, distance-weighted majority vote, same architecture as knn_router.py)
- LinUCB contextual bandit for fusing 3 curation signals (kNN, LLM, heuristic)
- RAG pipeline (hybrid search + LLM answer generation with paper citations)
- MAP-Elites archive (4D grid, 240 cells, insertion/replacement by fitness)
- MCP server exposing 5 tools (discover, curate, query, explore_frontier, verify_claims)
- Heuristic filter (abstract length, staleness, blocklist patterns)
- Legacy ExoCortex (Google GenAI File Search) ingestion preserved as optional sync target

**Limitations (honest):**
- SPLADE sparse encoding works but quality is untested on scientific text specifically
- Claim extraction depends on LLM quality — flan-t5-small not yet wired (uses LLM provider)
- SMT verification only catches contradictions in quantitative claims (e.g., "X achieves 92% on B" vs "X achieves 45% on B"). Qualitative claims return "unknown"
- MAP-Elites frontier exploration requires LLM for query generation (keyword fallback without)
- Adaptive curation bandit starts with uniform weights — needs ~50 feedback rounds to converge
- kNN curator needs exemplar data (accepted/rejected papers) to be useful — starts neutral
- No SciBERT NER for entity extraction (mentioned in spec, not implemented)
- No automated survey generation (Approach C scope, not B)
- Semantic Scholar rate-limits aggressively (429s) — discovery can take 2-5 minutes
- Qdrant payload indexes have no effect in local mode (warning expected)

## Installation

```bash
cd sage-discover
pip install -e ".[dev]"
python -m pytest tests/ -v    # 100 tests (canonical: docs/status/current.json)
```

Dependencies: qdrant-client, sentence-transformers, docling, networkx, mcp, transformers, torch.
Models downloaded on first use: allenai/specter2_base (~440MB), naver/splade-cocondenser-ensembledistil (~440MB), cross-encoder/ms-marco-MiniLM-L6-v2 (~90MB).

## Usage

```bash
# Pipeline modes
python -m discover.pipeline --mode nightly                          # Papers from yesterday
python -m discover.pipeline --mode on-demand --query "MARL"         # Targeted search
python -m discover.pipeline --mode migrate                          # Bootstrap from NotebookLM
python -m discover.pipeline --mode mcp                              # Start MCP server

# MCP server (for Claude Code, other agents)
python -m discover.mcp
```

## Architecture

```
Discovery (arXiv + S2 + HF)
    ↓
Adaptive Curation (kNN + LLM + heuristic → LinUCB bandit)
    ↓
Embedding (SPECTER2 dense + SPLADE sparse)
    ↓
Qdrant Store (papers + claims collections)
    ↓
┌──────────────┬────────────────┬──────────────────┐
│ Claim        │ Citation       │ Frontier         │
│ Extraction   │ Graph          │ Explorer         │
│ + SMT Verify │ (PageRank)     │ (MAP-Elites 4D)  │
└──────────────┴────────────────┴──────────────────┘
    ↓
RAG (hybrid search + rerank + LLM answer)
    ↓
MCP Server (5 tools)
```

## Source Modules (`src/discover/`)

For detailed code documentation (data flow diagrams, method signatures, algorithm explanations), see [`src/discover/README.md`](src/discover/README.md).

| Module | Lines | Purpose |
|--------|-------|---------|
| `pipeline.py` | 118 | Orchestrator — wires discovery → curation → ingestion. Supports Qdrant (primary) and ExoCortex (legacy fallback) |
| `discovery.py` | 372 | Multi-source paper discovery: arXiv, Semantic Scholar, HuggingFace. 5 research domains, deduplication by normalized title |
| `curator.py` | 206 | Legacy curation: heuristic filter + LLM scoring (threshold >= 6/10) |
| `adaptive_curator.py` | 119 | Innovation #3: KnnCurator + CurationBandit (LinUCB) + adaptive_curate() |
| `store.py` | 133 | Qdrant local wrapper: papers + claims collections, dense + sparse vectors, payload filtering |
| `embeddings.py` | 115 | SPECTER2 dense + SPLADE sparse + cross-encoder reranker + RRF |
| `extractor.py` | 83 | Docling PDF-to-markdown with section parsing (intro/methodology/results/conclusion) |
| `claim_graph.py` | 220 | Innovation #1: Claim extraction, relation classification, OxiZ SMT verification |
| `citation_graph.py` | 62 | NetworkX DiGraph: PageRank, Louvain communities, betweenness centrality |
| `frontier.py` | 160 | Innovation #2: MAP-Elites 4D archive (240 cells), coverage-driven exploration |
| `rag.py` | 72 | Hybrid search (RRF dense+sparse+rerank) + LLM RAG answer generation |
| `mcp.py` | 131 | FastMCP server: discover, curate, query, explore_frontier, verify_claims |
| `ingestion.py` | 156 | Qdrant upsert + legacy ExoCortex upload + manifest tracking |
| `migration.py` | 138 | NotebookLM markdown import with arXiv ID extraction |
| `model_watcher.py` | 79 | Model registry monitoring for unprofiled models |

## Tests

95 tests, 0 failures, ~22 seconds:

| Test file | Tests | Coverage |
|-----------|-------|----------|
| test_store.py | 8 | KnowledgeStore CRUD, search, filtering |
| test_embeddings.py | 5 | SPECTER2/SPLADE/reranker encoding, RRF fusion |
| test_extractor.py | 4 | Section parsing, Docling mock, error fallback |
| test_citation_graph.py | 6 | PageRank, Louvain, betweenness, node/edge ops |
| test_claim_graph.py | 10 | Claim extraction, relation classification, SMT verification (consistent + contradictory) |
| test_adaptive_curator.py | 7 | kNN scoring, bandit decision/update/learning, adaptive_curate E2E |
| test_rag.py | 3 | Hybrid search, RAG with/without LLM |
| test_frontier.py | 9 | Archive insert/replace/reject, coverage, descriptor computation, seeding |
| test_mcp.py | 3 | Tool invocation with mocked components |
| test_pipeline.py | 4 | Nightly, no-LLM fallback, migrate mode |
| test_ingestion.py | 5 | Qdrant ingest, dedup, manifest roundtrip |
| test_integration.py | 2 | Full nightly E2E, store-to-RAG flow |
| test_discovery.py | 5 | DOMAINS structure, dedup, arXiv mock |
| test_curator.py | 7 | Heuristic filter, LLM scoring, blocklist |
| test_migration.py | 7 | arXiv ID extraction, markdown migration |
| test_model_watcher.py | 8 | Unprofiled model detection, report generation |

## Research References

Built on analysis of 18 SOTA papers and 12 open-source systems:
- PaperQA2 (arXiv 2409.13740) — superhuman RAG
- OpenScholar (Nature Feb 2026, arXiv 2411.14199) — 45M papers, self-feedback
- STORM (Stanford, arXiv 2402.14207) — multi-perspective
- AI Scientist v2 (Sakana, arXiv 2504.08066) — end-to-end research
- PaSa (arXiv 2501.10120) — RL-trained search
- ClaimFlow (arXiv 2603.16073) — claim relation taxonomy
- SurveyG (arXiv 2510.07733) — hierarchical citation graph
- See `docs/superpowers/specs/2026-03-27-discover-knowledge-pipeline-design.md` for full list
