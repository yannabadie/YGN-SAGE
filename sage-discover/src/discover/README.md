# sage-discover — Source Code Documentation

## How It Works

sage-discover is a pipeline that turns academic paper APIs into a searchable, verified knowledge store. Here's the data flow:

```
arXiv API ──┐
S2 API ─────┤──> PaperCandidate[] ──> Adaptive Curation ──> SPECTER2+SPLADE embed
HF Hub API ─┘        │                     │                       │
                 deduplicate          kNN + LLM + heuristic    dense 768d + sparse
                 by title             fused via LinUCB bandit       │
                                           │                       ▼
                                      CuratedPaper[]          Qdrant upsert
                                           │                  (papers collection)
                                           │                       │
                                           ▼                       ▼
                                    Claim Extraction          Hybrid Search
                                    (LLM prompting)       (RRF dense+sparse
                                           │                + rerank)
                                           ▼                       │
                                    OxiZ SMT Verify               ▼
                                    (contradictions)          RAG Answer
                                           │              (LLM + citations)
                                           ▼
                                    Citation Graph
                                    (PageRank, Louvain)
```

## Module Reference

### `pipeline.py` — Orchestrator (128 lines)

The entry point. Coordinates all components with graceful degradation.

```python
async def run_pipeline(
    mode="nightly",     # nightly | on-demand | migrate | watch | mcp
    query=None,         # search query (on-demand mode)
    since=None,         # date filter (default: yesterday)
    domains=None,       # restrict to specific research domains
    exocortex=None,     # legacy Google GenAI File Search
    llm=None,           # LLM provider (auto-detects GoogleProvider)
    store=None,         # KnowledgeStore (auto-initializes Qdrant)
    embedder=None,      # EmbeddingPipeline (auto-initializes SPECTER2)
) -> PipelineReport:
```

**Fallback chain:**
1. Curation: `adaptive_curate()` -> `curate()` -> `heuristic_filter()` (no LLM)
2. Ingestion: `ingest_all_to_store()` (Qdrant) -> `ingest_all()` (ExoCortex)
3. Components: auto-initialized if not provided, silently skipped if unavailable

Returns `PipelineReport(discovered=N, curated=N, ingested=N)`.

---

### `discovery.py` — Paper Discovery (375 lines)

Searches 3 academic sources in parallel for each domain.

**5 research domains** (hard-coded with arXiv categories + keywords):
- `marl` — Multi-Agent Reinforcement Learning (cs.MA, cs.AI, cs.GT)
- `cognitive_architectures` — Metacognition, dual-process (cs.AI, cs.CL, cs.NE)
- `formal_verification` — SMT, PRM, program synthesis (cs.LO, cs.SE, cs.PL)
- `evolutionary_computation` — MAP-Elites, QD, NAS (cs.NE, cs.AI)
- `memory_systems` — Episodic, RAG, knowledge graphs (cs.AI, cs.CL, cs.IR)

**Sources:**
- `discover_arxiv()` — arXiv API via `arxiv` library. Returns preprints with categories.
- `discover_semantic_scholar()` — S2 API via `semanticscholar` library. Returns citation counts. Configured with `retry=False, timeout=10` for fast-fail without API key.
- `discover_hf()` — HuggingFace Hub API. Returns model cards and dataset papers.

**Deduplication:** Papers from different sources with the same normalized title (lowercase, no punctuation) are merged. The version with the highest citation count is kept.

**Key dataclass:**
```python
@dataclass
class PaperCandidate:
    paper_id: str           # Unique ID from source (e.g., "2505.12601")
    title: str
    authors: list[str]
    abstract: str
    source: str             # "arxiv" | "s2" | "hf"
    domain: str             # One of 5 DOMAINS keys
    published: date
    pdf_url: str | None
    citation_count: int
    influential_citation_count: int = 0
```

---

### `store.py` — Qdrant Knowledge Store (178 lines)

Local vector database wrapping Qdrant. No server needed, runs in-process.

**Two collections:**

**`papers`** — Dense (SPECTER2 768d) + Sparse (SPLADE) named vectors
- Payload: title, authors, abstract, domain, source, year, citation_count, relevance_score, key_insights, pdf_path, ingested_at
- Payload indexes: domain (keyword), year (integer)

**`claims`** — Dense (SPECTER2 768d) vectors only
- Payload: statement, paper_id, claim_type, confidence, smt_verified, smt_status, smt_formula, relations
- Payload indexes: paper_id (keyword), smt_status (keyword)

**ID mapping:** String paper IDs are hashed to deterministic integers via SHA256 (Qdrant local mode requires int point IDs).

**Key methods:**
```python
store.upsert_paper(paper_id, dense_vector, sparse_vector, payload)
store.get_paper(paper_id) -> dict | None
store.search_dense(query_vector, limit=10, domain=None) -> list[dict]
store.search_sparse(sparse_vector, limit=10) -> list[dict]
store.upsert_claim(claim_id, dense_vector, payload)
store.get_claims_for_paper(paper_id) -> list[dict]
```

---

### `embeddings.py` — Embedding Pipeline (125 lines)

Manages 3 models with lazy loading (models downloaded on first use):

| Model | Size | Purpose | Load |
|-------|------|---------|------|
| `allenai/specter2_base` | 110M | Dense paper embeddings (768d) | Eager at init |
| `naver/splade-cocondenser-ensembledistil` | 110M | Sparse vocab-level weights (30K dims) | Eager at init |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | Pairwise reranking | Lazy on first `rerank()` |

**SPLADE handling:** The sparse encoder runs on GPU but returns a sparse CUDA tensor that doesn't support `nonzero()`. The fix: `cpu().to_dense()` before extracting non-zero indices. Produces ~100-150 non-zero entries per paper.

**RRF (Reciprocal Rank Fusion):**
```python
def reciprocal_rank_fusion(*result_lists, k=60) -> list[dict]:
    # Score per item = sum over lists of 1/(k + rank + 1)
    # Items appearing in multiple lists get boosted
```

---

### `extractor.py` — PDF Extraction (90 lines)

Converts PDFs to structured markdown using Docling (IBM's document parser).

**Section parsing:** Regex-based detection of Introduction, Methodology, Results, Conclusion from markdown headings. Handles numbered sections (`## 1. Introduction`) and alternative names (`## Approach`, `## Experiments`, `## Discussion`).

```python
extract_full_text(pdf_path) -> {
    "full_text": str,           # Complete markdown
    "sections": {
        "introduction": str,
        "methodology": str,
        "results": str,
        "conclusion": str,
    },
    "tables": list[str],        # Markdown tables
    "error": str | None,
}
```

Graceful degradation: returns `None` fields if Docling is not installed or extraction fails.

---

### `citation_graph.py` — Citation Graph (57 lines)

NetworkX directed graph where nodes are papers and edges are citations.

**Analysis methods:**
- `pagerank(alpha=0.85)` — Identifies influential papers (cited by many)
- `communities()` — Louvain clustering on undirected projection, finds research clusters
- `bridges()` — Betweenness centrality, finds papers connecting different research areas
- `neighbors(paper_id, direction)` — Get citing/cited papers

```python
builder = CitationGraphBuilder()
builder.add_paper("p1", title="Paper A", year=2025)
builder.add_citation("p1", "p2")  # p1 cites p2
ranks = builder.pagerank()  # {"p1": 0.15, "p2": 0.35, ...}
```

---

### `claim_graph.py` — SMT-Verified Claims (269 lines) — Innovation #1

The core scientific contribution. Extracts claims from papers, classifies their relationships, and verifies logical consistency using formal methods.

**Step 1: Claim Extraction** (LLM-based)
```python
claims = await extract_claims_from_text(abstract, paper_id, llm)
# Returns: [Claim(statement="X achieves 92% accuracy", type="finding", confidence=0.9)]
```
Uses a structured JSON prompt. Handles markdown fence stripping and JSON parse failures gracefully.

**Step 2: Relation Classification** (LLM-based)
```python
relation = await classify_relation(claim_a, claim_b, llm)
# Returns: ClaimRelation(source_id, target_id, type="supports"|"refutes"|...)
```

**Step 3: SMT Translation** (regex-based)

Three patterns, tried in order of specificity:
1. `_COMPARE_PATTERN` — "Method X achieves 92% on benchmark B" -> `(= perf_method_benchmark 92)`
2. `_IMPROVE_PATTERN` — "improves over baseline by 5pp" -> `(= improvement_id 5)`
3. `_PERF_PATTERN` — "achieves 92% accuracy" -> `(= perf_id 92)`

Qualitative claims ("more elegant", "simpler") -> `None` (not translatable).

**Step 4: SMT Verification** (sage_core.SmtVerifier)
```python
status = verify_claim_cluster(claims)  # "consistent" | "contradictory" | "unknown"
```

Groups claims by variable key (same method + benchmark). For each pair in a group, calls `sage_core.SmtVerifier.verify_arithmetic(val_a, val_b, tolerance=0)`. If any pair has different values, result is "contradictory".

**Example:** Paper A says "X achieves 92% on B", Paper B says "X achieves 45% on B". Same variable key `perf_x_b`, different values. `verify_arithmetic(92, 45, 0) = False` -> "contradictory".

---

### `adaptive_curator.py` — Adaptive Curation (119 lines) — Innovation #3

Replaces static LLM-only scoring with a learning system.

**Three signals fused:**
1. `knn_score` — KnnCurator: cosine similarity to accepted/rejected paper exemplars, distance-weighted majority vote (same architecture as `strategy/knn_router.py` which achieves 92% routing accuracy)
2. `llm_score` — LLM relevance rating (0-10, normalized to 0-1)
3. `heuristic_score` — Binary: passed heuristic filter or not

**LinUCB Bandit:**
```
context x = [knn_score, llm_score, heuristic_score]
score = theta @ x + alpha * sqrt(x @ A_inv @ x)    # UCB exploration bonus
accept = score > 0.5
```

After user feedback (`reward=1` useful, `reward=0` not), the bandit updates:
```
A += outer(x, x)
b += reward * x
theta = inv(A) @ b
```

Starts with uniform weights (0.33 each). After ~50 feedback rounds, converges to the optimal signal weighting.

---

### `rag.py` — RAG Pipeline (73 lines)

Hybrid search followed by LLM answer generation.

**Search flow:**
1. Encode query with SPECTER2 (dense) and SPLADE (sparse)
2. Search Qdrant on both vector types (3x limit each)
3. Fuse results via RRF (items in both lists get boosted)
4. Rerank top-2k candidates with cross-encoder
5. Return top-k

**Answer generation:**
- With LLM: structured prompt with paper context -> synthesized answer with citations
- Without LLM: returns "Found N relevant papers:" + paper summaries

---

### `frontier.py` — MAP-Elites Frontier Explorer (191 lines) — Innovation #2

Evolutionary algorithm maximizing *diversity* of research coverage.

**4D Behavior Descriptor:**
| Dimension | Bins | Meaning |
|-----------|------|---------|
| `domain_idx` | 5 | Which research domain (marl, cognitive, formal, evolution, memory) |
| `recency` | 4 | How old (0=today, 1=1+ year) |
| `citation_velocity` | 4 | Citations per month normalized (0=none, 1=50+) |
| `novelty` | 3 | 1 - max cosine similarity to existing papers |

**Grid:** 5 x 4 x 4 x 3 = **240 cells**. Each cell holds the paper with the highest fitness (relevance score).

**Exploration loop (per generation):**
1. Find empty/under-covered cells
2. For each target cell, generate a search query (LLM or keyword fallback)
3. Discover papers matching the query
4. Compute descriptors, insert best into archive

**Key insight:** Unlike naive search which keeps finding the same popular papers, MAP-Elites forces exploration of under-covered regions. If you have 50 MARL papers but 0 formal verification papers, the next generation prioritizes formal verification.

---

### `mcp.py` — MCP Server (161 lines)

Exposes the pipeline as 5 tools callable by any MCP-compatible agent (Claude Code, other agents).

| Tool | Signature | What it does |
|------|-----------|--------------|
| `discover_papers` | `(query, domains?, since?, max_results?)` | Discover papers from 3 sources |
| `curate_papers` | `(paper_ids)` | Retrieve stored papers with scores |
| `query_knowledge` | `(question, top_k?, domain?)` | RAG query with hybrid search |
| `explore_frontier` | `(domain?, generations?)` | MAP-Elites frontier exploration |
| `verify_claims` | `(paper_id)` | Extract + SMT-verify claims |

**Lazy initialization:** Components (Qdrant, SPECTER2, LLM) are created on first tool call and cached in a module-level `_components` dict.

```bash
# Run as MCP server
python -m discover.mcp

# Or via CLI
python -m discover.pipeline --mode mcp
```

---

### `curator.py` — Legacy Curation (207 lines)

The original curation pipeline, preserved as fallback.

**Three-stage filter:**
1. **Heuristic filter** — Rejects: abstract < 100 chars, published > 90 days ago with 0 citations, title matches blocklist ("survey of surveys", "correction to", "erratum")
2. **LLM scoring** — Batch of 20 papers -> JSON prompt asking for 0-10 relevance score + reason + key insights. Score 5 fallback on parse failure.
3. **Threshold** — Keep only papers with score >= 6

Used by `adaptive_curator.py` internally (calls `heuristic_filter()` and `llm_score()`).

---

### `ingestion.py` — Paper Ingestion (251 lines)

Two ingestion paths:

**Qdrant (primary):**
```python
await ingest_to_store(paper, store, embedder)
# 1. Check dedup (store.get_paper)
# 2. Download PDF if available (HTTPS only)
# 3. Embed with SPECTER2+SPLADE
# 4. Upsert to Qdrant with full metadata payload
```

**ExoCortex (legacy fallback):**
```python
await ingest(paper, exocortex, manifest_path)
# 1. Check manifest JSON for dedup
# 2. Download PDF or create markdown fallback
# 3. Upload to Google GenAI File Search
# 4. Record in manifest
```

---

### `migration.py` — NotebookLM Migration (139 lines)

Imports existing research from NotebookLM markdown exports.

- Extracts arXiv IDs via 3 regex patterns (`arXiv:XXXX.XXXXX`, URL form, bare ID with version)
- Maps notebook filenames to domains (`technical` -> cognitive_architectures, etc.)
- Uploads each markdown file to ExoCortex

---

### `model_watcher.py` — Model Registry Monitor (80 lines)

Compares models available in sage's ModelRegistry against TOML profiles. Reports models with `cost_input=0, cost_output=0` as unprofiled, needing new entries in `cards.toml`.

---

## Cross-Module Dependencies

```
pipeline.py
  ├── discovery.discover()
  ├── adaptive_curator.adaptive_curate()  ──> curator.heuristic_filter() + llm_score()
  ├── ingestion.ingest_all_to_store()     ──> store.upsert_paper() + embeddings.embed_paper()
  └── migration.migrate_notebooks()       ──> exocortex.upload()

mcp.py (lazy init)
  ├── store.KnowledgeStore
  ├── embeddings.EmbeddingPipeline
  ├── rag.RAGPipeline                     ──> store + embeddings + reciprocal_rank_fusion
  ├── frontier.FrontierExplorer           ──> store + embeddings + discovery.discover()
  └── claim_graph.extract_claims + verify ──> sage_core.SmtVerifier
```

## Testing

95 tests across 16 test files. All use `pytest-asyncio` with `asyncio_mode="auto"`.

**Mocking approach:** External APIs (arXiv, S2, HF, LLM, ExoCortex) are fully mocked via `unittest.mock.AsyncMock` and `MagicMock`. Qdrant runs in real local mode (temp directories). SPECTER2/SPLADE are mocked at the module level via `@patch`.

```bash
python -m pytest tests/ -v          # All 95 tests
python -m pytest tests/ -k store    # Just store tests
python -m pytest tests/ -k claim    # Just claim graph tests
```
