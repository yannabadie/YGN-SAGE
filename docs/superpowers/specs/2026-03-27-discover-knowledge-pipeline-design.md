# sage-discover Knowledge Pipeline — Approach B Design Spec

**Date**: 2026-03-27
**Branch**: `feat/discover-knowledge`
**Status**: Approved
**Scope**: SOTA parity + 3 scientific innovations unique to SAGE

## 1. Executive Summary

Transform sage-discover from a basic arXiv-to-ExoCortex ingestion pipeline into a
**formally verified knowledge discovery engine** with three innovations no existing
system (PaperQA2, OpenScholar, STORM, AI Scientist v2, PaSa) provides:

1. **ClaimGraph with SMT verification** — extract scientific claims, classify
   relations (supports/extends/refutes), verify logical consistency via OxiZ
2. **MAP-Elites Frontier Explorer** — evolutionary search maximizing *diversity*
   of research frontier coverage across 4 behavior dimensions
3. **Adaptive Curation** — kNN-learned + bandit-fused paper relevance scoring
   that improves from user feedback

Additionally: local Qdrant hybrid search (SPECTER2 + SPLADE), Docling full-text
PDF extraction, citation graph with PageRank, and MCP server exposure.

## 2. Architecture

```
                         ┌─────────────────────────────────────┐
                         │         MCP Server (FastMCP)        │
                         │  discover / curate / query /        │
                         │  explore_frontier / verify_claims   │
                         └──────────┬──────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
  ┌──────────────┐        ┌────────────────┐         ┌────────────────┐
  │   Discovery  │        │  On-Demand RAG │         │ Frontier       │
  │   Pipeline   │        │    Query       │         │ Explorer       │
  │  (enhanced)  │        │                │         │ (MAP-Elites)   │
  └──────┬───────┘        └───────┬────────┘         └───────┬────────┘
         │                        │                          │
         ▼                        ▼                          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     Adaptive Curator                             │
  │  kNN relevance (learned) + LLM scoring + self-feedback verify   │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
  │   Docling    │   │   Claim      │   │   Citation       │
  │   Extractor  │   │   Extractor  │   │   Graph Builder  │
  │  (full-text) │   │  (flan-t5)   │   │  (NetworkX)      │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    Knowledge Store (Qdrant local)                │
  │  Named vectors: SPECTER2 (dense) + SPLADE (sparse)              │
  │  Payloads: paper metadata, claims, domain, scores               │
  │  Collections: papers, claims, reading_paths                     │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    ClaimGraph (Innovation #1)                    │
  │  Claims extracted -> relations (supports/extends/refutes)       │
  │  OxiZ SMT verification of logical consistency                   │
  │  NetworkX DiGraph + PageRank credit propagation                 │
  └─────────────────────────────────────────────────────────────────┘
```

### Data Flow (nightly mode)

```
1. Discovery    arXiv + Semantic Scholar + HuggingFace APIs
                    |
2. Extraction   Docling full-text (PDF -> markdown + tables)
                    |
3. Embedding    SPECTER2 (768d dense) + SPLADE (sparse)
                    |
4. Curation     kNN score + LLM score + heuristic -> bandit fusion
                Self-feedback: verify coverage gaps, iterate if needed
                    |
5. Ingestion    Upsert to Qdrant (papers collection)
                    |
6. Claims       flan-t5-small extraction -> SciBERT NLI relations
                OxiZ SMT consistency check per cluster
                Upsert to Qdrant (claims collection)
                    |
7. Graph        Citation edges from S2 API -> NetworkX DiGraph
                PageRank scoring -> update paper payloads
                    |
8. Frontier     MAP-Elites: select under-covered regions,
                generate exploration queries, discover, repeat
```

## 3. Knowledge Store — Qdrant Local

Replaces Google GenAI File Search (ExoCortex) as primary backend.
ExoCortex remains as optional cloud sync target.

### Collection: `papers`

```python
vectors_config = {
    "specter2": models.VectorParams(size=768, distance=models.Distance.COSINE),
}
sparse_vectors_config = {
    "splade": models.SparseVectorParams(),
}
quantization_config = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8,
        quantile=0.99,
        always_ram=True,
    )
)
```

**Payload schema:**
```json
{
  "title": "str",
  "authors": ["str"],
  "abstract": "str",
  "full_text": "str | null",
  "domain": "str",
  "source": "arxiv | s2 | hf",
  "year": "int",
  "arxiv_id": "str | null",
  "s2_id": "str | null",
  "doi": "str | null",
  "pdf_url": "str | null",
  "pdf_path": "str | null",
  "citation_count": "int",
  "influential_citation_count": "int",
  "citation_velocity": "float",
  "relevance_score": "float",
  "curation_method": "knn | llm | heuristic | bandit",
  "claims_ids": ["str"],
  "pagerank": "float",
  "ingested_at": "str (ISO)",
  "sections": {
    "introduction": "str | null",
    "methodology": "str | null",
    "results": "str | null",
    "conclusion": "str | null"
  }
}
```

### Collection: `claims`

```python
vectors_config = {
    "specter2": models.VectorParams(size=768, distance=models.Distance.COSINE),
}
```

**Payload schema:**
```json
{
  "statement": "str",
  "paper_id": "str",
  "section": "str (introduction|methodology|results|conclusion)",
  "confidence": "float (0-1)",
  "claim_type": "finding | method | limitation | hypothesis",
  "relations": [
    {"target_claim_id": "str", "type": "supports|extends|refutes|qualifies"}
  ],
  "smt_verified": "bool",
  "smt_status": "consistent | contradictory | unknown | not_checked",
  "smt_formula": "str | null"
}
```

### Collection: `reading_paths`

```python
# No vector search needed; payload-only collection
```

**Payload schema:**
```json
{
  "query": "str",
  "nodes": [
    {"paper_id": "str", "role": "foundation|development|frontier", "order": "int"}
  ],
  "edges": [
    {"from_paper": "str", "to_paper": "str", "relation": "cites|extends|refutes"}
  ],
  "quality_score": "float",
  "generated_at": "str (ISO)"
}
```

### Hybrid Search

Every query uses reciprocal rank fusion (RRF) of dense + sparse results:

```python
async def hybrid_search(query: str, top_k: int = 10, domain: str | None = None):
    dense_emb = specter2.encode(query)
    sparse_emb = splade.encode(query)

    dense_results = client.query_points("papers", query=dense_emb, using="specter2", limit=top_k * 2)
    sparse_results = client.query_points("papers", query=sparse_emb, using="splade", limit=top_k * 2)

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    reranked = cross_encoder.rank(query, fused[:top_k * 2])
    return reranked[:top_k]
```

## 4. ClaimGraph — Innovation #1 (SMT-Verified Claims)

No existing system (PaperQA2, OpenScholar, STORM, AI Scientist, PaSa) performs
formal verification of scientific claim consistency.

### 4.1 Claim Extraction

**Model**: `google/flan-t5-small` (77M params) with structured prompts.

```python
CLAIM_EXTRACTION_PROMPT = """Extract the main scientific claims from this text.
For each claim, provide:
- statement: the claim in one sentence
- type: finding | method | limitation | hypothesis
- confidence: 0.0-1.0 (how certain the authors are)

Text: {text}

Output JSON array:"""
```

**Fallback**: If flan-t5-small quality is insufficient on scientific text, escalate
to the pipeline's LLM provider (e.g., Gemini Flash Lite) for claim extraction.
The local model handles 80%+ of cases; LLM handles edge cases.

### 4.2 Relation Classification

**Model**: `allenai/scibert_scivocab_uncased` with NLI head, zero-shot.

For each pair of claims (A, B) from different papers in the same domain:

```python
RELATION_PROMPT = """
Claim A: "{claim_a}"
Claim B: "{claim_b}"

What is the relationship?
- supports: B provides evidence for A
- extends: B builds upon A with new contributions
- refutes: B contradicts A
- qualifies: B limits the scope of A
- independent: no direct relationship

Relationship:"""
```

**Optimization**: Only compare claims within the same domain AND with
SPECTER2 cosine similarity > 0.5 (pre-filter to avoid O(n^2) comparisons).

### 4.3 SMT Verification (OxiZ)

For each cluster of related claims, translate quantitative assertions into
SMT-LIB2 formulas and check satisfiability:

**Translatable claims** (automated):
- Performance comparisons: "X achieves Y% accuracy" -> `(= perf_X Y)`
- Relative improvements: "X improves over Y by Z%" -> `(> perf_X (+ perf_Y Z))`
- Ordering claims: "X > Y > Z" -> `(and (> X Y) (> Y Z))`
- Threshold claims: "X requires at least N samples" -> `(>= samples_X N)`

**Non-translatable claims** (flagged as `smt_status: unknown`):
- Qualitative claims ("X is more elegant than Y")
- Causal claims without quantitative evidence
- Claims about methodology without measurable outcomes

```python
async def verify_claim_cluster(claims: list[Claim], relations: list[Relation]) -> str:
    """Returns: consistent | contradictory | unknown"""
    verifier = sage_core.SmtVerifier()
    verifier.set_logic("QF_LIA")

    # Translate each claim to SMT assertions
    formulas = []
    for claim in claims:
        formula = translate_claim_to_smt(claim)
        if formula:
            formulas.append(formula)
            verifier.assert_(formula)

    if not formulas:
        return "unknown"

    result = verifier.check_sat()
    if result == "sat":
        return "consistent"
    elif result == "unsat":
        return "contradictory"
    else:
        return "unknown"
```

**Output**: Contradictions are surfaced as **research opportunities** —
where two papers disagree, there is a gap worth investigating.

### 4.4 PageRank Credit Propagation

Using RewardFlow pattern (OpenReview 5oGJbM5u86) on the citation graph:

```python
import networkx as nx

def compute_claim_credit(citation_graph: nx.DiGraph) -> dict[str, float]:
    """PageRank on citation graph gives per-paper importance.
    Propagate to claims proportionally."""
    paper_rank = nx.pagerank(citation_graph, alpha=0.85)

    claim_credit = {}
    for paper_id, rank in paper_rank.items():
        paper = get_paper(paper_id)
        n_claims = len(paper.claims_ids)
        if n_claims > 0:
            per_claim = rank / n_claims
            for claim_id in paper.claims_ids:
                claim_credit[claim_id] = per_claim

    return claim_credit
```

## 5. MAP-Elites Frontier Explorer — Innovation #2

Uses sage-core's EvolutionEngine to explore the research frontier with
maximum diversity, not just relevance.

### 5.1 Behavior Descriptors (4D grid)

```python
@dataclass
class FrontierDescriptor:
    domain_idx: int           # 0-4 (5 research domains), 5 bins
    recency: float            # 0-1 (days_old / 365, clamped), 4 bins
    citation_velocity: float  # 0-1 (citations/month normalized), 4 bins
    novelty: float            # 0-1 (1 - max_cosine_sim to archive), 3 bins

# Grid: 5 * 4 * 4 * 3 = 240 cells
# Each cell holds the best paper (highest relevance) for that descriptor combo
```

### 5.2 Exploration Loop

```python
async def explore_frontier(self, generations: int = 5) -> FrontierReport:
    archive = MapElitesArchive(descriptor_bins=[5, 4, 4, 3])

    # Seed: top papers from each domain in Qdrant
    seed_papers = await self._get_seed_papers()
    for paper in seed_papers:
        desc = self._compute_descriptor(paper)
        archive.try_insert(paper, desc, fitness=paper.relevance_score)

    for gen in range(generations):
        # 1. Select under-covered regions
        empty_cells = archive.get_empty_cells()
        low_quality_cells = archive.get_cells_below_threshold(0.5)
        target_cells = empty_cells + low_quality_cells

        # 2. For each target, generate exploration query via LLM
        for cell in target_cells[:10]:  # batch of 10 per generation
            target_desc = archive.cell_to_descriptor(cell)
            query = await self._generate_exploration_query(target_desc)

            # 3. Discover papers matching this query
            candidates = await discover(
                since=self._recency_to_date(target_desc.recency),
                query=query,
                domains=[DOMAINS_LIST[target_desc.domain_idx]],
            )

            # 4. Score and insert into archive
            for paper in candidates:
                desc = self._compute_descriptor(paper)
                archive.try_insert(paper, desc, fitness=paper.relevance_score)

    return FrontierReport(
        coverage=archive.coverage(),
        total_papers=archive.size(),
        empty_regions=archive.get_empty_cells(),
        best_per_domain=archive.get_best_per_dimension(0),
    )
```

### 5.3 Query Generation (mutation)

```python
async def _generate_exploration_query(self, target: FrontierDescriptor) -> str:
    domain_name = DOMAINS_LIST[target.domain_idx]
    recency_hint = "very recent" if target.recency < 0.1 else "established"
    novelty_hint = "highly novel, unexplored" if target.novelty > 0.7 else "well-studied"

    prompt = f"""Generate a specific arXiv search query for:
    - Domain: {domain_name}
    - Paper type: {recency_hint}
    - Desired novelty: {novelty_hint}
    - Focus: areas underrepresented in current knowledge base

    Return only the search query string, no explanation."""

    response = await self.llm.generate(prompt)
    return response.text.strip()
```

## 6. Adaptive Curation — Innovation #3 (kNN + Bandit)

### 6.1 Three-Signal Architecture

```python
@dataclass
class CurationSignals:
    knn_score: float      # kNN on SPECTER2 embeddings (learned from past accepts/rejects)
    llm_score: float      # LLM relevance rating (0-10, normalized to 0-1)
    heuristic_score: float  # Rules-based (abstract length, citations, recency)
```

### 6.2 kNN Relevance (reuses sage knn_router.py architecture)

```python
class KnnCurator:
    """Same architecture as strategy/knn_router.py (92% GT accuracy)
    but trained on paper accept/reject exemplars instead of routing labels."""

    def __init__(self, exemplars_path: Path):
        self.exemplars = load_exemplars(exemplars_path)  # .npz file
        # Features: SPECTER2 embeddings of paper (title + abstract)
        # Labels: 1 (accepted/useful) or 0 (rejected/not useful)

    def score(self, paper_embedding: np.ndarray, k: int = 7) -> float:
        """Distance-weighted majority vote, exactly like kNN router."""
        distances, labels = self._find_neighbors(paper_embedding, k)
        weights = 1.0 / (distances + 1e-6)
        return float(np.average(labels, weights=weights))
```

### 6.3 Contextual Bandit Fusion (LinUCB)

```python
class CurationBandit:
    """LinUCB bandit that learns optimal fusion weights for 3 signals."""

    def __init__(self):
        self.alpha = 0.25  # exploration parameter
        # Context: [knn_score, llm_score, heuristic_score]
        # Actions: accept (1) or reject (0)
        # Reward: user feedback (1=useful, 0=not useful)

    def decide(self, signals: CurationSignals) -> tuple[bool, float]:
        """Returns (accept: bool, confidence: float)"""
        context = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        # LinUCB upper confidence bound
        score = self.theta @ context + self.alpha * np.sqrt(context @ self.A_inv @ context)
        return score > 0.5, float(score)

    def update(self, signals: CurationSignals, reward: float):
        """Update after user feedback."""
        context = np.array([signals.knn_score, signals.llm_score, signals.heuristic_score])
        # Standard LinUCB update
        self.A += np.outer(context, context)
        self.b += reward * context
        self.A_inv = np.linalg.inv(self.A)
        self.theta = self.A_inv @ self.b
```

### 6.4 Self-Feedback Verification (OpenScholar pattern)

After curation, verify quality before finalizing:

```python
async def self_feedback_verify(curated: list[CuratedPaper], query: str) -> list[CuratedPaper]:
    """Check coverage and relevance of curated set."""
    prompt = f"""Review these {len(curated)} curated papers for the query: "{query}"

    Evaluate:
    1. Coverage: are major sub-topics represented?
    2. Relevance: are all papers actually relevant?
    3. Gaps: what important aspects are missing?

    Papers:
    {format_paper_list(curated)}

    Return JSON: {{"gaps": ["str"], "irrelevant_ids": ["str"], "coverage_score": 0-1}}"""

    feedback = await llm.generate(prompt)
    parsed = json.loads(feedback.text)

    # Remove false positives
    curated = [p for p in curated if p.candidate.paper_id not in parsed["irrelevant_ids"]]

    # If coverage < 0.7, trigger additional discovery for identified gaps
    if parsed["coverage_score"] < 0.7:
        for gap in parsed["gaps"]:
            additional = await discover(since=date.today() - timedelta(days=90), query=gap)
            curated.extend(await curate_batch(additional))

    return curated
```

## 7. MCP Server

Exposes the full pipeline as MCP tools for any agent in the SAGE ecosystem.

### Implementation: FastMCP (stdio transport)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("sage-discover")

@mcp.tool()
async def discover_papers(
    query: str,
    domains: list[str] | None = None,
    since: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Discover papers from arXiv + Semantic Scholar + HuggingFace.

    Args:
        query: Search query string
        domains: Filter to specific domains (marl, cognitive_architectures, etc.)
        since: ISO date string (YYYY-MM-DD), defaults to 7 days ago
        max_results: Maximum papers to return
    """

@mcp.tool()
async def curate_papers(paper_ids: list[str]) -> list[dict]:
    """Score and filter papers using adaptive curation (kNN + LLM + bandit)."""

@mcp.tool()
async def query_knowledge(
    question: str,
    top_k: int = 10,
    domain: str | None = None,
) -> str:
    """RAG query over the local knowledge store.
    Returns grounded answer with citations."""

@mcp.tool()
async def explore_frontier(
    domain: str | None = None,
    generations: int = 5,
) -> dict:
    """MAP-Elites exploration of the research frontier.
    Returns coverage report with under-explored regions."""

@mcp.tool()
async def verify_claims(paper_id: str) -> dict:
    """Extract claims from a paper and verify logical consistency via SMT.
    Returns claims, relations, and any detected contradictions."""
```

### CLI integration

```bash
# Run as MCP server
python -m discover.mcp

# Run existing pipeline modes (unchanged interface)
python -m discover.pipeline --mode nightly
python -m discover.pipeline --mode on-demand --query "multi-agent RL"
```

## 8. Enhanced Discovery Pipeline

### 8.1 PDF Extraction (Docling)

```python
from docling.document_converter import DocumentConverter

async def extract_full_text(pdf_path: Path) -> dict:
    """Extract structured content from PDF using Docling."""
    converter = DocumentConverter()
    result = await asyncio.to_thread(converter.convert, str(pdf_path))
    doc = result.document

    return {
        "full_text": doc.export_to_markdown(),
        "sections": extract_sections(doc),  # intro, methodology, results, conclusion
        "tables": [t.export_to_markdown() for t in doc.tables],
        "figures": [f.caption for f in doc.figures if f.caption],
    }
```

### 8.2 Citation Graph Builder

```python
import networkx as nx
from semanticscholar import SemanticScholar

class CitationGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.s2 = SemanticScholar()

    async def build_from_papers(self, paper_ids: list[str], depth: int = 1):
        """Build citation subgraph from seed papers."""
        for paper_id in paper_ids:
            await self._add_paper_and_citations(paper_id, depth)

    async def _add_paper_and_citations(self, paper_id: str, depth: int):
        if depth <= 0 or paper_id in self.graph:
            return

        paper = await asyncio.to_thread(
            self.s2.get_paper, paper_id,
            fields=["title", "year", "citationCount", "references", "citations"]
        )

        self.graph.add_node(paper_id, title=paper.title, year=paper.year,
                           citation_count=paper.citationCount)

        # Add citation edges (paper -> references, citations -> paper)
        for ref in (paper.references or [])[:20]:  # cap to avoid explosion
            if ref.paperId:
                self.graph.add_edge(paper_id, ref.paperId, relation="cites")

        for cit in (paper.citations or [])[:20]:
            if cit.paperId:
                self.graph.add_edge(cit.paperId, paper_id, relation="cites")

    def pagerank(self) -> dict[str, float]:
        return nx.pagerank(self.graph, alpha=0.85)

    def communities(self) -> list[set]:
        return list(nx.community.louvain_communities(self.graph.to_undirected()))

    def bridges(self) -> dict[str, float]:
        return nx.betweenness_centrality(self.graph)
```

### 8.3 Embedding Pipeline

```python
from sentence_transformers import SentenceTransformer, SparseEncoder

class EmbeddingPipeline:
    def __init__(self):
        self.specter2 = SentenceTransformer("allenai/specter2_base")
        self.splade = SparseEncoder("naver/splade-cocondenser-ensembledistil")
        self.reranker = None  # lazy-loaded

    def embed_paper(self, title: str, abstract: str) -> tuple[np.ndarray, SparseVector]:
        text = f"{title}. {abstract}"
        dense = self.specter2.encode(text)
        sparse = self.splade.encode(text)
        return dense, sparse

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        pairs = [(query, c["title"] + ". " + c["abstract"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]
```

## 9. Model Stack

| Stage | Model | Params | Already in SAGE | Load strategy |
|-------|-------|--------|-----------------|---------------|
| Paper embedding | `allenai/specter2_base` | 110M | No | Eager at boot |
| Sparse search | `naver/splade-cocondenser-ensembledistil` | ~110M | No | Eager at boot |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | No | Lazy on first query |
| Claim extraction | `google/flan-t5-small` | 77M | No | Lazy on first claim task |
| Claim verification | OxiZ SmtVerifier | Rust native | Yes (30+ tests) | Via sage_core |
| kNN curation | arctic-embed-m or SPECTER2 | 109M | Yes (92% acc) | Shares SPECTER2 |
| PDF extraction | Docling | Python lib | No | Lazy on first PDF |
| Citation graph | NetworkX | Python lib | No | Always available |
| Vector store | Qdrant (local mode) | N/A | No | Eager at boot |

**Total unique model memory**: ~440M params (~1.8GB fp32, ~0.9GB fp16).
Fits comfortably on CPU or any GPU >= 4GB.

## 10. Modules (file plan)

### New files

```
src/discover/
  store.py            — Qdrant wrapper (KnowledgeStore class)
  embeddings.py       — EmbeddingPipeline (SPECTER2 + SPLADE + reranker)
  extractor.py        — Docling PDF extraction + section parser
  claim_graph.py      — ClaimExtractor + RelationClassifier + SMT verification
  citation_graph.py   — CitationGraphBuilder (NetworkX + S2 API)
  frontier.py         — MapElitesFrontierExplorer
  adaptive_curator.py — KnnCurator + CurationBandit + self-feedback
  mcp.py              — FastMCP server (5 tools)
  rag.py              — Hybrid search + RAG answer generation
```

### Modified files

```
src/discover/
  pipeline.py         — Wire new components, keep existing interface
  discovery.py        — Add S2 recommendations API, citation data
  curator.py          — Delegate to adaptive_curator, keep as legacy fallback
  ingestion.py        — Upsert to Qdrant instead of ExoCortex upload
  __init__.py         — Export new public API
  __main__.py         — Add MCP server mode
```

### Deleted files

```
src/discover/
  knowledge.py        — NotebookLMBridge (replaced by Qdrant RAG)
  workflow.py          — DiscoverWorkflow (replaced by frontier.py)
  researcher.py        — ResearchAgent (absorbed into frontier.py)
```

### Moved/absorbed

```
mcp_gateway.py        — Z3 SQL verification tool absorbed into mcp.py
```

## 11. Dependencies (pyproject.toml additions)

```toml
[project]
dependencies = [
    "ygn-sage>=0.1.0",
    "arxiv>=2.1",
    "semanticscholar>=0.8",
    # New
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
```

## 12. Testing Strategy

### Unit tests (per module, all mocked)

| Module | Tests | Key scenarios |
|--------|-------|---------------|
| store.py | 8 | CRUD, hybrid search, RRF fusion, filter by domain |
| embeddings.py | 5 | SPECTER2 encode, SPLADE encode, rerank, lazy loading |
| extractor.py | 5 | PDF extraction, section parsing, fallback to abstract |
| claim_graph.py | 10 | Claim extraction, relation classification, SMT verification (consistent/contradictory/unknown), PageRank credit |
| citation_graph.py | 6 | Build graph, PageRank, communities, bridges, depth limiting |
| frontier.py | 8 | Descriptor computation, archive insert, exploration query gen, coverage |
| adaptive_curator.py | 8 | kNN scoring, bandit fusion, update, self-feedback loop |
| mcp.py | 5 | Each tool, error handling |
| rag.py | 5 | Hybrid search, RAG answer, citation formatting |

### Integration tests

| Test | What it validates |
|------|-------------------|
| Nightly pipeline E2E | discover -> curate -> extract -> embed -> ingest -> claim -> graph |
| On-demand query E2E | query -> hybrid search -> rerank -> RAG answer |
| Frontier exploration E2E | seed -> MAP-Elites loop -> coverage report |
| SMT claim verification E2E | paper -> claims -> relations -> OxiZ check |
| MCP server E2E | start server, call each tool, verify responses |

### Benchmark

```bash
# Knowledge store retrieval quality
python -m discover.bench --type retrieval --queries 50

# Claim extraction accuracy (vs manual annotations)
python -m discover.bench --type claims --papers 20

# Frontier coverage (MAP-Elites grid fill %)
python -m discover.bench --type frontier --generations 10
```

## 13. Deletions

| What | Why |
|------|-----|
| `knowledge.py` (NotebookLMBridge) | Stub; SDK never in deps. Replaced by Qdrant RAG |
| `workflow.py` (DiscoverWorkflow) | Fake eBPF evolution. Replaced by real MAP-Elites frontier |
| `researcher.py` (ResearchAgent) | Data-only class absorbed into frontier.py |
| `mcp_gateway.py` (standalone) | Z3 SQL tool moved into unified MCP server |
| ExoCortex as primary backend | Replaced by Qdrant local. ExoCortex available as optional sync |
| Manifest JSON (`~/.sage/manifest.json`) | Replaced by Qdrant dedup (point ID = paper_id) |
| Score threshold 42.0 | Replaced by bandit-learned threshold |

## 14. Migration Path

### ExoCortex sync (optional)

```python
class ExoCortexSync:
    """Optional: sync Qdrant papers to Google GenAI File Search."""

    async def sync(self, store: KnowledgeStore, exocortex):
        papers = store.get_all_papers()
        for paper in papers:
            if not paper.payload.get("exocortex_synced"):
                await exocortex.upload(paper.pdf_path, paper.title)
                store.update_payload(paper.id, {"exocortex_synced": True})
```

### Existing data

Papers already in ExoCortex remain accessible. New pipeline writes to Qdrant.
Migration script can backfill Qdrant from ExoCortex manifest if needed.

## 15. Success Criteria

1. **Hybrid search quality**: SPECTER2+SPLADE retrieval precision@10 >= 0.8 on domain queries
2. **Claim extraction**: >= 70% F1 on extracting claims from paper abstracts
3. **SMT verification**: correctly identify >= 90% of synthetically injected contradictions
4. **MAP-Elites coverage**: >= 60% grid fill after 10 generations of exploration
5. **Adaptive curation**: bandit-fused scoring achieves higher user satisfaction than LLM-only after 50 feedback rounds
6. **All 52 existing tests still pass** (with updated mocking)
7. **New test count**: >= 60 new tests across new modules
8. **MCP server**: all 5 tools callable and returning valid responses

## 16. Research References

### SOTA Systems Benchmarked
- PaperQA2 (FutureHouse, arXiv 2409.13740) — superhuman RAG, contradiction detection
- OpenScholar (Allen AI, Nature Feb 2026, arXiv 2411.14199) — 45M papers, self-feedback
- STORM (Stanford, arXiv 2402.14207) — multi-perspective, mind map, 28K stars
- AI Scientist v2 (Sakana, arXiv 2504.08066) — end-to-end research, Nature 2025
- PaSa (arXiv 2501.10120) — RL-trained search, +37.78% vs Google Scholar
- OpenResearcher (TIGER, arXiv 2603.20278) — local corpus, MoE 30B
- ArXiv Paper Curator (GitHub 5.3K stars) — Airflow + Docling + LangGraph

### Key Papers
- ClaimFlow (arXiv 2603.16073) — claim relation taxonomy (supports/extends/refutes)
- NLP-AKG (arXiv 2502.14192) — 620K entities from 60K papers via few-shot LLM
- GraphRAG (arXiv 2507.03226) — hybrid retrieval 15% over vector-only
- A-RAG (arXiv 2602.03442) — agentic retrieval strategy selection
- SurveyG (arXiv 2510.07733) — hierarchical citation graph (Foundation/Development/Frontier)
- SciNetBench (arXiv 2601.03260) — relation-aware retrieval +23% quality
- MACC (arXiv 2603.03780) — multi-agent blackboard for scientific exploration
- Deep Research Survey (arXiv 2508.12752) — 4-stage pipeline taxonomy
- Agentic Science Survey (arXiv 2508.14111) — 5 core capabilities for scientific agency
- SciFact (arXiv 2004.14974) — claim verification benchmark
- RewardFlow (OpenReview 5oGJbM5u86) — PageRank-based credit propagation

### Models
- SPECTER2 (Allen AI, arXiv 2004.07180v2) — citation-informed paper embeddings
- SciNCL (arXiv 2202.06671) — neighborhood contrastive learning
- SciBERT (arXiv 1903.10676) — scientific language model, 1.14M papers
