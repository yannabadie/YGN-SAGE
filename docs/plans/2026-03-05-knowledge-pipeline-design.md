# SOTA Knowledge Pipeline — Design Document

**Date**: 2026-03-05
**Author**: Yann Abadie + Claude (brainstorming skill)
**Status**: Approved

## Goal

Build an auto-refreshing knowledge pipeline that discovers, curates, and ingests SOTA research papers into ExoCortex (Google GenAI File Search), providing YGN-SAGE agents with always-current scientific grounding across 5 domains.

## Architecture

Pipeline modulaire: 4 independent modules orchestrated by a central `pipeline.py`. Papers flow through discovery → curation → ingestion. ExoCortex is wired end-to-end in `agent_loop.py` for passive grounding and as an explicit agent tool for active research.

**Account**: Google AI Ultra (extended quotas, no file/storage limits concern).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frequency | Nightly + on-demand | Balances freshness vs. cost |
| Store organization | Single store `ygn-sage-research` with `custom_metadata` | Simpler ops, metadata_filter for domain routing |
| Dependencies | `arxiv` + `semanticscholar` (no `paper-qa` in v1) | PaperQA2 too heavy for nightly; reserved for future |
| Pipeline pattern | Sequential: discover → curate → ingest | Simple, debuggable, each stage independently testable |
| NotebookLM migration | Q&A extraction + re-discovery of real papers | Day 1 bootstrap, then nightly takes over |

---

## Section 1: ExoCortex End-to-End Wiring

### Current State

- `ExoCortex` instantiated in `boot.py` (line 152), stored on `loop.exocortex` (line 178)
- `google.py` `generate()` accepts `file_search_store_names` parameter
- **GAP**: `agent_loop.py` `_think()` never passes `file_search_store_names` to `generate()`

### Design

In `agent_loop.py::_think()`:
- Check `self.exocortex.is_available`
- If yes: `store_names = [self.exocortex.store_name]`
- Pass `file_search_store_names=store_names` to `self._llm.generate()`
- Google provider injects `types.Tool(file_search=...)` automatically

**Scope**: Only `_think()` gets ExoCortex grounding. Graceful degradation if `SAGE_EXOCORTEX_STORE` not set.

### New Agent Tool: `search_exocortex`

```python
@tool
def search_exocortex(query: str, domain: str | None = None) -> str:
    """Search the ExoCortex knowledge store for research papers and insights."""
```

Two paths to ExoCortex:
1. **Passive** — every `_think()` call is automatically grounded
2. **Active** — agent explicitly calls `search_exocortex` for targeted research

---

## Section 2: Discovery Module

**File**: `sage-discover/src/discover/discovery.py`

### Target Domains

```python
DOMAINS = {
    "marl": {
        "arxiv_categories": ["cs.MA", "cs.AI", "cs.GT"],
        "keywords": ["multi-agent reinforcement learning", "PSRO", "Nash equilibrium"],
    },
    "cognitive_architectures": {
        "arxiv_categories": ["cs.AI", "cs.CL"],
        "keywords": ["cognitive architecture", "metacognition", "SOFAI", "dual process"],
    },
    "formal_verification": {
        "arxiv_categories": ["cs.LO", "cs.SE"],
        "keywords": ["SMT solver", "Z3", "formal verification", "program synthesis"],
    },
    "evolutionary_computation": {
        "arxiv_categories": ["cs.NE", "cs.AI"],
        "keywords": ["MAP-Elites", "quality diversity", "evolutionary strategy", "LLM mutation"],
    },
    "memory_systems": {
        "arxiv_categories": ["cs.AI", "cs.CL"],
        "keywords": ["episodic memory", "working memory", "RAG", "retrieval augmented"],
    },
}
```

### Three Sources

| Source | Method | Rate Limit | Freshness |
|--------|--------|------------|-----------|
| arXiv | `arxiv.Search(query, sort_by=SubmittedDate)` | 3s between calls | Real-time |
| Semantic Scholar | `sse.search_paper(query, year=">2025", bulk=True)` | 1 req/s (no key) / 10 req/s (key) | ~24h delay |
| HuggingFace | `HfApi().list_papers(query)` | No documented limit | Daily papers feed |

### Output

```python
@dataclass
class PaperCandidate:
    paper_id: str          # arXiv ID or S2 corpus ID
    title: str
    authors: list[str]
    abstract: str
    source: str            # "arxiv" | "s2" | "hf"
    domain: str            # key from DOMAINS
    published: date
    pdf_url: str | None
    citation_count: int    # 0 for arXiv (no citation data)
```

### Deduplication

By normalized title (lowercase, strip punctuation) across all 3 sources. Same paper found on arXiv + S2 = merge, keep highest citation count.

---

## Section 3: Curator Module

**File**: `sage-discover/src/discover/curator.py`

### Two-Stage Curation

**Stage 1 — Heuristic filter** (zero cost):
- Reject if `len(abstract) < 100` (placeholder/withdrawn)
- Reject if published > 90 days ago AND `citation_count == 0`
- Reject if title matches blocklist patterns (surveys-of-surveys, correction notices)
- Boost if: cited by >10, multiple domains matched, HF daily_papers overlap

**Stage 2 — LLM scoring** (Gemini Flash):
```
CURATION_PROMPT = """Rate this paper's relevance to YGN-SAGE (0-10).
YGN-SAGE builds: multi-agent systems, evolutionary code generation,
metacognitive routing, formal verification, memory architectures.

Paper: {title}
Abstract: {abstract}

Return JSON: {"score": int, "reason": str, "key_insights": list[str]}"""
```

- Threshold: score >= 6 passes to ingestion
- Batch: up to 20 papers per LLM call
- Cost: ~$0.05/night

### Output

```python
@dataclass
class CuratedPaper:
    candidate: PaperCandidate
    relevance_score: int       # 0-10
    reason: str
    key_insights: list[str]    # 2-5 bullet points
    pdf_path: Path | None      # Downloaded PDF (if score >= 6)
```

PDF storage: `~/.sage/papers/{domain}/{paper_id}.pdf`

---

## Section 4: Ingestion Module

**File**: `sage-discover/src/discover/ingestion.py`

### Upload with custom_metadata

```python
async def ingest(paper: CuratedPaper, exocortex: ExoCortex):
    client = genai.Client(api_key=api_key)
    operation = client.file_search_stores.upload_to_file_search_store(
        file=str(paper.pdf_path),
        file_search_store_name=exocortex.store_name,
        config={
            "display_name": paper.candidate.title,
            "custom_metadata": {
                "domain": paper.candidate.domain,
                "paper_id": paper.candidate.paper_id,
                "source": paper.candidate.source,
                "published": paper.candidate.published.isoformat(),
                "relevance_score": str(paper.relevance_score),
                "ingested_at": datetime.utcnow().isoformat(),
            },
        },
    )
```

### Dedup at Ingestion

Local manifest (`~/.sage/manifest.json`):
```json
{
    "store_name": "file_search_stores/abc123",
    "papers": {
        "2503.01234": {
            "title": "...",
            "domain": "marl",
            "ingested_at": "2026-03-05T...",
            "file_resource": "file_search_stores/abc123/files/xyz"
        }
    }
}
```

Skip upload if `paper_id` already in manifest.

### Error Handling

Upload failures logged + skipped, retry on next nightly run. No crash on transient Google API errors.

---

## Section 5: Migration + Orchestration

### 5a. Migration Module

**File**: `sage-discover/src/discover/migration.py`

Three NotebookLM notebooks to migrate:

| Notebook | Strategy |
|----------|----------|
| Technical (ExoCortex) | Q&A extraction -> markdown syntheses |
| YGN | Q&A extraction -> markdown syntheses |
| Discover AI | Q&A extraction + re-discovery via arXiv IDs |

Migration flow:
1. Manual: Export Q&A pairs from NotebookLM (10-15 key questions per notebook)
2. Save as markdown: `~/.sage/migration/{notebook_name}.md`
3. Auto: `migration.py` uploads markdowns to ExoCortex with `custom_metadata: {"source": "notebooklm_migration"}`

Re-discovery (Discover AI only): Extract paper titles/arXiv IDs from export, feed to `discovery.py` as seed queries.

One-shot: Migration runs once, nightly pipeline takes over.

### 5b. Orchestrator

**File**: `sage-discover/src/discover/pipeline.py`

```python
async def run_pipeline(
    mode: str = "nightly",
    query: str | None = None,
    since: date | None = None,
    domains: list[str] | None = None,
):
    exocortex = ExoCortex(store_name=os.environ["SAGE_EXOCORTEX_STORE"])
    if mode == "migrate":
        await migrate_notebooks(exocortex)
        return
    candidates = await discover(since=since, query=query, domains=domains)
    curated = await curate(candidates)
    ingested = await ingest_all(curated, exocortex)
    return PipelineReport(discovered=len(candidates), curated=len(curated), ingested=len(ingested))
```

CLI entry point:
```bash
python -m discover.pipeline --mode nightly
python -m discover.pipeline --mode on-demand --query "attention mechanism pruning"
python -m discover.pipeline --mode migrate
```

### Agent Tool: `refresh_knowledge`

```python
@tool
def refresh_knowledge(query: str | None = None, domain: str | None = None) -> str:
    """Trigger on-demand knowledge discovery and ingestion."""
```

---

## Architecture Diagram

```
                    +---------------+
                    |  Nightly      |     +--------------+
                    |  Cron/Manual  |     |  Agent Tool   |
                    +-------+-------+     |  refresh_     |
                            |             |  knowledge    |
                            v             +------+-------+
                   +----------------+            |
                   |  pipeline.py   |<-----------+
                   |  (orchestrator)|
                   +--+-----+----+-+
          +-----------+     |    +------------+
          v                 v                 v
   +-------------+  +----------+  +--------------+
   | discovery.py|  |curator.py|  |ingestion.py  |
   | arXiv+S2+HF|  |Heuristic |  |ExoCortex     |
   +-------------+  |+LLM score|  |Upload+Meta   |
                    +----------+  +------+-------+
                                         |
                                         v
                               +------------------+
                               |  Google GenAI     |
                               |  File Search Store|
                               |  ygn-sage-research|
                               +--------+---------+
                                        |
                          +-------------+
                          v             v
                   +-----------+  +--------------+
                   |agent_loop |  |search_exo-   |
                   |_think()   |  |cortex tool   |
                   |(passive)  |  |(active)      |
                   +-----------+  +--------------+
```

## New Dependencies

In `sage-discover/pyproject.toml`:
- `arxiv` — arXiv API client
- `semanticscholar` — Semantic Scholar API client

## Cost Estimate (Google AI Ultra)

- Storage: included in Ultra
- Nightly LLM curation: ~$0.05/night (Gemini Flash)
- File Search queries: included in Ultra
