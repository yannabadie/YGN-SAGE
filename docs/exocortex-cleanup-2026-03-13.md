# ExoCortex Cleanup Report (2026-03-13)

## Summary

Store: `fileSearchStores/ygnsageresearch-wii7kwkqozrd` (ygn-sage-research)

| Metric | Before | After |
|--------|--------|-------|
| Total documents | 890 | 408 |
| Duplicates removed | -- | 458 |
| Noise removed | -- | 38 |
| Papers ingested | -- | 14 |
| Key papers verified | 2/14 | 14/14 |

## Noise Removed (38 documents, 19 unique)

All from `Discover_AI_Frontiers` pipeline -- web scrapes unrelated to project research:

- YouTube channel lists (6 docs): AI content creator directories, Reddit threads
- Aashish Jaini LinkedIn profile (2 docs)
- Flatlogic Hackathons 2024 event recap (2 docs)
- Medical AI Career Guide - Physician handbook appendix (2 docs)
- "Is the AI bubble about to pop" opinion piece (2 docs)
- Ignite Board tech innovation project approvals (2 docs)
- AI Tools Business Success 2026 guide (2 docs)
- AI Superintelligence Discovered Princeton clickbait (2 docs)
- code4ai GitHub MSc code repository (4 docs)
- YouTube search bug report (2 docs)
- AI content creator directory (2 docs)
- Free Video AI Agents biomedical (2 docs)
- The Intellify AI Agents Developers ad (2 docs)

Each was duplicated (likely from two pipeline runs). Deleted with `force=True` (documents contained indexed chunks).

## Duplicates Removed (458 documents)

The store had pervasive duplication from multiple ingestion runs:
- MEM1 paper: 18 copies
- Metacognition Language Models paper: 10 copies
- KG-MAS paper: 8 copies
- Most Technical_Implementation and Discover_AI_Frontiers entries: 2-6 copies each

After dedup: 408 unique documents remain.

## Papers Ingested (14 new)

### From arXiv (9 downloaded + uploaded)

| arXiv ID | Title | Size |
|----------|-------|------|
| 2602.16873 | AdaptOrch: Adaptive Orchestration for MAS | 1044 KB |
| 2406.18665 | RouteLLM: Learning to Route LLMs with Preference Data (ICLR 2025) | 790 KB |
| 2508.21141 | PILOT: Contextual Bandit LLM Routing with Budget | 1988 KB |
| 2601.07206 | LLMRouterBench: Benchmarking LLM Routing Methods | 4701 KB |
| 2601.12996 | OFA-MAS: MoE Graph Generative for Universal MAS Topology (WWW 2026) | 1466 KB |
| 2505.12601 | kNN Routing for LLMs: Embedding-based Router Selection | 593 KB |
| 2603.04445 | Dynamic Model Routing and Cascading for LLMs: A Survey | 4040 KB |
| 2505.22467 | Topology Structure Learning for Multi-Agent Systems | 500 KB |
| 2410.10347 | A Unified Approach to Routing and Cascading for LLMs (ETH-SRI, ICLR 2025) | 811 KB |

### From local Researches/ folder (5 uploaded)

| File | Title | Size |
|------|-------|------|
| MAS-FACTORY-2603.06007v1.pdf | MASFactory: Graph-centric Framework for MAS with Vibe Graphing | 772 KB |
| 2603.08068v1.pdf | In-Context RL for Tool Use in LLMs | 1860 KB |
| 2603.08647v1.pdf | Grow, Don't Overwrite: Fine-tuning Without Forgetting | 3426 KB |
| 2602.16891v1.pdf | OpenSage: Self-programming Agent Generation Engine | 1094 KB |
| 2504.01990v2.pdf | Advances and Challenges in Foundation Agents (396 pages) | 40 MB |

## Final Store Composition (408 documents)

| Category | Count | Description |
|----------|-------|-------------|
| Other (papers + sources) | 207 | arXiv papers, research sources |
| Core_Research_MARL | 50 | MARL, game theory, PSRO papers |
| MetaScaffold_Core | 48 | Metacognition, S1/S2 reasoning |
| Technical_Implementation | 47 | Memory, tools, agent frameworks |
| Discover_AI_Frontiers | 40 | Curated AI frontier content |
| arXiv papers (labeled) | 16 | Newly ingested with arXiv labels |
| **TOTAL** | **408** | |

## Key Papers Coverage (CLAUDE.md References)

All papers referenced in CLAUDE.md are now present:

- [x] AlphaEvolve (2506.13131) - was already present
- [x] PSRO variants - was already present (5+ papers)
- [x] AdaptOrch (2602.16873) - NEW
- [x] RouteLLM (2406.18665) - NEW
- [x] PILOT (2508.21141) - NEW
- [x] LLMRouterBench (2601.07206) - NEW
- [x] OFA-MAS (2601.12996) - NEW
- [x] kNN Routing (2505.12601) - NEW
- [x] Routing Survey (2603.04445) - NEW
- [x] Cascade Routing ETH-SRI (2410.10347) - NEW
- [x] MASFactory (2603.06007) - NEW
- [x] Topology Structure Learning (2505.22467) - NEW
- [x] MAP-Elites (1504.04909) - was already present
- [x] Cognitive Architectures for Language Agents (2309.02427) - was already present

## Tooling Created

### `scripts/exocortex_ingest.py`

Reusable CLI for ExoCortex management:

```bash
# Single paper from arXiv
python scripts/exocortex_ingest.py --arxiv 2406.18665 --title "RouteLLM"

# Local PDF
python scripts/exocortex_ingest.py --pdf path/to/paper.pdf --title "Paper Title"

# Batch from manifest
python scripts/exocortex_ingest.py --manifest scripts/missing_papers.json

# Store management
python scripts/exocortex_ingest.py --list      # List all documents
python scripts/exocortex_ingest.py --count     # Count documents
python scripts/exocortex_ingest.py --delete RESOURCE_NAME  # Delete (force=True)
```

### `scripts/missing_papers.json`

Manifest of all 13 papers ingested in this session (reusable format).

## API Notes

- `client.file_search_stores.documents.list(parent=store_id)` - list all documents
- `client.file_search_stores.documents.delete(name=..., config=DeleteDocumentConfig(force=True))` - delete with chunks
- `client.file_search_stores.upload_to_file_search_store(file=path, ...)` - returns an Operation (poll until done)
- Upload accepts PDFs up to at least 40 MB / 396 pages
- SSL verify=False required on this machine (corporate proxy)
