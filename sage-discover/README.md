# sage-discover

Knowledge Pipeline for YGN-SAGE. Discovers, curates, and ingests research papers into the ExoCortex (Google GenAI File Search API) for persistent RAG grounding.

## Installation

```bash
cd sage-discover
pip install -e .
python -m pytest tests/ -v    # 52 tests
```

## Usage

```bash
python -m discover.pipeline --mode nightly              # Papers from yesterday
python -m discover.pipeline --mode on-demand --query "PSRO"  # Targeted search
python -m discover.pipeline --mode migrate              # Bootstrap from NotebookLM exports
```

## Source Modules (`src/discover/`)

### `pipeline.py` -- Orchestrator

Main pipeline entry point. Coordinates the discover-curate-ingest flow across three modes: `nightly` (daily cron), `on-demand` (keyword search), and `migrate` (bulk import from NotebookLM).

### `discovery.py` -- Paper Discovery

Searches multiple academic sources for relevant papers:
- arXiv API (preprints)
- Semantic Scholar (S2) API (citation graph, references)
- Hugging Face Hub (model cards, datasets)

### `curator.py` -- Curation and Ranking

Filters and ranks discovered papers by relevance to YGN-SAGE research topics. Deduplicates against already-ingested content.

### `ingestion.py` -- ExoCortex Ingestion

Uploads curated papers to the ExoCortex (Google GenAI File Search API) for indexing. Handles chunking, metadata attachment, and store management.

### `migration.py` -- NotebookLM Migration

Imports existing research from NotebookLM exports. Converts notebook format to the pipeline's internal representation and feeds into the ingestion stage.

### `knowledge.py` -- Knowledge Representation

Internal data model for research papers: title, abstract, authors, source, relevance score, and ingestion metadata.

### `researcher.py` -- Research Agent

Autonomous research agent that uses the pipeline components to explore topics and build knowledge.

### `model_watcher.py` -- Model Watcher

Monitors model registries (HuggingFace, provider APIs) for new model releases relevant to YGN-SAGE capabilities.

### `workflow.py` -- Workflow Definitions

Predefined research workflows combining discovery, curation, and ingestion steps.

## Tests

52 tests covering all pipeline stages:
- `test_pipeline.py` -- End-to-end pipeline orchestration
- `test_discovery.py`, `test_discover.py` -- Source API integration
- `test_curator.py` -- Ranking and dedup logic
- `test_ingestion.py` -- ExoCortex upload
- `test_migration.py` -- NotebookLM import
- `test_model_watcher.py` -- Model registry monitoring
