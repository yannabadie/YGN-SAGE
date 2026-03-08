# discover -- Source Modules

Core implementation of the YGN-SAGE knowledge pipeline. Each module handles a stage of the discover-curate-ingest flow.

## Module Index

### `pipeline.py` -- Pipeline Orchestrator

Entry point for all pipeline modes. Wires together discovery, curation, and ingestion stages. Supports three modes:
- `nightly` -- Discovers papers published since last run (daily cron job).
- `on-demand` -- Searches for papers matching a specific query string.
- `migrate` -- Bulk imports from NotebookLM exports or other external sources.

### `discovery.py` -- Multi-Source Paper Discovery

Queries academic APIs to find relevant research:
- **arXiv** -- Preprint search by category and keyword.
- **Semantic Scholar (S2)** -- Citation-aware search with reference traversal.
- **Hugging Face Hub** -- Model cards and dataset documentation.

Returns a list of candidate papers with metadata (title, abstract, authors, URL, source).

### `curator.py` -- Curation and Relevance Ranking

Scores and filters discovered papers against YGN-SAGE research topics. Removes duplicates (by DOI, arXiv ID, or title similarity). Outputs a ranked shortlist for ingestion.

### `ingestion.py` -- ExoCortex Ingestion

Uploads curated content to the ExoCortex store (Google GenAI File Search API). Handles document chunking, metadata tagging, and upload verification.

### `migration.py` -- NotebookLM Migration

Parses NotebookLM export format and converts entries into the pipeline's internal `Knowledge` objects for re-ingestion into ExoCortex.

### `knowledge.py` -- Knowledge Data Model

Defines the `Knowledge` dataclass: title, abstract, authors, source URL, relevance score, discovery timestamp, and ingestion status.

### `researcher.py` -- Autonomous Research Agent

Higher-level agent that chains discovery + curation to explore a research topic autonomously. Produces a structured research journal.

### `model_watcher.py` -- Model Release Monitor

Watches for new model releases on HuggingFace and provider APIs. Emits notifications when models relevant to YGN-SAGE tiers become available.

### `workflow.py` -- Predefined Workflows

Reusable workflow templates combining pipeline stages for common research patterns.

### `__main__.py` -- CLI Entry Point

Enables `python -m discover.pipeline` invocation. Parses CLI arguments (`--mode`, `--query`, `-v`).
