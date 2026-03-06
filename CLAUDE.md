# YGN-SAGE - CLAUDE.md

## Project Overview
YGN-SAGE (Yann's Generative Neural Self-Adaptive Generation Engine) is a next-generation
Agent Development Kit built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Architecture
- `sage-core/` - Rust orchestrator (PyO3 bindings to Python)
- `sage-python/` - Python SDK for building agents
- `sage-discover/` - Flagship Research & Discovery agent + Knowledge Pipeline
- `ui/` - Control Dashboard (FastAPI + Tailwind + WebSocket)
- `docs/plans/` - Architecture and implementation plans
- `Researches/` - Research papers (OpenSage, AlphaEvolve, PSRO, etc.)
- `.github/workflows/ci.yml` - CI pipeline (Rust + Python sage + Python discover)

### Key Python Modules (sage-python/src/sage/)
- `boot.py` - Boot sequence, wires all pillars + ExoCortex into `AgentSystem`
- `agent_loop.py` - Structured perceive->think->act->learn runtime with S2 AVR loop, Z3-aligned S3 prompts, AgentEvent schema, independent S2/S3 retry budgets
- `agent_pool.py` - Dynamic sub-agent pool (create/run/ensemble)
- `llm/router.py` - Model Router with 7 tiers, data-driven lookup from TOML + env vars
- `llm/config_loader.py` - TOML config loader + env var resolution (SAGE_MODEL_<TIER>)
- `llm/google.py` - Google Gemini provider + File Search grounding (`file_search_store_names`)
- `llm/codex.py` - OpenAI Codex CLI provider (+ Google fallback)
- `strategy/metacognition.py` - Stanovich S1/S2/S3 tripartite routing + CGRS self-braking
- `topology/evo_topology.py` - MAP-Elites evolutionary topology search
- `topology/kg_rlvr.py` - Process Reward Model (Z3 DSL: `assert bounds/loop/arithmetic/invariant`)
- `evolution/engine.py` - Evolution engine with DGM context injection (5 SAMPO actions)
- `evolution/llm_mutator.py` - LLM-driven code mutation with DGM Directive prompt section
- `memory/memory_agent.py` - Autonomous entity extraction (heuristic or LLM)
- `memory/compressor.py` - MEM1 per-step internal state + pressure-triggered compression
- `memory/episodic.py` - In-memory episodic store with CRUD + keyword search
- `memory/remote_rag.py` - ExoCortex (Google GenAI File Search API) + `query()` method + RagCacheFallback
- `tools/memory_tools.py` - 7 AgeMem tools (3 STM + 4 LTM) exposed to agent
- `tools/exocortex_tools.py` - `search_exocortex` + `refresh_knowledge` agent tools

### Key Discovery Modules (sage-discover/src/discover/)
- `discovery.py` - Scan arXiv, Semantic Scholar, HuggingFace across 5 domains (MARL, cog arch, formal verif, evo comp, memory)
- `curator.py` - Two-stage curation: heuristic filter + LLM scoring (threshold >= 6)
- `ingestion.py` - ExoCortex upload with `custom_metadata` + JSON manifest dedup (`~/.sage/manifest.json`)
- `migration.py` - One-time NotebookLM markdown bootstrap + arXiv ID re-discovery
- `pipeline.py` - Orchestrator: `run_pipeline(mode="nightly"|"on-demand"|"migrate")`
- `__main__.py` - CLI: `python -m discover.pipeline --mode nightly`

### Key Rust Modules (sage-core/src/)
- `memory/mod.rs` - Arrow-backed working memory (SIMD/AVX-512) + S-MMU paging
- `memory/rag_cache.rs` - FIFO+TTL cache for File Search results (DashMap + atomic counters)
- `sandbox/ebpf.rs` - eBPF executor (solana_rbpf) + SnapBPF (CoW memory snapshots)
- `sandbox/wasm.rs` - Wasm sandbox (wasmtime)
- `z3/` - Z3 formal verification bindings

### Dashboard (ui/)
- `ui/app.py` - FastAPI backend: REST API + WebSocket event streaming
- `ui/static/index.html` - Single-file production dashboard (Tailwind + vanilla JS)
- Endpoints: `GET /`, `GET /api/state`, `POST /api/task`, `POST /api/stop`, `POST /api/reset`, `GET /api/providers`, `WS /ws`

## Development Commands

### Python SDK
```bash
cd sage-python
pip install -e ".[all,dev]"    # Install in dev mode with all providers
python -m pytest tests/ -v     # Run tests (200 passed, 1 skipped)
ruff check src/                 # Lint
mypy src/                       # Type check
```

### Dashboard
```bash
# From repo root:
python ui/app.py                # Start dashboard on http://localhost:8000
```

### Rust Core
```bash
cd sage-core
cargo build                    # Build Rust core
cargo test                     # Run Rust tests (38 passing)
cargo clippy                   # Lint Rust code
maturin develop                # Build + install Python bindings
```

### Discovery Pipeline
```bash
cd sage-discover
pip install -e .               # Install sage-discover
python -m pytest tests/ -v     # Run tests (45 passed)
python -m discover.pipeline --mode nightly -v  # Run nightly pipeline
```

### Full Build
```bash
cd sage-core && maturin develop && cd ..
cd sage-python && pip install -e ".[all,dev]"
cd sage-discover && pip install -e .
```

## LLM Configuration

### Active Models (March 2026, verified working)
| Tier | Model ID | Provider | Notes |
|------|----------|----------|-------|
| codex | `gpt-5.3-codex` | Codex CLI | Default, SOTA coding |
| codex_max | `gpt-5.2` | Codex CLI | Most powerful reasoning, xhigh effort |
| reasoner | `gemini-3.1-pro-preview` | Google API | Complex evaluation |
| mutator | `gemini-3-flash-preview` | Google API | Code mutation |
| fast | `gemini-3.1-flash-lite-preview` | Google API | Low-latency |
| budget | `gemini-2.5-flash-lite` | Google API | Cheapest |
| fallback | `gemini-2.5-flash` | Google API | If 3.x unavailable |

### Required Environment Variables
```bash
export GOOGLE_API_KEY="..."                  # Required for Gemini models
export SAGE_EXOCORTEX_STORE="projects/..."   # Optional: File Search store for ExoCortex
export SAGE_MODEL_FAST="gemini-2.5-flash"    # Optional: override any tier model ID
# Codex CLI uses ChatGPT Pro account (codex login)
```

### Model Config Resolution
Model IDs resolved in order: env var `SAGE_MODEL_<TIER>` > `config/models.toml` > hardcoded defaults.
TOML searched in: `cwd/config/`, `sage-python/config/` (package), `~/.sage/`.

### Structured Output
- Codex: `--output-schema file.json` (requires `additionalProperties: false`)
- Gemini: `response_schema=PydanticModel.model_json_schema()`

## Tech Stack
- Rust 1.90+ (orchestrator, via PyO3) -- SnapBPF, RagCache, Arrow memory, eBPF
- Python 3.12+ (SDK, agents)
- OpenAI Codex CLI + gpt-5.3-codex (primary LLM)
- Google Gemini 3.x via `google-genai` SDK (secondary LLM, fallback, File Search)
- FastAPI + WebSocket (dashboard -- install via `pip install -e ".[ui]"`)
- Z3 Solver 4.16 (formal verification, S3)
- Apache Arrow / PyArrow (zero-copy memory compaction)
- Wasm (wasmtime) + eBPF (solana_rbpf) + Docker (multi-tier sandboxing)
- DashMap (Rust) -- lock-free FIFO+TTL RAG cache + CoW snapshots

## Memory System
- **MEM1 Internal State**: `MemoryCompressor.generate_internal_state()` runs every agent step, producing a rolling `<IS_t>` summary
- **Pressure Compression**: When `event_count >= threshold`, LLM summarizes old events and prunes the buffer
- **Episodic Memory**: In-memory keyword-search store with CRUD
- **ExoCortex**: Persistent RAG via Google GenAI File Search API (`memory/remote_rag.py`). **Passive grounding** in `_think()` (auto-injects `file_search_store_names`). **Active grounding** via `search_exocortex` tool. `query()` method for synchronous tool use.
- **Knowledge Pipeline**: `sage-discover` auto-refreshes ExoCortex: discovery (arXiv/S2/HF) -> curator (heuristic+LLM) -> ingestion (upload+manifest). `refresh_knowledge` agent tool triggers on-demand.
- **RAG Cache**: FIFO+TTL cache (`sage_core.RagCache` in Rust, `RagCacheFallback` in Python). Factory: `get_rag_cache()`
- **9 Agent Tools**: 7 AgeMem (3 STM + 4 LTM) + `search_exocortex` + `refresh_knowledge`
- **S2 AVR Loop**: Act-Verify-Refine cycle -- sandbox execution, structured error feedback, retry budget (`S2_AVR_MAX_ITERATIONS=3`), escalation to S3

## Evolution System
- **DGM Context**: SAMPO solver chooses 1 of 5 strategic actions. Action + description + parent_score + generation passed to LLM mutator as `dgm_context` dict. Prompt includes "## DGM Directive" section
- **SnapBPF**: Rust CoW memory snapshots (`DashMap<String, Arc<Vec<u8>>>`). `snapshot()`, `restore()`, `delete()`, `count()` via PyO3
- **DGM_ACTION_DESCRIPTIONS**: Maps actions 0-4 to semantic directives (optimize perf, fix bugs, explore, constrain, simplify)

## Z3 Formal Verification (S3)
- S3 system prompt teaches Z3 DSL: `assert bounds(addr, limit)`, `assert loop(var)`, `assert arithmetic(expr, val)`, `assert invariant("pre", "post")`
- S3 retry prompt reinforces syntax when PRM score is low
- S2->S3 escalation prompt mentions all 4 assertion types
- `kg_rlvr.py` parses `<think>` blocks, scores each step via regex + Z3

## Observability
- **AgentEvent schema** (v1): Versioned dataclass emitted on every loop phase (PERCEIVE, THINK, ACT, LEARN)
- Fields: `type`, `step`, `timestamp`, `schema_version`, `latency_ms`, `cost_usd`, `tokens_est`, `model`, `system`, `routing_source`, `validation`, `meta`
- `_emit()` constructs `AgentEvent` and calls `on_event` callback (default: JSONL to `agent_stream.jsonl`)
- Dashboard detects `schema_version` and adapts display

## Key Design Principles
- AI-centered: agents create their own topology, tools, and memory
- Self-evolving: evolutionary pipeline with DGM strategic context improves all components
- Game-theoretic: PSRO-based strategy for multi-agent orchestration
- Multi-provider: Codex CLI (primary), Gemini (fallback), Mock (testing)
- Metacognitive: SOFAI S1/S2/S3 tripartite cognitive routing with CGRS self-braking
- Rust-accelerated: SnapBPF, RagCache, Arrow memory, eBPF -- Python fallbacks when Rust unavailable

## ExoCortex (replaces NotebookLM)

The ExoCortex module (`memory/remote_rag.py`) replaces the fragile `notebooklm-py` CLI bridge with the native Google GenAI File Search API.

```python
from sage.memory.remote_rag import ExoCortex

exo = ExoCortex()  # Reads SAGE_EXOCORTEX_STORE env var
tool = exo.get_file_search_tool()  # Returns types.Tool or None

# Or create a new store:
store_name = await exo.create_store("my-research")
await exo.upload("paper.pdf")
```

The ExoCortex is instantiated in `boot.py` and attached to `AgentLoop` as `loop.exocortex`.
`agent_loop.py` `_think()` automatically passes `file_search_store_names` to `generate()` when ExoCortex is available.

### Knowledge Pipeline (sage-discover)
Auto-refreshing research paper ingestion. Single store `ygn-sage-research` with `custom_metadata` for domain filtering.
```bash
python -m discover.pipeline --mode nightly           # Papers from yesterday
python -m discover.pipeline --mode on-demand --query "PSRO"  # Targeted search
python -m discover.pipeline --mode migrate           # Bootstrap from ~/.sage/migration/*.md
```
Pipeline: `discover()` -> `curate()` -> `ingest_all()` -> ExoCortex store.
Manifest dedup: `~/.sage/manifest.json`. PDFs cached in `~/.sage/papers/{domain}/`.

