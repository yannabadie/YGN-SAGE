# memory

4-tier memory system for YGN-SAGE agents. Each tier trades latency for persistence and capacity.

| Tier | Module | Backend | Persistence |
|------|--------|---------|-------------|
| 0 -- Working (STM) | `working.py` | Rust Arrow / Python mock | Per-session |
| 1 -- Episodic | `episodic.py` | SQLite (aiosqlite) | Cross-session |
| 2 -- Semantic | `semantic.py` | Entity-relation graph + SQLite | Cross-session |
| 3 -- ExoCortex | `remote_rag.py` | Google GenAI File Search API | Permanent |

## Modules

### `base.py` -- MemoryStore Protocol

Abstract `MemoryStore` protocol defining `store(key, content, metadata)` and `search(query, top_k)`.

### `working.py` -- WorkingMemory (Tier 0)

Short-term, per-agent execution memory. Delegates to `sage_core` Rust Arrow buffer when compiled; falls back to a pure-Python mock with a loud warning otherwise. The mock returns dummy values for Arrow/S-MMU operations.

- **Key exports**: `WorkingMemory`, `_has_rust` (bool flag)

### `compressor.py` -- MemoryCompressor

Monitors working memory size and triggers LLM-driven compression when event count exceeds `compression_threshold`. Implements the MEM1 per-step internal state pattern. On compression, writes to S-MMU via `compact_to_arrow_with_meta()` with keywords, embedding (via `Embedder`), and a dynamic summary.

- **Key exports**: `MemoryCompressor`

### `embedder.py` -- Embedder

Unified embedding adapter with 3-tier auto-detection fallback:
1. **RustEmbedder** (ONNX via `sage_core`, native SIMD) -- fastest
2. **sentence-transformers** (Python, `all-MiniLM-L6-v2`) -- accurate
3. **Hash fallback** (SHA-256 projection) -- deterministic, no ML

All backends produce 384-dimensional vectors (matching `all-MiniLM-L6-v2`).

- **Key exports**: `Embedder`, `EMBEDDING_DIM` (384)

### `smmu_context.py` -- S-MMU Context Retrieval

Queries the multi-view S-MMU graph (temporal, semantic, causal, entity edges) via BFS with configurable weights. Returns a formatted context string injected as a SYSTEM message during the THINK phase of `AgentLoop`. Best-effort: all failures return empty string.

- **Key exports**: `retrieve_smmu_context()`

### `semantic.py` -- SemanticMemory (Tier 2)

In-memory entity-relation graph built by `MemoryAgent`. Stores entities as a deduplicated set and relationships as `(subject, predicate, object)` triples with adjacency index. Supports SQLite persistence (`~/.sage/semantic.db`) via `save()`/`load()`. Bounded by `max_relations` (default 10,000) with deduplication.

- **Key exports**: `SemanticMemory`

### `episodic.py` -- EpisodicMemory (Tier 1)

Cross-session persistent store backed by SQLite (`aiosqlite`). Supports CRUD + keyword search. Defaults to `~/.sage/episodic.db` when booted via `boot_agent_system()`. Falls back to in-memory when `db_path=None`.

- **Key exports**: `EpisodicMemory`

### `causal.py` -- CausalMemory

Entity-relation graph with directed causal edges (`CausalEdge`: caused, enabled, triggered, inhibited). BFS chain traversal for ancestor/descendant provenance queries. Bounded entity and context growth with eviction. Inspired by AMA-Bench (2602.22769).

- **Key exports**: `CausalMemory`, `CausalEdge`

### `remote_rag.py` -- ExoCortex (Tier 3)

Persistent managed RAG via Google GenAI File Search API. Auto-configured with a default store. Provides passive grounding during `_think()` and active `search_exocortex` agent tool. 500+ research sources indexed.

- **Key exports**: `ExoCortex`, `DEFAULT_STORE`

### `memory_agent.py` -- MemoryAgent

Autonomous entity/relation extraction agent. Runs during the LEARN phase of the agent loop. Supports heuristic (regex) or LLM-powered extraction. Feeds results into `SemanticMemory`.

- **Key exports**: `MemoryAgent`, `ExtractionResult`

### `write_gate.py` -- WriteGate

Confidence-based write gating with deduplication. Rejects low-confidence writes (below configurable threshold) and duplicate content. Uses bounded `OrderedDict` for seen-content tracking.

- **Key exports**: `WriteGate`, `WriteDecision`
