# memory/

Multi-tier memory data plane for YGN-SAGE agents. Combines an append-only active buffer (Tier 1), immutable Arrow columnar storage (Tier 2), and a multi-view semantic graph (S-MMU) for cognitive routing.

## Module Layout

### `mod.rs` -- WorkingMemory

The central PyClass for per-agent memory. Manages three internal tiers:

- **Active buffer** (`Vec<MemoryEvent>`) -- Fast O(1) append for runtime events.
- **Arrow chunks** (`Vec<Arc<RecordBatch>>`) -- Immutable columnar storage after compaction.
- **S-MMU** (`MultiViewMMU`) -- Multi-view graph linking compacted chunks.

Key methods:
- `add_event(type, content)` -- Append to active buffer, returns ULID.
- `compress_old_events(keep_recent, summary)` -- Evict old events, insert summary.
- `compact_to_arrow()` / `compact_to_arrow_with_meta(keywords, embedding, parent_chunk_id, summary)` -- Compact active buffer into Arrow RecordBatch, register in S-MMU, clear buffer.
- `retrieve_relevant_chunks(chunk_id, max_hops, weights)` -- BFS traversal of S-MMU graph. Weights: `[temporal, semantic, causal, entity]`.
- `get_page_out_candidates(chunk_id, max_hops, budget)` -- Identify least-relevant chunks for eviction.
- `get_latest_arrow_chunk(py)` -- Zero-copy export of latest RecordBatch to Python/PyArrow.

### `arrow_tier.rs` -- Arrow Compaction

Converts the active buffer into an Apache Arrow RecordBatch with schema: `agent_id`, `parent_id`, `id`, `event_type`, `content`, `timestamp` (nanosecond), `is_summary`.

- `compact_buffer_to_arrow(...)` -- Builds the RecordBatch, registers the chunk in the S-MMU with metadata (keywords, embedding, parent chunk, summary). Returns the assigned chunk ID.

### `smmu.rs` -- Multi-View S-MMU

Semantic Memory Management Unit with 4 orthogonal graph views stored in a single `petgraph::DiGraph`:

- **Temporal** -- Chronological links between sequential chunks, weighted by time proximity.
- **Semantic** -- Cosine similarity links between chunks with embeddings (threshold > 0.5).
- **Causal** -- Parent-child agent causality links (weight 1.0).
- **Entity** -- Shared keyword links using Jaccard similarity (any overlap).

Key types:
- `ChunkMetadata` -- Per-chunk: id, time range, summary, embedding, keywords, parent.
- `EdgeKind` -- `Temporal`, `Semantic`, `Causal`, `Entity`.
- `MultiEdge` -- Edge label with kind + weight.
- `MultiViewMMU` -- `register_chunk(...)` builds all applicable edges. `retrieve_relevant(chunk_id, max_hops, weights)` performs weighted BFS with score accumulation across multiple paths.

### `paging.rs` -- Semantic Paging

Eviction policy for working memory pressure:

- `page_out_candidates(smmu, active_chunk_id, max_hops, budget)` -- Returns chunk IDs to evict, prioritizing unreachable chunks first, then least-relevant reachable chunks.

### `event.rs` -- MemoryEvent

Immutable event record (PyClass):

- Fields: `id` (ULID), `event_type`, `content`, `timestamp` (DateTime<Utc>), `is_summary`.
- Constructors: `new(type, content)`, `summary(content)`.
- Getters: `timestamp_str` (RFC 3339), `timestamp_ns` (nanoseconds).

### `rag_cache.rs` -- RagCache

FIFO + TTL cache for File Search (ExoCortex) query results (PyClass):

- Backed by `DashMap<u64, CacheEntry>` for lock-free concurrent access.
- `put(query_hash, data)` -- Store bytes, evict oldest at capacity.
- `get(query_hash)` -- Retrieve if not expired, else remove and return None.
- `stats()` -- Returns `(hits, misses, entries)` via atomic counters.
- Defaults: 1000 entries, 3600s TTL.

### `embedder.rs` -- RustEmbedder (behind `onnx` feature)

ONNX Runtime embedder for S-MMU semantic edges (PyClass):

- Model: all-MiniLM-L6-v2 (384-dim, L2-normalized).
- Uses `ort` 2.0 for inference, `tokenizers` 0.21 for HuggingFace tokenization.
- `embed(text)` -- Single text to 384-dim vector.
- `embed_batch(texts)` -- Batch embedding with mean pooling + L2 normalization.
- Properties: `dim` (384), `is_semantic` (true).
- Part of the Python Embedder 3-tier fallback: RustEmbedder > sentence-transformers > SHA-256 hash.
