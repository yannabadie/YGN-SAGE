# tests/

Integration tests for the `sage_core` crate. Run with `cargo test --workspace` (36 tests with default features, +5 with `onnx`).

## Test Files

### `test_types.rs` (8 tests)

Validates core data types:
- `AgentConfig` creation, builder chain, serialization roundtrip.
- `ToolSpec` construction.
- Default values for `MemoryScope` (Isolated) and `TopologyRole` (Root).

### `test_pool.rs` (5 tests)

Validates `AgentPool` thread-safe registry:
- Register and retrieve agents by ID.
- Search agents by name substring.
- List all agents.
- Terminate agent (status update).
- Terminate nonexistent agent (returns false).

### `test_memory.rs` (4 tests)

Validates `WorkingMemory` active buffer operations:
- Add and retrieve events by ID.
- Child agent registration.
- Recent events retrieval (last N).
- Compression: old events replaced by summary, event count preserved.

### `test_simd.rs` (5 tests)

Validates SIMD sort functions:
- `h96_quicksort` on empty, unsorted, and large (10K element) arrays.
- `vectorized_partition_h96` correctness (left < pivot, right >= pivot).
- `h96_argsort` index ordering.

### `test_smmu.rs` (7 tests)

Validates S-MMU multi-view graph and WorkingMemory integration:
- Multi-path BFS score accumulation via entity edges (diamond topology).
- Multi-path accumulation via causal edges.
- `compact_to_arrow_with_meta` registers chunks in S-MMU.
- Multi-chunk temporal linking (sequential chunks reachable, closer = higher score).
- Causal linking (parent-child chunk reachability).
- Page-out candidates (distant chunks evicted first, active chunk excluded).

### `test_embedder.rs` (5 tests, `onnx` feature-gated)

Validates `RustEmbedder` ONNX inference. All tests skip gracefully if the model file is not present:
- Single text embedding (384-dim, L2-normalized).
- Batch embedding (3 texts).
- Deterministic output (same input = same output).
- Empty batch returns empty.
- Semantic similarity ordering (similar texts closer than unrelated).

## Running Tests

```bash
# Default features (36 tests)
cargo test --workspace

# With ONNX embedder tests (requires model download first)
python sage-core/models/download_model.py
cargo test --features onnx
```
