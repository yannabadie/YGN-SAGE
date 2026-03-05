# YGN-SAGE: Z3 + SIMD + S-MMU Integration Design

**Date**: 2026-03-05
**Author**: Yann Abadie + Claude Opus 4.6
**Status**: Approved

## Context

Three critical features in sage-core are incomplete or disabled:
- **Z3 Validator**: Dependency commented out, module unreachable from Python
- **SIMD H96 Sort**: AVX-512 partition is a stub falling back to scalar
- **S-MMU**: Single temporal graph with hardcoded 1.0 weights, no paging

These features are core to YGN-SAGE's ASI vision:
- Z3 prevents evolved code from being unsafe
- SIMD enables sub-ms MCTS topology selection at scale
- S-MMU manages agent memory beyond context window limits

## Decision Record

| Feature | Decision | Rationale |
|---------|----------|-----------|
| SIMD | Replace manual AVX-512 with `vqsort-rs` | Portable, proven, 3-5x faster, zero unsafe |
| Z3 | Both Safety Gate + PRM, in two phases | Full formal verification pipeline |
| S-MMU | Multi-graph hierarchical (4 orthogonal views) | Most powerful retrieval, aligns with SOTA 2026 |

## 1. SIMD — vqsort-rs Integration

### Architecture

```
sage-core/src/simd_sort.rs (refactored)
├── h96_quicksort(Vec<f32>) → Vec<f32>          # vqsort::sort() wrapper
├── h96_quicksort_zerocopy(&PyArray1<f32>)       # vqsort in-place on NumPy slice
├── vectorized_partition_h96(Vec<f32>, f32)       # vqsort + binary_search partition
└── h96_argsort(Vec<f32>) → Vec<usize>           # NEW: indexed sort for MCTS UCB
```

### Changes

**Cargo.toml**:
- Add: `vqsort-rs = "0.2"`
- Remove: manual `_mm512_*` intrinsics from simd_sort.rs
- Keep: `raw-cpuid` (HardwareProfile), `numpy` (zero-copy)

**simd_sort.rs**:
- Remove `partition_avx512()` unsafe function entirely
- Remove `partition_scalar()` internal function
- `h96_quicksort()`: delegates to `vqsort::sort()`
- `h96_quicksort_zerocopy()`: calls `vqsort::sort()` on NumPy slice
- `vectorized_partition_h96()`: `vqsort::sort()` + `partition_point()` (O(N log N + log N))
- NEW `h96_argsort()`: creates index array, sorts by value via vqsort key-value pairs

### Integration Points

- `TopologyPlanner.plan()`: `h96_argsort(ucb_scores)` for MCTS node selection
- `debug/test_simd.py`: update benchmarks to reflect real SIMD performance
- `EvolutionEngine`: sort population scores for elite selection

### Benefits

- Portable: AVX2/AVX-512/NEON auto-detected by Highway
- 3-5x faster than `sort_unstable` on 50k+ elements
- Zero unsafe code to maintain
- Supports f32, f64, i32, u32, i64, u64

## 2. Z3 — Formal Verification Pipeline

### Phase 1: Safety Gate for Evolution (Blocking)

Z3 validates code invariants BEFORE eBPF/Wasm execution.

```
sage-core/src/sandbox/z3_validator.rs (reactivated + extended)
├── prove_memory_safety(addr, limit) → bool           # Existing
├── check_loop_bound(expr, cap) → bool                # Existing
├── verify_array_bounds(accesses, len) → bool          # NEW
└── validate_mutation(constraints) → ValidationResult  # NEW: unified entry point
```

**ValidationResult**:
```rust
#[pyclass]
pub struct ValidationResult {
    pub safe: bool,
    pub violations: Vec<String>,   // Human-readable rejection reasons
    pub proof_time_ms: f64,
}
```

**Pipeline integration**:
```
LLMMutator.mutate() → mutated code
    ↓
Z3Validator.validate_mutation(constraints)  ← NEW gate
    ↓ PASS
EbpfEvaluator.evaluate() / WasmSandbox.execute()
    ↓ FAIL
Reject + feedback to mutator (why Z3 rejected)
```

### Phase 2: Process Reward Model (Scoring)

Z3 integrated into KG-RLVR to score reasoning in `<think>` blocks.

```
sage-python/src/sage/topology/kg_rlvr.py (extended)
├── FormalKnowledgeGraph
│   ├── verify_bounds()         # Existing
│   ├── verify_loop()           # Existing
│   ├── verify_arithmetic()     # NEW: overflow/underflow detection
│   └── verify_invariant()      # NEW: custom pre/post-conditions
├── ProcessRewardModel
│   ├── calculate_r_path()      # Existing, extended with more patterns
│   └── score_with_z3()         # NEW: delegates to Rust Z3Validator via sage_core
```

### Changes

**Cargo.toml**:
- `z3 = { version = "0.13.3", optional = true }`
- `default = ["z3"]`

**lib.rs**:
- Uncomment `m.add_class::<sandbox::z3_validator::Z3Validator>()?;`
- Add `m.add_class::<sandbox::z3_validator::ValidationResult>()?;`

**Graceful fallback**: If Z3 feature disabled, Python keeps its soft implementation — no regression.

### Integration Points

- `EvolutionEngine.evolve()`: Z3 gate before evaluation
- `ProcessRewardModel.calculate_r_path()`: Z3-backed scoring
- `Agent.run()` with `enforce_system3=True`: reasoning verification

## 3. S-MMU — Multi-Graph Hierarchical Architecture

### Architecture

Refactor `memory.rs` into a module:

```
sage-core/src/memory/
├── mod.rs              # WorkingMemory (Tier 1 unchanged)
├── arrow_tier.rs       # ArrowTier (Tier 2, extracted from existing compaction)
├── smmu.rs             # MultiViewMMU with 4 orthogonal graphs
│   ├── TemporalGraph   # Chronological links (extended from existing)
│   ├── SemanticGraph   # Embedding similarity links
│   ├── CausalGraph     # Parent-child agent causality
│   └── EntityGraph     # Shared entity/keyword links
└── paging.rs           # page_out/page_in distant chunks
```

### MultiViewMMU Structure

```rust
pub struct MultiViewMMU {
    temporal: DiGraph<ChunkMeta, f32>,   // weight = 1.0 / (1.0 + delta_seconds)
    semantic: DiGraph<ChunkMeta, f32>,   // weight = cosine similarity
    causal:   DiGraph<ChunkMeta, f32>,   // weight = causal strength (0 or 1)
    entity:   DiGraph<ChunkMeta, f32>,   // weight = Jaccard(entities_A, entities_B)
    chunk_map: HashMap<usize, NodeIndex>,
    next_chunk_id: usize,
}

pub struct ChunkMeta {
    pub chunk_id: usize,
    pub agent_id: String,
    pub start_time: i64,
    pub end_time: i64,
    pub summary: String,
    pub keywords: Vec<String>,          // NEW: for EntityGraph
    pub embedding: Option<Vec<f32>>,    // NEW: for SemanticGraph
    pub parent_chunk_id: Option<usize>, // NEW: for CausalGraph
}
```

### Weight Computation

| Graph | Weight Formula | Computed In |
|-------|---------------|-------------|
| Temporal | `1.0 / (1.0 + delta_seconds)` | Rust (no dependency) |
| Semantic | `cosine(embedding_A, embedding_B)` | Python → Rust (via Qdrant/Gemini) |
| Causal | `1.0` if agent_B spawned by agent_A in same chunk | Rust (parent_id tracking) |
| Entity | `\|kw_A ∩ kw_B\| / \|kw_A ∪ kw_B\|` (Jaccard) | Rust (keywords passed from Python) |

### Multi-View Retrieval

```rust
impl MultiViewMMU {
    pub fn retrieve_relevant(
        &self,
        active_node: NodeIndex,
        max_hops: usize,
        weights: [f32; 4],  // [temporal, semantic, causal, entity]
    ) -> Vec<(usize, f32)>  // (chunk_id, relevance_score)

    pub fn page_out_distant_chunks(
        &mut self,
        active_node: NodeIndex,
        max_hops: usize,
        budget: usize,
    ) -> Vec<usize>  // chunk_ids paged out
}
```

Scoring: `score = Σ(weight_i × normalized_distance_in_graph_i)` via bounded BFS on each graph, then score fusion.

Paging: Chunks beyond `max_hops` in ALL graphs are candidates. Chunks close in at least one graph are retained.

### A-MEM / Zettelkasten Mapping

- **Notes** = Arrow chunks (RecordBatch)
- **Links** = Multi-graph edges (4 types)
- **Evolution** = Re-scoring edges when new chunk arrives
- **Keywords/Tags** = Extracted by Python LLM, stored in EntityGraph
- **Embeddings** = Computed by Gemini/Qdrant, stored in SemanticGraph

### Integration with Python

`compact_to_arrow()` extended to accept metadata:
```rust
pub fn compact_to_arrow_with_meta(
    &mut self,
    py: Python<'_>,
    keywords: Vec<String>,
    embedding: Option<Vec<f32>>,
) -> PyResult<Option<PyObject>>
```

Python `MemoryCompressor` calls this instead of bare `compact_to_arrow()`, passing LLM-extracted keywords and Gemini embeddings.

### What Does NOT Change

- Tier 1 (active buffer `Vec<MemoryEvent>`) — unchanged
- Arrow schema — extended with 2 new optional columns (keywords, embedding_ref)
- PyO3 API — same methods, enriched signatures

## 4. Integration Flow

```
Agent.run() loop
    │
    ├─ WorkingMemory.add_event()              ← Tier 1 (unchanged)
    │
    ├─ if event_count >= threshold:
    │   ├─ compact_to_arrow_with_meta()       ← Tier 2 (extended)
    │   ├─ MultiViewMMU.register_chunk()      ← Tier 3 (NEW: 4 graphs)
    │   ├─ MultiViewMMU.page_out_distant()    ← Semantic paging (NEW)
    │   └─ MemoryCompressor → Neo4j/Qdrant   ← Long-term (existing, fixed)
    │
    ├─ TopologyPlanner.plan()
    │   └─ h96_argsort(ucb_scores)            ← vqsort-rs (REPLACED)
    │
    └─ EvolutionEngine.evolve()
        ├─ LLMMutator.mutate() → code
        ├─ Z3Validator.validate_mutation()     ← Phase 1 (NEW)
        │   ├─ PASS → EbpfEvaluator.evaluate()
        │   └─ FAIL → feedback + reject
        └─ ProcessRewardModel.score()          ← Phase 2 (NEW)
            └─ score_with_z3()
```

## 5. Implementation Order

| Phase | Feature | Dependency | Estimated Scope |
|-------|---------|------------|-----------------|
| 1 | vqsort-rs integration | None | ~100 LOC Rust |
| 2 | Z3 Phase 1 (Safety Gate) | None | ~80 LOC Rust + ~30 LOC Python |
| 3 | S-MMU multi-graph refactor | None (Rust side) | ~400 LOC Rust + ~60 LOC Python |
| 4 | Z3 Phase 2 (PRM) | Phase 2 | ~50 LOC Rust + ~80 LOC Python |

## 6. Bug Fixes Included

| Bug | Fix |
|-----|-----|
| test_memory.rs signature mismatch | Update `WorkingMemory::new()` calls to 2 args |
| MemoryCompressor type mismatch | `str` → `[Message(role=Role.USER, content=prompt)]` |
| Missing Python deps | Add `ulid`, `pyarrow` to pyproject.toml |
| Unused Rust deps | Remove `tracing`, `tracing-subscriber`, `uuid`; minimize `tokio` features |

## References

- [vqsort-rs](https://github.com/lincot/vqsort-rs) — Google Highway vectorized quicksort for Rust
- [Z3 Rust bindings](https://github.com/prove-rs/z3.rs) — v0.13.3, high-level SMT solver
- [A-MEM (NeurIPS 2025)](https://arxiv.org/abs/2502.12110) — Agentic Memory with Zettelkasten linking
- [Mem0](https://arxiv.org/abs/2504.19413) — Production-ready agent memory with graph retrieval
- [SIMD in Rust 2025](https://news.ycombinator.com/item?id=45826348) — State of the art analysis
- [Arrow-RS](https://github.com/apache/arrow-rs) — Zero-copy columnar data for Rust/Python FFI
