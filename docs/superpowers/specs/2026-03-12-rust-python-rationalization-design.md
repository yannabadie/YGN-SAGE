# Rust/Python Architecture Rationalization — Design Spec

**Date**: 2026-03-12
**Status**: Approved
**Approach**: "Shrink Rust, Then Densify" (Approach 1 — clean before optimize)

---

## Problem Statement

Three comprehensive audits revealed architectural imbalances in YGN-SAGE:

1. **PyO3 bridge audit**: 6/7 bridges active, HybridVerifier instantiated but never called (dead code)
2. **Routing audit**: Regex-based heuristics eliminated from production path but dead code remains in 2 files (~19 regex patterns)
3. **Bidirectional migration audit**: ~4200 LOC Rust has no performance justification; ~1100 LOC Python sits on hot paths that would benefit from Rust

Current state: sage-core 17,148 LOC Rust, sage-python 18,946 LOC Python.

## Goal

Rationalize the Rust/Python boundary so that:
- **Every line of Rust** earns its keep (SIMD, ONNX, lock-free, SMT, sandbox, petgraph)
- **Every hot path** runs in Rust (kNN, relevance gate, quality estimator, event dispatch)
- **Zero dead code** across both codebases
- **All PyO3 bridges** are actively wired and tested

## Guiding Principles

1. Clean before optimize — remove dead code and migrate low-value Rust before adding new Rust
2. Each phase produces a stable, testable state with green CI
3. Python fallbacks (`try/except ImportError`) serve as safety net during migration
4. TDD: Python tests first, then implementation, then Rust removal
5. PyO3 `py.allow_threads()` → `py.detach()` migration deferred to PyO3 0.26 upgrade (deprecation is in 0.26, not 0.25)

---

## Phase 1 — Shrink Rust (Nettoyage)

### 1.1 Dead Code Removal

| Target | Action | Files |
|--------|--------|-------|
| HybridVerifier PyO3 | Remove instantiation in `boot.py:547`. Internal verifier inside `DynamicTopologyEngine` stays in Rust | `boot.py` |
| Regex heuristics | Gut `_assess_heuristic()` body: replace 10 regex patterns with a simple keyword-count fallback (3 lines, no regex). Log a deprecation warning. Keeps last-resort fallback functional for fresh installs without ONNX model | `metacognition.py`, `adaptive_router.py` |

### 1.2 Rust → Python Migration (5 modules, ~2712 LOC)

Bottom-up dependency order:

| # | Rust Module | LOC | Python Destination | Dependencies |
|---|-------------|-----|--------------------|--------------|
| 1 | `features.rs` | 287 | `strategy/structural_features.py` | None |
| 2 | `model_card.rs` | 469 | `llm/model_card.py` (dataclass + `tomllib`) | None |
| 3 | `model_registry.rs` | 438 | `llm/model_registry.py` (dict + deque) | model_card |
| 4 | `llm_synthesis.rs` | 629 | `topology/llm_synthesis.py` (Pydantic + JSON) | None |
| 5 | `system_router.rs` | 889 | `strategy/system_router.py` | model_registry, features, Rust bandit (via PyO3) |

**Removed from migration list:**
- `templates.rs` (661 LOC) — **stays Rust**: `engine.rs`, `mcts.rs`, `map_elites.rs`, `pyo3_wrappers.rs` all `use super::templates`. Removing it breaks compilation of the entire topology block.
- `mutations.rs` (808 LOC) — **stays Rust**: `engine.rs` and `mcts.rs` import `apply_random_mutation`. Same dependency chain as templates.

**Cross-language coupling for system_router.py (item 5):**
The Python `system_router.py` will call the Rust `ContextualBandit` via PyO3 (`from sage_core import ContextualBandit`). This is a single method call per routing decision (`bandit.sample(context)`) — negligible overhead. The Rust `bandit.rs` already exposes `sample()` and `update()` as PyO3 methods.

**Migration protocol per module:**
1. Write Python implementation with full tests (TDD)
2. Verify feature parity via existing Rust tests translated to pytest
3. Update Python callers to use new Python module
4. Keep Rust shim alive until all downstream Rust consumers are migrated (strict dependency order: items 2+3 shims stay until item 5 is done)
5. Remove Rust module and PyO3 exports only when zero Rust `use` references remain

### 1.3 Unchanged in Phase 1

- `router.rs` (ONNX BERT inference) — performance-critical
- `bandit.rs` — called by Python system_router via PyO3
- `templates.rs` — hard dependency from engine.rs, mcts.rs, map_elites.rs, pyo3_wrappers.rs
- `mutations.rs` — hard dependency from engine.rs, mcts.rs
- `engine.rs` — complex orchestrator, too many internal dependencies
- `verifier.rs` + `ltl.rs` — borderline, reevaluate in Phase 3
- All memory Rust (Arrow, S-MMU, embedder, rag_cache)
- All sandbox (Wasm, eBPF, tree-sitter, ToolExecutor)
- `smt.rs` (OxiZ) — indisputable Rust value

### 1.4 Phase 1 Testing & Rollback

- Each migration: Python tests first (TDD), then switchover
- CI: existing Rust tests remain until confirmed removal
- Fallback: `try: from sage_core import X; except: from sage.local import X`

---

## Phase 2 — Densify Rust (Performance)

### 2.1 Python → Rust Migration (3 hot paths)

| # | Python Module | LOC | Rust Destination | Frequency | Expected Speedup |
|---|---------------|-----|------------------|-----------|------------------|
| 1 | `relevance_gate.py` | 61 | `memory/relevance_gate.rs` | Per memory injection (thousands/session) | 8x |
| 2 | `quality_estimator.py` | 69 | `routing/quality.rs` | Per LLM response | 4x |
| 3 | `knn_router.py` | 296 | Consolidate into `routing/router.rs` (existing SIMD kNN) | Per task | 8x |

Order: relevance_gate first (smallest, most frequent), then quality_estimator, then kNN consolidation.

### 2.2 Rust EventBus

| Aspect | Current (Python) | Target (Rust) |
|--------|-----------------|--------------|
| Dispatch | `threading.Lock` per emit | `crossbeam-channel` bounded(4096) lock-free |
| Subscribers | `dict` + lock | `DashMap` (already in Cargo.toml) |
| Async stream | `asyncio.Queue` | `crossbeam::Receiver` + PyO3 async wrapper |
| Throughput | TBD (benchmark before migration) | Target: 10x current |

**Prerequisite**: Run a micro-benchmark on current Python EventBus to establish baseline throughput. If current throughput is adequate for the ~200 events/session use case, defer this migration.

New dependency in `Cargo.toml`: `crossbeam-channel = "0.5"`.

PyO3 class `RustEventBus`:
- `emit(event_json: str)` — lock-free publish
- `subscribe(phase: Optional[str]) -> subscriber_id`
- `poll(subscriber_id: str) -> Optional[str]` — non-blocking receive

### 2.3 PyO3 Interfaces

Each new Rust module exposes a minimal PyO3 class:

```
RustRelevanceGate.score(task: str, context: str) -> f32
RustQualityEstimator.estimate(task: str, response: str, latency_ms: f64, had_errors: bool, avr_iterations: u32) -> f32
RustEventBus.emit(event_json: str)
RustEventBus.poll(subscriber_id: str) -> Optional[str]
```

Python pattern unchanged: `try: from sage_core import RustX; except: from sage.local import X`

### 2.4 Unchanged in Phase 2

- EventBus Python remains as fallback
- semantic.py / causal.py — deferred to Phase 3
- agent_loop.py — async orchestration, stays Python
- All I/O-bound code (providers, protocols, dashboard)

---

## Phase 3 — Consolidation

### 3.1 Unified Entity Graph

Merge `semantic.py` (222 LOC) + `causal.py` (293 LOC) → `memory/entity_graph.rs`

| Aspect | Current | Target |
|--------|---------|--------|
| Structure | 2 separate Python graphs (adjacency dicts + deque BFS) | 1 `petgraph::DiGraph` with typed edges (Semantic, Causal, Temporal) |
| Persistence | 2 SQLite DBs (`semantic.db`, `causal.db`) | 1 SQLite DB (`entity_graph.db`) via rusqlite (`cognitive` feature) |
| BFS | Python deque, O(V+E) with interpreter overhead | petgraph native, cache-friendly |
| PyO3 | No bridge today | `RustEntityGraph` class: `add_entity()`, `add_relation()`, `get_context_for()`, `get_causal_chain()` |

### 3.2 Borderline Modules — Final Verdict

| Module | LOC | Verdict | Reason |
|--------|-----|---------|--------|
| `bandit.rs` | 813 | **Keep Rust** | Thompson sampling per-task, integrated with SystemRouter Python |
| `verifier.rs` | 660 | **Keep Rust** | Used internally by `engine.rs` |
| `ltl.rs` | 533 | **Keep Rust** | Wired into verifier.rs |
| `executor.rs` | 476 | **Keep Rust** | Topology scheduling in evolution loop |
| `engine.rs` | 872 | **Keep Rust** | MCTS + 6 paths + MAP-Elites — core topology search |

The topology block (engine + verifier + ltl + executor + templates + mutations) stays Rust as a cohesive unit.

### 3.3 PyO3 `py.detach()` Migration

`py.allow_threads()` is deprecated in PyO3 **0.26** (not 0.25). Project currently uses PyO3 0.25.

**Action**: Deferred to PyO3 0.26 upgrade. When upgrading:

```rust
// Before (PyO3 0.25)
py.allow_threads(|| { /* ... */ })

// After (PyO3 0.26+)
py.detach(|| { /* ... */ })
```

Add as a maintenance task, not part of this rationalization.

### 3.4 SQLite Schema Migration (Entity Graph)

Merging `semantic.db` + `causal.db` → `entity_graph.db` requires data migration:

1. First run after upgrade: detect old DBs in `~/.sage/`
2. Auto-migrate: read all entities/relations from old DBs, insert into new unified DB
3. Rename old DBs to `.bak` (not delete)
4. Subsequent runs: use `entity_graph.db` only

---

## Final State

| Metric | Before | After Phase 3 |
|--------|--------|---------------|
| **Rust LOC** | 17,148 | ~15,300 (-11%) — Phase 1 removes ~2,712, Phase 2 adds ~400, Phase 3 adds ~500 |
| **Python LOC** | 18,946 | ~18,700 (-1%) — gains ~1,500 from Rust migration, loses ~500 from Python→Rust |
| **Active PyO3 bridges** | 6 (+ 1 dead) | 10 (all active) |
| **Dead code** | HybridVerifier, regex heuristics | 0 |
| **Rust modules without perf justification** | 5 (was 7, but templates+mutations stay) | 0 |
| **Python hot paths** | 4 (kNN, gate, quality, bus) | 0 |

Rust becomes smaller but **100% performance-dense**: SIMD, ONNX, lock-free, SMT, sandbox, petgraph.

## Performance Projections (cumulative)

| Component | Current | After Rust | Speedup |
|-----------|---------|------------|---------|
| Routing per-task | ~50ms | ~10ms | 5x |
| Memory injection | ~10ms | <1ms | 10x |
| Event dispatch (session) | ~200ms | ~30ms | 7x |
| Entity retrieval | ~50ms | ~15ms | 3x |
| **Total overhead/session** | **~310ms** | **~56ms** | **5.5x** |

## Risk Mitigation

1. **Rollback**: Python fallbacks (`try/except ImportError`) on every PyO3 bridge
2. **CI**: Both Rust and Python tests run until migration confirmed per module
3. **Shadow period**: 1 week dual-run per migrated module before Rust deletion
4. **Feature flags**: New Rust modules behind Cargo feature flags (existing pattern)

## Dependencies

- `crossbeam-channel = "0.5"` — new, for EventBus (Phase 2, conditional on benchmark)
- `tomllib` — Python stdlib since 3.11 (note: NOT `tomli` which is the third-party backport)
- No other new dependencies required

## Revision History

- **v1** (2026-03-12): Initial spec
- **v2** (2026-03-12): Fixed 10 issues from spec review:
  - [Critical] Kept `templates.rs` + `mutations.rs` in Rust (engine.rs dependency chain)
  - [Critical] Documented system_router → bandit cross-language coupling via PyO3
  - [Critical] Corrected PyO3 deprecation to 0.26 (not 0.25), deferred migration
  - [Important] Fixed dual-run ordering (shims stay until downstream consumers migrated)
  - [Important] Replaced `NotImplementedError` heuristic with degraded keyword-count fallback
  - [Important] Fixed `tomli` → `tomllib` (stdlib name)
  - [Important] Fixed QualityEstimator API to full 5-parameter signature
  - [Suggestion] Added EventBus benchmark prerequisite
  - [Suggestion] Added SQLite schema migration strategy for entity graph
  - [Suggestion] Updated LOC math with itemized breakdown
