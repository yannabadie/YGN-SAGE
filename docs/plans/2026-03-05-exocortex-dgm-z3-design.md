# YGN-SAGE Phase 3: ExoCortex Natif, DGM Coherence & Z3 Alignment

> **Date**: 2026-03-05
> **Status**: Approved
> **Author**: Yann Abadie + Claude Opus 4.6

## Goal

Elevate YGN-SAGE to autonomous operation by fixing 3 integration gaps (Z3 prompt, DGM context, NotebookLM fragility) and expanding Rust capabilities (SnapBPF, RAG cache).

## Research Backing

| Source | Finding | Applied To |
|--------|---------|------------|
| Kimina-Prover (NotebookLM) | Interleaved NL + formal DSL in thinking blocks | Z3 prompt alignment |
| ProofNet++ / Leanabell-v2 (NotebookLM) | Verifier error messages routed back into LLM context | Z3 retry loop |
| AlphaEvolve (NotebookLM) | Context-rich prompting with solver state + fitness history | DGM context injection |
| Google GenAI File Search API (Context7 + web) | `types.FileSearch` in SDK v1.65.0, persistent stores, free storage | ExoCortex migration |
| ExoCortex notebook | Rust SIMD for zero-copy Arrow traversal of external context | RAG cache layer |

## Constraint

- "Test it and if it doesn't work, use the research protocol to find another clever way" — every task includes validation + fallback research path.
- Rust adoption expanded, not underestimated.
- Generalist — no MES/ERP specialization.

---

## Task 1: Z3 Prompt Alignment (S3 Firewall Fix)

**Problem**: `agent_loop.py:164-168` says "Use `<think>` tags" but `kg_rlvr.py:80-115` expects `assert bounds(X, Y)`, `assert loop(var)`, `assert arithmetic(expr, val)`, `assert invariant("pre", "post")`. LLM never produces parseable Z3 assertions.

**Fix**: Update S3 system prompt to teach Z3 DSL (Kimina-Prover pattern). Also update S2→S3 escalation retry prompt.

**Files**: `agent_loop.py:164-168`, `agent_loop.py:~320`

**Test**: Write test with mock LLM response containing Z3 assertions → verify `kg_rlvr.py` scores > 0. Write test without → verify score = 0.

**Fallback**: If regex-based DSL is too rigid for real LLM output, research structured output (JSON Z3 constraints) parsed before regex matching.

---

## Task 2: DGM Context Injection (Directed Evolution)

**Problem**: `engine.py:144` calls `mutate_fn(parent.code)` — SAMPOSolver's `dgm_action` never reaches the LLM mutator. Mutations are blind.

**Fix**:
1. Define `DGM_ACTION_DESCRIPTIONS` mapping 5 actions to semantic directives
2. Widen `mutate_fn` signature: `Callable[[str, dict], ...]`
3. Build `dgm_context` dict with action, description, parent_score, generation
4. Wire into `LLMMutator._build_mutation_prompt()` as "DGM Directive" section

**Files**: `engine.py:108,128-145`, `llm_mutator.py:39-58`, `test_dgm_sampo.py`

**Test**: Verify `fake_mutate` receives context dict. Verify prompt contains DGM directive text.

**Fallback**: If signature change breaks external consumers, use a module-level `threading.local()` context variable instead.

---

## Task 3: Google GenAI File Search ExoCortex

**Problem**: `sage-discover/knowledge.py` depends on `notebooklm-py` (unofficial CLI, fragile OAuth, constant 400 errors).

**Fix**:
1. New module `sage-python/src/sage/memory/remote_rag.py` — `ExoCortex` class with `create_store()`, `upload()`, `get_tool()`
2. Integrate into `GoogleProvider.generate()` — optional `file_search_store_names` param injected as `types.Tool(file_search=types.FileSearch(...))`
3. Wire into `boot.py` — ExoCortex from env var `SAGE_EXOCORTEX_STORE`
4. Archive NotebookLM bridge to `_archived/`

**API** (validated against SDK v1.65.0):
```python
client.file_search_stores.create(config={'display_name': name})
client.file_search_stores.upload_to_file_search_store(file=path, file_search_store_name=store)
types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name]))
```

**Note**: `types.FileSearchTool` does NOT exist (Gemini hallucinated it). Correct class is `types.FileSearch`.

**Test**: Smoke test gated behind `GOOGLE_API_KEY` — create temp store, upload, query, delete.

**Fallback**: If File Search API has issues (quotas, model compatibility), fall back to `google_search` grounding + local episodic memory.

---

## Task 4: SnapBPF Rust Completion

**Problem**: `sage-core/src/sandbox/ebpf.rs:81-93` — `SnapBPF` is an empty skeleton. Evolution pillar re-executes from scratch on every mutation.

**Fix**: Implement userspace CoW memory snapshotting:
- `SnapBPF` struct with `DashMap<String, Arc<Vec<u8>>>` for snapshots
- `snapshot(id, memory)`, `restore(id) -> Vec<u8>`, `delete(id)` methods
- PyO3 `#[pymethods]` bindings
- Wire into `EbpfEvaluator` for pre-mutation snapshots

**Files**: `sage-core/src/sandbox/ebpf.rs:81-93`, `sage-python/src/sage/evolution/ebpf_evaluator.py`

**Test**: `cargo test` — snapshot → mutate → restore → verify memory integrity. Python test via mock.

**Fallback**: If DashMap overhead is too high for sub-ms requirements, use a fixed-size ring buffer with pre-allocated pages.

---

## Task 5: File Search Rust Cache Layer

**Problem**: Repeated ExoCortex queries add API latency. ExoCortex notebook prescribes Rust SIMD retrieval for zero-copy Arrow traversal.

**Fix**:
- New `sage-core/src/memory/rag_cache.rs` — `RagCache` struct with LRU + TTL
- Stores query results as Arrow IPC bytes, returns via PyArrow FFI zero-copy
- PyO3 bindings with `put(hash, ipc_bytes)`, `get(hash) -> Option<RecordBatch>`, `stats()`
- Wire into `remote_rag.py` — check cache before API call
- Python fallback: `dict` + TTL if `sage_core.RagCache` unavailable

**Files**: New `sage-core/src/memory/rag_cache.rs`, `sage-core/src/lib.rs`, `sage-python/src/sage/memory/remote_rag.py`

**Test**: `cargo test` — put/get/eviction/TTL. Python test — cache hit vs miss.

**Fallback**: If Arrow IPC serialization proves fragile, use msgpack bytes instead (simpler, still fast).

---

## Dependency Order

```
Task 1 (Z3 prompt) ─────────────────────────── independent
Task 2 (DGM context) ───────────────────────── independent
Task 3 (File Search ExoCortex) ──┐
                                 ├── Task 5 depends on Task 3
Task 4 (SnapBPF) ───────────────┘── independent
Task 5 (RAG cache) ─────────────── depends on Task 3
```

Tasks 1, 2, 3, 4 are independent. Task 5 depends on Task 3 (needs `remote_rag.py` to exist).

## Success Criteria

- 162+ tests passing (no regressions)
- Z3 PRM scores > 0 when LLM produces formal assertions
- DGM context visible in mutation prompts
- File Search store creation/upload/query works with `GOOGLE_API_KEY`
- SnapBPF snapshot/restore passes `cargo test`
- RAG cache hit/miss/eviction passes both `cargo test` and `pytest`
