# YGN-SAGE Codebase Connectivity Audit

**Date**: 2026-03-05
**Author**: Yann Abadie + Claude Opus 4.6

---

## Executive Summary

Full audit of all files/modules in the YGN-SAGE repository. Identified **6 dead-code modules**, **6 partially-wired modules**, **~15 orphaned peripheral files**, and confirmed that all **core production code is properly connected**. Dependency versions are current — no critical upgrades required.

---

## 1. Dead Code (Never Imported Anywhere)

| Module | Purpose | Why it exists | Action |
|--------|---------|---------------|--------|
| `sage/evolution/sandbox_evaluator.py` | Sandbox-based code evaluation | Planned evolution evaluator, never connected to EvolutionEngine | Wire or remove |
| `sage/memory/neo4j_driver.py` | Neo4j graph DB driver | Future graph persistence (dep installed: `neo4j>=5.17`) | Wire into compressor or remove dep |
| `sage/memory/qdrant_driver.py` | Qdrant vector DB driver | Future vector persistence (dep installed: `qdrant-client>=1.8`) | Wire into compressor or remove dep |
| `sage/memory/python_backend.py` | Pure-Python WorkingMemory alt | Benchmark alternative to Rust backend | Archive or remove |
| `sage/llm/anthropic.py` | Anthropic Claude provider | Alternative LLM backend (dep: `anthropic>=0.52`) | Wire into ModelRouter or remove |
| `sage/llm/openai.py` | OpenAI GPT provider (non-Codex) | Alternative LLM backend (dep: `openai>=1.82`) | Wire into ModelRouter or remove |

### Impact
These 6 files add ~600 lines of dead code and 4 unused dependencies (`neo4j`, `qdrant-client`, `anthropic`, `openai`) that bloat install time.

---

## 2. Partially Wired (Exist but not fully integrated)

| Module | Status | Gap |
|--------|--------|-----|
| `sage/evolution/ebpf_evaluator.py` | Exported in `__init__.py` | Never instantiated — no code path calls it |
| `sage/memory/compressor.py` | Accepted as parameter in Agent/AgentLoop | Never instantiated anywhere — parameter always `None` |
| `sage/memory/episodic.py` | Exported in `__init__.py`, tested | Not connected to main AgentLoop — never called in production |
| `sage/memory/base.py` | Protocol definitions | Type hints only — no concrete use beyond `compressor.py` |
| `sage/llm/registry.py` | Provider registry pattern defined | Never populated — boot.py instantiates providers directly, bypassing it |
| `sage/tools/builtin.py` | BashTool defined | Never registered into any ToolRegistry by default |

---

## 3. Orphaned Peripheral Files

### `labs/` (3 files) — ORPHANED
| File | Purpose |
|------|---------|
| `commercial_hft_optimizer.py` | HFT eBPF mutation demo |
| `deploy_autonomous_hft.py` | Autonomous trading loop with Gemini reflection |
| `gcp_hft_node.py` | GCP HFT deployment with Grok integration |

**Note**: GEMINI.md explicitly says *"Stop simulating HFT and focus on architecture."* These are historical experiments, not production code.

### `debug/` (10 files) — DIAGNOSTIC, NOT IN TEST SUITE
All standalone scripts. Two have **broken imports**:
- `debug/run_ygn_sage_agent.py` — imports `query_research_nbs` and `notebooklm_agent_sync` from `scripts/` (don't exist)
- `debug/sync_large_file.py` — imports `notebooklm_agent_sync` (doesn't exist)

### `scripts/` (3 files) — MIXED
| File | Status |
|------|--------|
| `gemini_memory_sync.py` | Legitimate CLI tool (manual use) |
| `deploy_gcp.ps1` | Deployment utility (manual use) |
| `list_notebooks.py` | BROKEN — uses unofficial `notebooklm` lib not in pyproject.toml |

### `research_journal/` — KNOWLEDGE ONLY
79 JSON hypothesis logs + 3 synthesis docs. Used for analysis, not runtime.

### `conductor/` — DOCUMENTATION ONLY
Strategic roadmaps and track registries. No code imports them.

### `memory-bank/` — ACTIVE REFERENCE
Architectural memory files, actively cited.

---

## 4. Wiring Summary by Pillar

| Pillar | Wired | Partial | Dead | Total | Health |
|--------|-------|---------|------|-------|--------|
| **Topology** | 5 | 0 | 0 | 5 | 100% |
| **Strategy** | 4 | 0 | 0 | 4 | 100% |
| **Tools** | 4 | 1 | 0 | 5 | 80% |
| **Sandbox** | 2 | 0 | 0 | 2 | 100% |
| **Evolution** | 6 | 1 | 1 | 8 | 75% |
| **Memory** | 3 | 3 | 3 | 9 | 33% |
| **LLM** | 6 | 1 | 2 | 9 | 67% |
| **Core** | 4 | 0 | 0 | 4 | 100% |
| **TOTAL** | **34** | **6** | **6** | **46** | **74%** |

**Weakest pillar: Memory** — 6 out of 9 modules are dead or partial. The neo4j/qdrant/episodic/compressor chain was designed but never connected to the runtime.

---

## 5. Dependency Version Audit

### Python (installed vs pinned)

| Package | Pinned | Installed | Latest Known | Status |
|---------|--------|-----------|--------------|--------|
| `google-genai` | `>=1` | 1.65.0 | ~1.65 | OK — using correct unified SDK (`from google import genai`) |
| `fastapi` | not pinned | 0.135.1 | ~0.131+ | OK — no breaking WebSocket changes |
| `pydantic` | `>=2.10` | 2.12.5 | ~2.12 | OK |
| `httpx` | `>=0.28` | 0.28.1 | ~0.28 | OK |
| `z3-solver` | `>=4.13` | 4.16.0 | ~4.16 | OK |
| `pyarrow` | `>=18` | 23.0.1 | ~23 | OK |
| `rich` | `>=13` | 14.3.3 | ~14 | OK |
| `pytest` | `>=8` | 9.0.2 | ~9 | OK |
| `ruff` | `>=0.11` | 0.15.4 | ~0.15 | OK |

### Rust (Cargo.lock)

| Crate | Pinned | Locked | Latest Known | Status |
|-------|--------|--------|--------------|--------|
| `pyo3` | 0.25 | 0.25.1 | **0.28** | OUTDATED — 0.26+ removes GIL refs API, 0.28 has `PyClassInitializer` changes |
| `arrow` | 55.0 | 55.2.0 | **57+** | OUTDATED — monthly releases, but non-breaking minor bumps |
| `wasmtime` | 29.0 | 29.0.1 | **38.0** | OUTDATED — significant version gap |
| `solana_rbpf` | 0.8.5 | 0.8.5 | ~0.8 | OK |
| `pyo3-arrow` | 0.10.0 | 0.10.0 | needs pyo3 upgrade first | BLOCKED |

### Rust Upgrade Notes

- **PyO3 0.25 -> 0.28**: Major migration required. `FromPyObject::extract` -> `extract_bound`, `new_bound` -> `new`, `IntoPyDict` becomes fallible. Must be done carefully.
- **Wasmtime 29 -> 38**: Major version jump. Check changelog for API changes in sandbox manager.
- **Arrow 55 -> 57**: Usually non-breaking minor bumps, but pin to exact major.

**Recommendation**: PyO3 and Wasmtime upgrades are significant efforts. Schedule them as dedicated tasks, not ad-hoc.

---

## 6. sage-discover Assessment

**NOT orphaned** — intentionally standalone. It is:
- The flagship research agent using all 5 pillars
- The Dockerfile's entrypoint (`mcp_gateway.py`)
- The B2B integration layer (MCP tools for ERP/MES)
- Actively tracked in conductor (`flagship-discover: DONE`)

**But**: Not connected to the dashboard. Runs as a separate service.

---

## 7. Recommended Actions (Priority Order)

### P0 — Fix Broken Code
1. Fix `debug/run_ygn_sage_agent.py` — remove broken imports (`notebooklm_agent_sync`, `query_research_nbs`)
2. Fix `debug/sync_large_file.py` — same broken imports
3. Remove `scripts/list_notebooks.py` — uses unofficial lib, replaced by `gemini_memory_sync.py`
4. Fix `seedEvoGrid()` reference in dashboard reset — DONE (replaced with `drawEvoGrid()`)

### P1 — Wire Partially Connected Modules
5. Wire `MemoryCompressor` into `boot.py` — instantiate and pass to AgentLoop
6. Wire `EpisodicMemory` into compressor chain — enable cross-session recall
7. Register `BashTool` in default ToolRegistry in `boot.py`
8. Populate `LLMRegistry` in `boot.py` instead of hardcoded provider instantiation

### P2 — Decide: Wire or Remove Dead Code
9. `neo4j_driver.py` + `qdrant_driver.py` — wire into compressor or remove + drop deps
10. `anthropic.py` + `openai.py` — wire into ModelRouter or remove + drop optional deps
11. `sandbox_evaluator.py` — wire into EvolutionEngine or remove
12. `python_backend.py` — remove (Rust backend is canonical)

### P3 — Housekeeping
13. Archive `labs/` to `Researches/experimental_hft/` (per GEMINI.md directive)
14. Add `debug/README.md` documenting each diagnostic script
15. Schedule Rust dependency upgrades (PyO3 0.28, Wasmtime 38, Arrow 57)
