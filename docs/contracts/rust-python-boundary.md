# Rust ↔ Python Boundary Contract

**Status**: documentary contract (no code refactor in cycle 8 / 9). Per cgpro 2026-04-30 architect review Q-B: *"créer docs/contracts/rust-python-boundary.md … avec ces règles"*.

YGN-SAGE is a hybrid Rust/Python stack:
- `sage-core` (Rust + PyO3): hot-path orchestration, sandbox, routing/bandit, topology engine, formal verifier, S-MMU, posterior_epoch guard.
- `sage-python`: SDK, providers, runtime contracts (event log / oracle / evidence / run_frame), bench, ops, UI orchestration.
- `sage-discover`: knowledge pipeline (mostly separate, but does import sage_core/sage-python in places).
- `ui/`: FastAPI + WebSocket dashboard, depends on sage-python and references sage_core in app surface.

This document **locks the dependency direction** and the **ownership matrix** so Python-side workarounds don't accrete into a "misc PyO3 helpers" boundary.

## Dependency direction (locked)

```
sage-core (Rust)
  ↓ exported via PyO3 (sage-core/src/lib.rs)
sage-python
  ↓ imported by
ui / bench / protocols / ops / providers / runtime
sage-discover
  mostly separate pipeline,
  but currently imports/uses sage_core (audit follow-up: tighten)
```

**Rules**:
1. Rust MUST NOT depend on Python logically. PyO3 is *export* boundary, not *import* dependency. Verified 2026-04-30 by `grep -r "use pyo3" sage-core/src` — only PyO3 export markers, no Python imports.
2. Python MAY depend on Rust via the PyO3-exported surface. Pin via `sage-core>=0.1,<0.2` in `sage-python/pyproject.toml`.
3. `sage-discover` should *not* depend on Rust hot paths if avoidable. Today it does in places — tracked as A31-followup / boundary-tightening cleanup.
4. `ui/` must access Rust state ONLY through Python orchestration APIs (no direct PyO3 import in FastAPI routes).

## Ownership matrix

| Domain | Owner | Python may | Python MUST NOT |
|---|---|---|---|
| **Topology persistent state** (bandit_state.db, archive_state.db, engine_extras.json, topology_state_manifest.json) | Rust (`sage-core/src/topology/engine.rs`, `posterior_epoch.rs`) | call `load_state` / `save_state` via PyO3, run reset script preflight, write `posterior_epoch.json` via `a14_reset.py` | recompute alternate state truth (no shadow bandit posterior storage in Python), bypass A14 guard via in-Python state load |
| **Routing decision** (Stage 0) | Rust (`sage-core/src/routing/system_router.rs`) | call `route_integrated_contextual()`, store `decision_id` in pipeline_context | maintain a Python heuristic ComplexityRouter as PRIMARY (it remains an emergency Priority-3 fallback only — CLAUDE.md directive #4) |
| **Bandit selection** (within routing) | Rust (`sage-core/src/routing/bandit.rs`) | call `select_with_context_for_template`, `record_outcome`, save/restore via PyO3 | record outcome against an arm whose model_id ≠ the executed model_id (cgpro 2026-04-26 found this prod bug; A14b cycle-9 closes the lingering Stage-0-bypass case) |
| **Sandbox execution** (Wasm-RustPython for tool execution) | Rust (`sage-core/src/sandbox/`, ADR-013 §5 flip 2026-04-22) | configure / invoke `validate_and_execute` | bypass via Python subprocess fallback (removed 2026-04-22), use `execute_raw` without `SAGE_UNSAFE_RAW_EXEC=1` gate |
| **Formal verifier** (Z3/OxiZ SmtVerifier, QualityLabeler) | Rust (`sage-core/src/formal_verifier/`, `--features smt`) | call from QualityLabeler / OracleStack via PyO3 | implement Python heuristic that pretends to be formal verification |
| **Memory primitives** (S-MMU paging, Arrow STM) | Rust (`sage-core/src/memory/`) | call STM read/write via PyO3, run consolidation orchestration in Python | hold the canonical STM bytes in Python (Rust holds, Python observes) |
| **Runtime event log** (13 typed events + payload_schemas) | Python (`sage-python/src/sage/runtime/event_log/`) | emit / validate events, run audit mode validator | hide policy semantics in free-form strings (cycle-7 round-1 controller_decision.reason class) |
| **Oracle / evidence** (Tool/Formal/Spec/LLMJudge oracles, evidence producers) | Python (`sage-python/src/sage/runtime/oracle/`, `evidence/`) | run oracles, build runtime deltas | update training sinks (bandit/MAP-Elites/online-evolution/training-memory) without `OracleVerdict.trainable=True` |
| **Providers / LLM** (7 providers, model routing per cards.toml) | Python (`sage-python/src/sage/providers/`) | API orchestration, retries, cost tracking | inline LiteLLM (deprecated, use PydanticAIProvider per feedback memory 2026-04-18), hardcode model quirks not in cards.toml (CLAUDE.md directive #7) |
| **UI / Dashboard** | Python (`ui/`, FastAPI + WebSocket) | observe pipeline state via safe APIs, render traces | mutate topology state directly, bypass A14 guard, emit runtime events outside the registered schemas |

## Three "shim accretion" zones to monitor (cgpro Q-B 2026-04-30)

cgpro flagged three areas where the boundary is "thicker than ideal." Documented here so they don't accrete further without architectural decision:

### 1. Graph helper functions (PyO3/Windows workarounds)
Functions like `graph_get_predecessors` / `graph_get_edges` exist in `sage-core/src/lib.rs` as PyO3 wrappers around Rust `TopologyGraph` methods that don't transit cleanly through PyO3 on Windows. Acceptable today; document as "shim, scheduled for cleanup when PyO3 ≥ 0.27 stabilizes the relevant ABI."

### 2. TopologyController split-brain (Rust + Python both decide)
Rust exposes `RustTopologyController` (ADR-012, 2026-04-20 Rust-First migration). Python remains the orchestrator of many side-effects/runtime contracts (Stage 6 learning gate, controller_decision payload allowlist, bypass guard). **This is not a bug — it is the boundary**: Rust owns adaptation state + per-path primitives; Python owns orchestration + runtime contract emission. ADR-012 amendment 2026-04-23 documented this. Do not market "Rust controller" as if all control were Rust-first.

### 3. Observability bridge (Rust spans → Python OTel)
B1.b 2026-04-25 added Rust→Python OTel bridging via W3C traceparent across PyO3. Useful, but risks becoming "second runtime semantics layer" if B1.c/d/e + B1.b.7 + B1.b.9 keep growing without an explicit "trace + replay" pillar (B2 deferred). Hold further OTel expansion until B2 design lands (cgpro Q-G recommendation: defer OTel luxuries to v0.2 cleanup).

## Maintenance discipline

- When adding a new PyO3 export to `sage-core/src/lib.rs`: it MUST land in the ownership matrix above. If Python is the natural owner of the concept, prefer pure-Python implementation calling smaller Rust primitives; if Rust is the natural owner (state, hot-path, formal proof), the export must be small/atomic, not a "misc helpers" surface.
- When sage-discover or ui/ adds a new dependency on sage_core or sage-python: document it in the dependency-direction map above. Long-term goal: sage-discover and ui/ depend ONLY on sage-python public APIs, never directly on sage_core.
- Cross-component PRs (touching both Rust and Python in a non-PyO3-only way) require explicit boundary-impact note in the commit message.

## References

- ADR-011 (Rust-First migration plan, 2026-04-20)
- ADR-012 (Rust-primary TopologyController + Python orchestration boundary, 2026-04-20)
- ADR-013 (Wasm sandbox by default, §5 flip 2026-04-22)
- ADR-018 / ADR-019 (Runtime cycle 5/6 design)
- cgpro 2026-04-30 architect review Q-B (saved at `.tmp/cgpro_architect_review_finaltext.md`)
- `CLAUDE.md` directive #1 (Rust First, Python Tolerant) and #4 (kNN router primary, ComplexityRouter emergency fallback only)
- `docs/contracts/runtime-integrity-ledger.md` (companion document, 5 invariants)
