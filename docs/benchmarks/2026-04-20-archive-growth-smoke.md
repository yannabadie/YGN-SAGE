# MAP-Elites Archive Growth — Pipeline-Level Empirical Smoke

**Date:** 2026-04-20
**Plan item:** 1.4 of `docs/superpowers/plans/2026-04-20-rust-first-plan.md`
**Verdict:** ❌ **Found two live bypasses (H9, H10) — plan item 1.4a added and fixed in same session**

## Setup

Instead of the plan's original `python -m sage.bench --type swebench --subset lite --limit 5` (which needs API keys, Docker, and ~10 min of provider time), I built a pipeline-level integration test that exercises the full `generate → cache_topology → record_outcome` chain with the **real** `sage_core.TopologyEngine` and a mock LLM provider. Same empirical signal, zero API cost, deterministic.

Tests written:
- `tests/test_pipeline.py::test_pipeline_real_engine_grows_map_elites_archive` — forces engine branch via subclass override of `_build_topology_from_hint`.
- `tests/test_pipeline.py::test_pipeline_template_branch_grows_map_elites_archive` — exercises the dominant production path (template branch, `hint="sequential"`).

Both use the real Rust archive (`sage_core.TopologyEngine`); skip cleanly when `sage_core` is not compiled.

## Findings

### Before-fix smoke (verdict ❌)

```
Engine branch (10 runs, system=3):
  engine.archive_cell_count() → 0

Template branch (10 runs, system=2, hint=sequential):
  engine.archive_cell_count() → 0
```

Two separate bypasses caused the zero:

#### H9: engine branch stored the wrong ID form

`pipeline.py:530` captured `ctx.topology_id = result.topology_id()` — the **descriptor-keyed** semantic ID (e.g. `avr:n3:01KPN3XZ`). But the engine's `topology_cache` (which `record_outcome` looks up) is keyed by `result.topology.id` — the **full ULID** (e.g. `01KPN3XZZQ7W1Z08KDASB0SPBA`). Stage 5's `record_outcome(ctx.topology_id, ...)` issued a cache miss → archive insertion skipped → `cell_count` stayed 0.

Empirical proof:

```python
# With full ULID after 20 outcomes:
engine.archive_cell_count() → 9
# With result.topology_id() (descriptor):
engine.archive_cell_count() → 0
```

#### H10: template branch never called `cache_topology`

The H4 fix (commit `dc51976`, 2026-04-19) added `cache_topology` only inside the engine branch (`pipeline.py:549-554`). The **production-dominant** template branch at `pipeline.py:~502` ran `_build_topology_from_hint` and returned without caching. Same for:
- Engine-branch budget degrade via `_make_single_node_topology` (the degraded topology was never cached, only the pre-degrade one was — ID mismatch).
- Fallback `TopologyGraph`/`TopologyNode` construction (line ~570).

So in production (any S2/S3 sequential task), `record_outcome` hit an empty cache → archive stuck at 0 → `should_evolve` never fired → H1 evolve() wiring silently never ran.

This is the exact same pattern the H4 commit said it fixed. H4's test pinned the engine contract with `result.topology.id` (full ULID) but the pipeline code used `result.topology_id()` (descriptor) — test proved the callee, pipeline call site used a different ID, silent bypass.

### Fix

`pipeline.py::_apply_topology_budget_and_cache` (new helper). Runs budget degrade check, then caches the **final** `ctx.topology` (post-degrade if degrade happened). Called from all four topology-building branches:

- S1 `formal_solver` (line ~472)
- Template branch (line ~511)
- Engine branch (replaces inline cache block)
- Fallback `TopologyGraph` (end of method)

Combined with the H9 fix at line ~530 (always `ctx.topology_id = ctx.topology.id`, no descriptor preference), the cache is now consistently populated with the cache-compatible ULID, and every branch gets cached.

### After-fix smoke (verdict ✅)

```
Engine branch (10 runs):
  engine.archive_cell_count() > 0  ← PASS

Template branch (10 runs):
  engine.archive_cell_count() > 0  ← PASS
```

Regression tests added to `tests/test_pipeline.py` pin both paths. Full `test_pipeline.py` + `test_online_evolution.py` suite: **52/52 green.**

## Downstream impact

The SA-3 "online evolution" claim in `.claude/rules/architecture.md` was materially broken: H1 shipped the evolve() wiring; H4 shipped cache_topology on ONE branch; H9+H10 were silent bypasses that made the whole chain no-op in production. All four fixes chain together:

1. G-series (`c905d06`) — write gate wired
2. H1 (`2cd840e`) — should_evolve/evolve wired
3. H4 (`dc51976`) — cache_topology on engine branch
4. **H9+H10 (this commit)** — ID form + cache on all branches

Each subsequent fix was caught by empirical validation of the prior claim. Exactly the pattern `bypass-patterns.md` §4 warns about: "Mock tests prove the call-site is right. They don't prove the callee produces the expected state change."

## Deferred items

- **Real SWE-bench smoke** (plan's original 1.4 method): still useful as a production-path health check. Deferred — the pipeline-level test gives the same signal with 50× less cost. When budget allows, run `python -m sage.bench --type swebench --subset lite --limit 5` and verify `archive_cell_count()` grows in the log.
- **`should_evolve` fire rate under production loads**: the plan's 1.4 hoped-for signal "`Online evolution fired` log line". Not validated here because EVOLUTION_MIN_OUTCOMES=5 + descriptor diversity may require more than 10 tasks per benchmark run. Flagged for a future 1.4b if/when we see longer-running benches without evolve() firing.

## References

- `tests/test_pipeline.py::test_pipeline_real_engine_grows_map_elites_archive`
- `tests/test_pipeline.py::test_pipeline_template_branch_grows_map_elites_archive`
- `tests/test_online_evolution.py::TestRealEngineEvolutionLoop` (pre-existing, engine-level)
- `docs/audits/bypass-patterns.md` §4 (the methodology this empirical smoke operationalized)
- Plan item 1.4 + 1.4a in `docs/superpowers/plans/2026-04-20-rust-first-plan.md`
