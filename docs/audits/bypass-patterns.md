# Bypass Patterns — "Rust built, Python doesn't call it"

**Status:** living document — append a row each time a new bypass is found.
**First written:** 2026-04-19 after the G-series + H1-H6 iteration.

---

## The pattern

A `pyclass`/`pymethods` Rust implementation exists in `sage-core/` and is
correctly exposed via PyO3 to `sage_core.*`. Python code imports the
symbol, the type-check passes, unit tests against the Rust class in
isolation pass. **But no runtime code path actually calls the method
during a real `pipeline.run()` cycle.** The feature is documented as
"wired" in architecture.md; it isn't.

Why this bites us:
- **Unit tests silently pass.** The gate/estimator/controller works
  when tested in isolation; `assert X.evaluate(...)` is fine.
- **Integration tests silently pass.** Most integration tests mock
  around the bypassed surface — they test what's wired, not what's
  documented.
- **Pass-rate metrics lie.** Architecture claims a feature; benchmarks
  don't measure whether it fired. So "SA-3 complete" can be false for
  months without anyone noticing.
- **The silent-None fallback.** The call site reads
  `if thing_or_None is not None: thing_or_None.do_work()`, and when
  wiring drops, the `is not None` guard skips gracefully. The wiring
  defaults to off, and off is indistinguishable from "works."

This is the **exact failure mode** the advisor flagged on the H1
commit: *"you'd wire engine.evolve() perfectly and observe zero
effect."* H4 proved it — the H1 commit was correct at the call-site
level but `record_outcome` silently no-op'd without `cache_topology`.

---

## Catalog of bypasses found and fixed (2026-04-19 iteration)

| ID | Surface | Commit | Root cause | Test pinning regression |
|---|---|---|---|---|
| G-series | `CompositeWriteGate` → `phases/act.py` memory writes | `c905d06` | Gate built + exported but 0 runtime call sites. 3 writes (episodic.store, semantic.add_extraction, causal.add_entity) called storage APIs directly | `test_factory_wires_write_gate_onto_loop` |
| H1 | `engine.should_evolve()` / `engine.evolve()` → LEARN stage | `2cd840e` | Rust impl + PyO3 bindings existed, `_auto_evolve=True` set on AgentLoop but read by nothing. No Python call site invoked either method | `TestPipelineEvolutionWiring` |
| H4 | `engine.cache_topology()` between generate and record_outcome | `dc51976` | H1 was correctly wired but `record_outcome` silently no-ops on the MAP-Elites archive unless the topology is in `topology_cache` first. `engine.generate()` does NOT auto-cache. Empirical poke caught it: 8 outcomes → cell_count stays 0 | `TestRealEngineEvolutionLoop` |
| H5 | `write_gate` on singleton AgentLoop (single-agent bypass) | `27a9a4c` | G-series wired the gate through `create_node_agent_loop` for multi-node traversal. The single-agent path at `pipeline.py:941` reused the pre-existing singleton, which never went through the factory | `test_pipeline_single_agent_wires_write_gate_onto_agent_loop` |
| H6 | `_on_drift` on singleton AgentLoop (single-agent bypass) | `aa348e1` | Same shape as H5. Multi-node path builds the drift callback via `runner.py:502-521` and passes through the factory. Bypass path never set it → drift events logged but `ProviderPool.record_failure` never called | `test_pipeline_single_agent_wires_on_drift_onto_agent_loop` |
| H7 | singleton `max_steps` not scaled by system | `b7ced9d` | `boot.py:279` built singleton with `max_steps=MAX_AGENT_STEPS=20`; factory scales 5/10/20 per system tier. Bypass path ran S1 tasks at 4× factory-intended budget | `test_pipeline_single_agent_scales_max_steps_by_system` |
| H8 | singleton `stall_cap` disabled (D8 off) | `0b5a272` | `AgentConfig.stall_after_tool_steps` defaults to 0. Factory computes `max_steps-1` for S2/S3. Singleton could thrash the full budget on consecutive tool steps without breaking early | `test_pipeline_single_agent_sets_stall_cap_matching_factory` |
| H9 | pipeline stored descriptor-keyed id (`avr:n3:…`) into `ctx.topology_id` instead of full ULID | (plan 1.4a, this commit) | `pipeline.py:531` preferred `result.topology_id()` over `ctx.topology.id`. The engine's `topology_cache`/archive is keyed by full ULID; descriptor form never hit the cache → record_outcome miss → archive stayed at 0 on the engine branch even with H4 in place | `test_pipeline_real_engine_grows_map_elites_archive` |
| H10 | template branch (production-dominant) never called `cache_topology` | (plan 1.4a, this commit) | H4 only added `cache_topology` inside the engine branch. Template branch at `pipeline.py:~502` built a topology and returned without caching → archive never grew for the common S2/S3 sequential production path. Fix adds `_apply_topology_budget_and_cache` called from all four topology-building branches | `test_pipeline_template_branch_grows_map_elites_archive` |

Six fixes, all in one evening, all the same pattern.

---

## Systematic-search checklist for future sessions

Before declaring a feature "wired," run all of these.

### 1. Inventory PyO3 surfaces

```bash
grep -rn "#\[pymethods\]" sage-core/src --include="*.rs" | head -50
grep -E "add_class|add_function" sage-core/src/lib.rs
```

Compare against what `sage-python/src/sage` imports from `sage_core`.
Any Rust class with zero Python imports is either dead code or a
bypass.

### 2. For each Rust class that IS imported, count RUNTIME call sites

```bash
# Example for a class `Foo`:
grep -rn "Foo\b" sage-python/src/sage --include="*.py"
```

Distinguish:
- **Test files** (`tests/*` — don't count)
- **Import + re-export** (`memory/__init__.py` — doesn't count)
- **Factory construction** (`create_foo()` in a module — counts ONCE)
- **Runtime invocation** (`foo.method()` inside phases / pipeline
  stages / runner — this is the signal)

If runtime invocations = 0, you have a bypass.

### 3. For each wired surface, check BOTH code paths

YGN-SAGE's pipeline has two execution branches:

- **Multi-node topology** — goes through `topology/runner.py` + the
  `agent_loop_factory`. Factory wires per-node state.
- **Single-agent bypass** — at `pipeline.py:941` reuses
  `self._agent_loop` (a singleton built at boot). Does NOT go through
  the factory.

Any state the factory sets on a per-node AgentLoop must also be set on
the singleton. The bypasses found:

| Field | Factory sets? | Bypass path sets? (before H5/H6) |
|---|---|---|
| `_skip_routing = True` | ✅ | ✅ |
| `_current_topology = None` | ✅ | ✅ |
| `validation_level` | ✅ (by system) | ✅ (different logic but set) |
| `write_gate` | ✅ (G-series) | ❌ → H5 fix |
| `gate_current_task` | ✅ | ❌ → H5 fix |
| `gate_source_tier` | ✅ | ❌ → H5 fix |
| `_on_drift` | ✅ (D6 + runner) | ❌ → H6 fix |
| `max_steps` | ✅ (by system) | ❌ → H7 fix (commit `b7ced9d`, plan item 1.1) |
| `stall_cap` | ✅ (D8) | ❌ → H8 fix (commit `0b5a272`, plan item 1.2) |
| `tools` | ✅ (by role) | ✅ (all tools — false positive, plan item 1.3) |

All three candidate singleton bypasses from the 2026-04-19 catalog are
now resolved: two real (H7, H8) + one false positive (`tools`, see
§"Not a bypass" below for evidence).

### 4. Empirical validation — don't trust mock unit tests alone

Mock tests prove the call-site is right. They don't prove the callee
produces the expected state change. For anything that mutates state
(archive, memory, bandit posteriors), write a test that:

1. Uses the **real** Rust object (skip with `@pytest.mark.skipif(not _HAS_SAGE_CORE)`)
2. Runs the full call chain
3. Asserts state actually changed

The H4 test
(`TestRealEngineEvolutionLoop.test_record_outcome_grows_archive_only_when_topology_cached`)
is the template. Two assertions — one with cache_topology,
one without — pin both directions of the contract.

### 5. Re-audit after each wiring commit

The pattern compounds: fixing H1 introduced H4 (dependency chain).
Fixing G-series introduced H5 (code-path asymmetry). Assume every
wiring commit has a hidden bypass one layer down. Plan a validation
pass into the commit before declaring done.

---

## Red-flag commit patterns to watch for

From this iteration's commits, these phrases in a commit message
SHOULD trigger an empirical-validation check:

- "wires X into Y"
- "X now called from Y"
- "enables online / live / per-request Z"
- "closes the SA-N architecture claim"
- "gate / controller / callback now fires"

If the commit only ships unit tests against the target class, don't
merge without at least one integration test that runs the full call
chain and asserts the observable state change.

---

## Not a bypass (common false positives)

Skip these — they're fine despite grep showing low Python use:

| Rust class | Why it's fine |
|---|---|
| `RustQualityEstimator` (lexical) | Deliberately removed per architecture.md ("5-signal heuristic REMOVED, r=0.34"). The rust code is stale, not a bypass |
| `HardwareProfile` | Utility for `detect() + is_simd_capable()` — trivially optional |
| `RustRagCache`, `RustSmmu` | Grep-name mismatch — Python uses `sage_core.WorkingMemory` etc. which delegate to these under different names |
| `RustEntityGraph` | Duplicate surface. Python `CausalMemory` is the wired impl; Rust version is either dead or awaiting a future refactor to SQLite-backed storage. Refactor scope, not bypass scope |
| `tools` filter on singleton AgentLoop | Plan item 1.3 (2026-04-20) — audited, false positive. Singleton has no role concept; factory's `tools=None` actor-default matches singleton's unset (→ None → all tools). Role-based filters (verifier, formatter, synthesizer) only apply in multi-node topology traversal because `_make_single_node_topology` always creates `role="agent"` (pipeline.py:603) and single-node paths hit the bypass branch before any role-filter ever runs. No code change needed. The singleton behaves identically to a factory-built actor node. |

---

## Next audit (queued for future session)

- [ ] Verify `max_steps`, `stall_cap`, `tools` on singleton AgentLoop
      match what the factory would set. If the singleton is built for
      S2 but the task is S3, the singleton underflows.
- [ ] Rust `TopologyController` port — the biggest remaining
      Critical-Directive-#1 violation. ~6 decision paths in Python,
      zero in Rust. Deferred (doesn't fit a session). Covered in
      `2026-04-18-astropy-14995-decision-path.md` §5.1.
- [ ] `RustEntityGraph` consolidation with Python `CausalMemory`.
      Refactor, not bypass.
