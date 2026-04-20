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
| H6 | `_on_drift` on singleton AgentLoop (single-agent bypass) | (this commit) | Same shape as H5. Multi-node path builds the drift callback via `runner.py:502-521` and passes through the factory. Bypass path never set it → drift events logged but `ProviderPool.record_failure` never called | `test_pipeline_single_agent_wires_on_drift_onto_agent_loop` |

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
| `max_steps` | ✅ (by system) | ❌ → H7 fix (plan item 1.1, commit TBD) |
| `stall_cap` | ✅ (D8) | ❌ (uses singleton default — plan item 1.2) |
| `tools` | ✅ (by role) | ❌ (uses singleton tools — plan item 1.3) |

**Two more candidate bypasses still to verify**: `stall_cap`, `tools`.
The singleton may be configured at boot with sensible defaults that
happen to work. Plan items 1.2 and 1.3 of
`docs/superpowers/plans/2026-04-20-rust-first-plan.md` close them.

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
