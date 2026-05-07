# ADR-016 — AgentLoop bypass factory (P6-A structural fix)

**Status:** Implemented — cycle-11 P6-B defensive serialization
landed 2026-05-04 at `450786a5` (lock + ContextVar reentry guard
wrapping the legacy shared-mutation bypass path — band-aid); P6-A
factory foundation landed cycle-12 at `9f7783cc` (3 implicit fields
propagated to `create_bypass_agent_loop()` per cgpro DESIGN trap
Q7); cycle-12 P6-A production swap shipped 2026-05-05 at `7e20372e`
(singleton AgentLoop bypass mutation block in
`pipeline_v2/execute.py` replaced with the per-run factory call —
the P6-B band-aid was closed by eliminating the shared mutable state
the lock was protecting). CI cleanup followed at `8761f0db`,
retiring the obsolete bypass-mutation tests that had locked the
P6-B contract.
**Related:** ADR-015 (pipeline.py decomposition characterization tests
shipped in cycle-11 P9 phase 1, HEAD `259b2066`); cycle-11 P6-B lock
(`450786a5`); cycle-7 / 8 / 9 / 10 runtime integrity ledger
(`docs/contracts/runtime-integrity-ledger.md`).

## Context

`sage-python/src/sage/pipeline.py:_stage_execute` single-agent bypass
branch (lines ~2300-2465) currently mutates **12 fields** on the boot
AgentLoop singleton per pipeline.run(), then restores them in a
`finally:` block:

```
_skip_routing, _current_topology, write_gate, gate_current_task,
gate_source_tier, _on_drift, _run_frame_builder,
_runtime_node_run_id, config.validation_level, config.max_steps,
config.stall_after_tool_steps, _llm + config.llm
```

Cycle-11 P6-B (`450786a5`, 2026-05-04) wrapped this snapshot/restore
block in a per-event-loop `asyncio.Lock` + `ContextVar` reentry guard
to prevent same-event-loop concurrent corruption. P6-B is the
**band-aid**; P6-A is the structural fix this ADR describes:
**replace the singleton mutation with a per-run AgentLoop factory**.
Once shipped, the snapshot/restore block disappears entirely — there
is no shared mutable state to corrupt, hence no lock needed.

The cycle-11 P9 phase 1 acceptance gate (5 files / 25 tests, HEAD
`6bb1b1e5`) locks the public surface that P6-A must preserve:

  - Test #4 Fix C tier=budget controller decision
  - Test #5 control-surface fields populated
  - Test #2 oracle gate side-effect blocking
  - Test #3 bandit attribution singleton settle invariant
  - Test #1 pipeline.run() byte-identical golden

P6-A must pass all 25 tests byte-identically against the post-swap
pipeline. ADR-015 §"Implementation order" item #1 says "Cycle-11
prep: write the 5 characterization tests" — done. P6-A's swap is
now unblocked but still requires cgpro DESIGN before implementing in
production.

## Decision

Add `create_bypass_agent_loop()` to `sage/agent_loop_factory.py`,
mirroring the existing `create_node_agent_loop()` pattern (used for
multi-agent topology execution since cycle-7 unified entry point
phase 2). The factory returns a **fresh `AgentLoop` instance per
call** with all per-run state pre-populated; cycle-12 phase 2 will
swap the bypass mutation block in `_stage_execute` for a single
factory call.

### Factory contract

```python
def create_bypass_agent_loop(
    *,
    singleton: AgentLoop,          # source of injected deps (sandbox, memory, ...)
    llm_provider: LLMProvider,     # per-run (bandit-selected or default)
    llm_config: LLMConfig,         # per-run
    system_level: int,             # 1/2/3 — drives max_steps + validation_level
    tool_registry: ToolRegistry,   # shared (boot-built)
    write_gate: Any | None,        # per-task pipeline-built gate
    task_text: str,                # for gate dedup
    on_event: Callable | None,     # event_bus.emit
    on_drift: Callable | None,     # ProviderPool.record_failure forwarder
    run_frame_builder: Any | None, # per-run, set by pipeline.run()
    runtime_node_run_id: str | None,  # per-run
) -> AgentLoop:
    """Fresh AgentLoop with all per-run state pre-populated. No
    mutation needed; instance is GC'd after pipeline.run() returns."""
```

### Field ownership matrix

| Field | Ownership | Source in factory |
|---|---|---|
| `config` | per-run | new AgentConfig instance |
| `_llm` / `config.llm` | per-run | factory args (bandit or default routing) |
| `_skip_routing` | per-run constant | always `True` (bypass means routing was Stage 0) |
| `_current_topology` | per-run constant | always `None` (bypass = no topology) |
| `config.validation_level` | derived | `{1: 1, 2: 2, 3: 3}.get(system_level, 1)` (sandbox-aware on S2) |
| `config.max_steps` | derived | `{1: 5, 2: 10, 3: 20}.get(system_level, 10)` |
| `config.stall_after_tool_steps` | derived | `(max_steps - 1) if max_steps > 5 else 0` |
| `write_gate` / `gate_current_task` / `gate_source_tier` | per-run | factory args |
| `_on_drift` | per-run | factory arg |
| `_run_frame_builder` / `_runtime_node_run_id` | per-run | factory args |
| `_tools` (ToolRegistry) | shared | from singleton |
| `working_memory` | per-instance | new `WorkingMemory(agent_id=config.name)` |
| `prm` (ProcessRewardModel) | per-instance | new `ProcessRewardModel()` |
| `sandbox_manager` | shared | copied from singleton |
| `episodic_memory` / `semantic_memory` / `memory_agent` / `causal_memory` / `consolidator` | shared | copied from singleton |
| `exocortex` / `guardrail_pipeline` / `tool_executor` / `topology_engine` | shared | copied from singleton |
| `agent_pool` / `metacognition` / `topology_population` | shared | copied from singleton |
| `_skip_memory` / `_skip_avr` / `_skip_guardrails` / `_auto_evolve` | shared | copied from singleton (ablation flags set at boot) |

Stats fields (`step_count`, `total_inference_time`, `total_cost_usd`,
`tool_call_count`, `tool_turn_count`, `executed_commands`,
`last_exhaustion`) are per-instance defaults (zeroed in `__init__`).

### Migration plan (cycle-12 phase 2)

**Phase A — additive (1 commit):**  <!-- narrative-guard: allow historical-record -->
- Land `create_bypass_agent_loop()` factory + tests in
  `agent_loop_factory.py` and `tests/test_agent_loop_bypass_factory.py`.
- Function is callable but no production code path uses it yet.
- Tests prove the factory produces an `AgentLoop` with the right
  state, given the right inputs.

**Phase B — swap (1 commit, behind cgpro DESIGN review):**  <!-- narrative-guard: allow historical-record -->
- Replace the `~150-line bypass mutation block` in
  `pipeline.py:_stage_execute` (lines 2300-2465) with a single
  `create_bypass_agent_loop()` call. The mutation snapshot/restore
  + the P6-B lock + the ContextVar reentry guard all become
  unnecessary and are removed.
- All 25 cycle-11 P9 phase 1 tests must pass byte-identically.

**Phase C — P6-B cleanup (1 commit, post-soak):**
- After phase B has been on `main` for ≥ 1 week with no regressions,  <!-- narrative-guard: allow historical-record -->
  remove the now-unused `_agent_loop_bypass_lock` /
  `_agent_loop_bypass_lock_loop` / `_BYPASS_AGENT_LOOP_ACTIVE` /
  `_acquire_bypass_lock` machinery (P6-B band-aid, ~80 lines).

This commit (cycle-11) ships only Phase A.  <!-- narrative-guard: allow historical-record -->

## Contracts that MUST be preserved

The 4 cycle-11 P9 phase 1 invariants apply transitively to P6-A:

1. **ADR-015 #1 byte-identical run surface** — final result text +
   ctx contract fields after `pipeline.run()` must match pre-P6-A
   exactly when inputs are identical. Bypass path stub coverage in
   `test_pipeline_v2_run_byte_identical.py` is the gate.

2. **Invariant 8 control-surface fields** — `ctx.executed_template ==
   "single_agent"` on the bypass path. `test_pipeline_v2_control_surface_fields.py`
   asserts this; Phase B must keep it.  <!-- narrative-guard: allow historical-record -->

3. **Invariant 6 bandit attribution singleton settle** — the
   `_record_bandit_outcome_checked` lifecycle still holds across
   ctx.bandit_decision_id from Stage 0 to Stage 5. `test_pipeline_v2_bandit_attribution_invariant.py`
   covers this; the factory does not change Stage 0/5 behavior.

4. **P6-B same-event-loop concurrency** — `test_pipeline_bypass_lock.py`
   currently asserts the lock serializes overlapping bypass entries.
   Phase B removes the lock; the test must be retired or rewritten as  <!-- narrative-guard: allow historical-record -->
   a "no shared mutable state" assertion (the structural-fix
   counterpart to the lock).

## Consequences

### Positive

- **No shared mutable state** in the bypass path. The whole class of
  cross-run corruption bugs (A0a 2026-04-23, P6-B 2026-05-04 same-loop
  concurrency) becomes structurally impossible.
- The 12-field snapshot/restore block (~50 LOC) disappears. The
  P6-B lock machinery (~80 LOC) follows in Phase C. Net code
  reduction: ~130 LOC from `pipeline.py`'s most sensitive runtime
  path.
- Stack traces and debug logs become per-run — `agent_loop.run_id`
  reflects the actual run, not the boot singleton's `__init__`-time id.

### Negative

- **Per-run AgentLoop construction cost.** Per the `__init__` audit:
  attribute init is cheap (no I/O, no model load), and shared
  injected deps are copied by reference. Estimated cost: ~50 µs per
  pipeline.run() on top of existing latency. Below the noise floor of
  any current bench.
- **Field-set drift risk.** Future commits adding a new attribute to
  `AgentLoop.__init__` must remember to ALSO copy it from the
  singleton in the factory. The matrix above documents the mapping;
  the test suite asserts the produced AgentLoop matches the singleton
  on shared fields. Drift would surface as a test failure.
- **Phase B requires cgpro DESIGN review.** The swap touches  <!-- narrative-guard: allow historical-record -->
  `pipeline.py:_stage_execute` — the most sensitive runtime path
  per cgpro 2026-04-30 architect review. Skipping cgpro DESIGN here
  re-introduces exactly the "weak-decision-on-sensitive-path" trap
  cycle-11 P9 phase 1 was designed to surface.

### Mitigations

- Phase A is reversible (factory function unused). Phase B is  <!-- narrative-guard: allow historical-record -->
  reversible via revert (swap is one commit). Phase C is reversible
  but unlikely to need it (P6-B was already a band-aid).
- The 25 cycle-11 P9 phase 1 tests are the byte-identical gate.
- A new `test_agent_loop_bypass_factory.py` (this commit, Phase A)  <!-- narrative-guard: allow historical-record -->
  asserts the factory output state matches the per-run mutation
  contract field-by-field.

## Alternatives Considered

### A. Pass a `RunContext` object into `agent_loop.run(task, ctx=...)`

**Rejected.** Means `AgentLoop` reads per-run state from an external
object during `.run()`. Spread across the loop's many phases (perceive
/ act / learn). Hard to verify all read sites; would still leave the
"singleton is mutated by Set/Get on a context arg" pattern. Per-run
factory is structurally simpler.

### B. Just keep the P6-B lock forever

**Rejected.** The lock prevents corruption but doesn't eliminate the
class. Future bypass-path features (e.g. nested sage_recurse calls,
bench parallelism) would need additional locks or context vars. The
cycle-11 cgpro round-2 review explicitly flagged P6-B as "structural
should follow" — this ADR is that follow.

### C. Refactor AgentLoop into a "Configurable" + "Stateless" split

**Rejected (for now).** Cleaner pure-functional approach but a
multi-week effort touching every `AgentLoop` consumer. The factory
is the minimal change that captures 90% of the structural benefit
at 10% of the cost.

## Open questions for cgpro DESIGN review (Phase B gate)  <!-- narrative-guard: allow historical-record -->

1. **Per-instance vs shared `prm` / `working_memory`**: the existing
   `create_node_agent_loop()` always builds fresh `WorkingMemory`
   and `ProcessRewardModel` per node (H8 isolation). Should bypass
   factory do the same, or share with singleton? Sharing means cross-
   run continuity in working memory; per-instance means clean state
   per task. Bench impact unknown — needs cgpro opinion.

2. **`agent_pool` dependency**: the singleton's `agent_pool` is set
   by `boot.py`. Should the factory copy by reference (cheap, shared
   sage_recurse pool) or new instance (clean per-task)? Sage_recurse
   relies on the pool persisting across nested calls within a single
   task — sharing is probably right.

3. **`run_id` / `node_run_id` semantics**: `_run_frame_builder` and
   `_runtime_node_run_id` are per-run. If a future bench-level driver
   creates multiple bypass loops in parallel, would they need
   separate `run_id`s on each AgentLoop? The factory assigns from
   args, so caller controls — but the contract should be explicit.

4. **Phase B test plan**: the P9 phase 1 byte-identical golden test  <!-- narrative-guard: allow historical-record -->
   uses one S1 fixture. Do we need to extend it with explicit bypass
   path coverage during Phase B's cgpro VERIFY round? Adding a 2nd  <!-- narrative-guard: allow historical-record -->
   fixture (`test_run_byte_identical_post_factory_swap.py` or similar)
   would be defense-in-depth.

## References

- `sage-python/src/sage/agent_loop_factory.py:create_node_agent_loop`
  (existing factory, established pattern)
- `sage-python/src/sage/pipeline.py:2300-2465` (current bypass
  mutation block, target of Phase B swap)  <!-- narrative-guard: allow historical-record -->
- `sage-python/tests/test_pipeline_bypass_lock.py` (P6-B lock
  regression suite, will be retired in Phase C)
- `docs/adr/ADR-015-pipeline-decomposition.md` (the meta-ADR; this
  ADR is one slice of its decomposition gate)
- `docs/contracts/runtime-integrity-ledger.md` (the 8 invariants)
- cgpro round-4 P6 design conv: `.tmp/cgpro_p6_complete_review_finaltext.md`
- cgpro round-5 P6-B closure conv: `.tmp/cgpro_p6b_close_phase2_review_finaltext.md`

## Status changes

- 2026-05-04 — P6-B defensive serialization (`450786a5`): asyncio.Lock
  + ContextVar reentry guard wrapping the legacy shared-mutation
  bypass path. Band-aid by design; structural fix deferred to P6-A.
- 2026-05-05 — P6-A factory foundation (`9f7783cc`): factory  <!-- narrative-guard: allow historical-record -->
  propagation groundwork — 3 implicit fields (`toolforge`,
  `evolution_memory`, `dangerous_tools`) propagated to
  `create_bypass_agent_loop()` per cgpro DESIGN trap Q7.
- 2026-05-05 — P6-A production swap (`7e20372e`): the singleton  <!-- narrative-guard: allow historical-record -->
  AgentLoop bypass mutation block in `pipeline_v2/execute.py` was
  replaced with the per-run factory call. The P6-B band-aid was
  closed by the swap removing the shared mutable state the lock
  was protecting (no separate post-soak Phase C follow-up was
  needed once the structural fix landed). cgpro VERIFY pre-push
  round returned `GO_PUSH` one-shot.
- 2026-05-05 — CI cleanup (`8761f0db`): retired the obsolete
  bypass-mutation tests that had locked the P6-B contract — deleted
  `test_pipeline_bypass_restoration.py` (3 obsolete tests strictly
  superseded by structural-isolation), pruned 4 obsolete
  bypass-mutation tests from `test_pipeline_bypass.py` (kept 5
  surviving), and added the autouse `_spy_loop_passthrough_factory`
  fixture in `test_pillar_logging.py` so `_SpyAgentLoop` can stand
  in for the singleton at the factory call site. 151/151 PASS post
  hot-fix.
