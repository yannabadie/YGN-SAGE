---
name: April 29 R6.1a EvidenceProducers cycle 6 closure
description: 3-round cgpro VERIFY trail for R6.1a deterministic delta producers. APPROVED 2026-04-29. Cycle-7 default-on flip gate locked.
type: project
originSessionId: b7b56b62-e6ea-4a71-965c-def15a6da3a2
---
# R6.1a EvidenceProducers (cycle 6) — closed 2026-04-29

**Final state**: APPROVED by cgpro round 3, commits `38c0da4e`..`25e604dd` pushed to origin/main.

**Cycle scope**: deterministic RuntimeDelta producers + Tool/Formal/Spec v1 oracles replacing R9 v0 None placeholders. 6 producers (tool / test_parser / diff / formal / code_node / planner). NEW `RunFrame.runtime_deltas` field. 3 LIVE emission points (agent_loop_execution + swebench_diff_verifier + swebench_patch_repair, all `SAGE_ORACLE=1`-gated).

## 3-round VERIFY trail (lessons for future cycles)

**Round 1** (`38c0da4e` initial, +2990/-23 across 54 files): PUSH BACK with 3 blocking findings:
1. `_spec_oracle` lexical fallback still reachable (_INVALIDATION_MARKERS + substring scan kept as primary trigger, structured deltas only corroborating). cgpro: "structured facts only; no substring fallback" was the lock — band-aid mention/reassertion guard not enough.
2. `_formal_oracle` could train on incomplete formal deltas (verifier_id/encoding/solver_status optional in payload schema → bypass to trainable verdict).
3. Generic tool exceptions could become trainable fail without "tied to claimed task output" proof — exact R9 lexical-fallback class on the tool side.

**Round 2** (`19ea317c` fixup, +349/-134 across 10 files): 1 fully closed (lexical scan removed, spec oracle becomes structured-only v1 stub returning None always). 2 producer-side fully closed; oracle defense-in-depth had ONE remaining hole (`_formal_delta_is_complete` checked verifier_id/encoding/solver_status but NOT obligation_id, leaving direct/synthetic RuntimeDelta bypass open). 3 fully closed (fatal_scope payload field with `claimed_task_output | incidental_tool_call | unknown` enum, ToolOracle trains fail only on `claimed_task_output`).

**Round 3** (`426dfb6f` final, +82/-5 across 2 files): obligation_id added to `_formal_delta_is_complete` + 3 direct-RuntimeDelta bypass tests. APPROVE.

## Cycle-7 default-on flip gate (cgpro round 3 lock)

Operational order, NOT to be skipped:
1. R6.1a approved/shipped (DONE 2026-04-29).
2. Gate C — no-spend synthetic ON smoke covering 10 oracle scenarios (already covered in 67 unit tests, no extra work).
3. Gate D — paid SWE-bench Lite N=10 with **throwaway bandit DB** (deferred — API budget gated; needs `runtime_delta_count > 0`, ToolOracle non-abstain present, no FormalOracle without complete obligation evidence, SpecOracle never trains from text substrings, no raw stdout/stderr in summaries, OFF mode `runtime_deltas == ()`).
4. Operator A14 reset checkpoint paired with the flip operation (NOT prerequisite for R6.1a closure; paired with the actual default-on operation).
5. Flip SAGE_ORACLE default-on.
6. Immediate post-flip smoke.

**Critical**: do NOT reset persisted production posteriors as part of R6.1a. That belongs to the default-on operation.

## Methodology insights (apply in future cycles)

1. **Wide cycles deserve wide VERIFY**: 2990 LOC across 54 files = high probability of subtle gaps. cgpro round 1 caught 3 trainable-evidence leaks unit tests didn't. Don't skip cgpro VERIFY on cycles >1000 LOC touching pipeline/runner/oracle.

2. **Defense-in-depth at producer AND oracle**: cgpro pushed back specifically because the producer was safe but the oracle hot path could be bypassed by direct RuntimeDelta construction. Always layer the check at BOTH layers when a payload field is required for a trainable verdict.

3. **Structured-only ≠ keep-substring-as-primary**: when a spec says "structured facts only", a corroborating-deltas-on-top approach is NOT compliance. The lexical path must go.

4. **Multi-round VERIFY is cheap**: round 2 caught a hole round 1 didn't see (because round 1 didn't ask about direct-bypass). Round 3 confirmed APPROVE in <5 minutes. Don't fear multiple rounds — each is fast and adds defense layers.

5. **fatal_scope pattern is reusable**: "scope of failure" payload tag is a generalizable pattern for any deterministic-evidence channel. When in doubt about whether a fatal should train fail, scope it.

## Open R6.1b/c work (cycle 7+)

R6.1b (cycle 7, after default-on flip):
- pytest parser anchoring on actual `=== N passed, M failed in T.Ts ===` line + negative fixtures (docstring containing "1 passed in 0.1s" must NOT match).
- planner producer live integration for structural-only facts (`topology_selected` / `decomposition_applied`).
- ToolOracle incidental-fatal-does-not-suppress-parser-pass cleanup (compute scoped_fatals before scope filter, parser pass + incidental fatal should pass not abstain — non-blocking, safe direction).

R6.1c (cycle 8): per-(producer, delta_kind) payload schema cleanup (currently per-producer, not per-kind).

## Final R6.1a code map

- `sage/runtime/evidence/` — 11 files, 1187 LOC src
  - `delta.py` — RuntimeDelta frozen dataclass + central kind/polarity tables
  - `errors.py`, `payloads.py` — schema + hash
  - `producers/{tool,formal,code_node,diff,parsers,planner}.py` — 6 producers
- `sage/runtime/oracle/_oracles.py` — Tool/Formal/Spec v1 (replaces R9 None placeholders)
- 3 LIVE emission points: agent_loop_execution.py, bench/swebench_diff_verifier.py, bench/swebench_patch_repair.py
- `RunFrame.runtime_deltas` field added (Q4.a)
- 22 fixture round-trip JSON pairs + 67 unit/integration tests + 13 Gate A regression tests (3 obligation_id direct + 6 formal completeness + 4 tool fatal_scope)
