---
name: April 20 — Rust-First plan complete in 1 session
description: All 12 items + 1 follow-up (1.4a) of the 2026-04-20 Rust-First plan shipped in one autonomous session; RustTopologyController Rust-primary now operational
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
**Plan `docs/superpowers/plans/2026-04-20-rust-first-plan.md` — COMPLETE.**
Budget: 6-8 sessions. Actual: 1 session (authorized by user mid-session:
"Je pense que tu peux faire tout le plan en une seule session").

## Commits landed (chronological)

| Item | Commit | Summary |
|---|---|---|
| 1.1 max_steps singleton | `b7ced9d` | `{1:5, 2:10, 3:20}` scaling on bypass path (H7) |
| session-close 1.1 | `70f9cbe` | SHA backfill |
| 1.2 stall_cap singleton | `0b5a272` | `max_steps-1 if >5 else 0` mirror (H8) |
| 1.3 tools filter | `e5e3811` | False positive, doc-only |
| 1.4 + 1.4a archive growth | `c65659b` | Found+fixed H9 (descriptor-keyed id) + H10 (template branch missing cache_topology). Helper `_apply_topology_budget_and_cache` consolidates cache + budget check across all four topology-building branches |
| 1.5 PyO3 inventory | `f6193eb` | 47 pyclass audited, 0 new bypasses, 13 documented false positives |
| 1.6 ADR-011 finalize | `d398196` | Singleton-vs-factory rule + MOC index refresh ADRs 7-11 |
| 2.1 scaffold | `152fe5e` | `RustTopologyController` + `RustAdaptationDecision` pyclasses |
| 2.2 path 1 | `3ee6b5f` | `check_empty_error_reroute` + `is_empty_or_error` helper + regex crate |
| 2.3 paths 2-3 | `84b1c1f` | `check_quality_cascade` + `is_in_gate_band` |
| 2.4 paths 4-5 | `6684665` | `check_parallel_inconsistency` + `check_importance_prune` (pre-computed scores) |
| 2.5 path 6 | `b1d75c9` | `check_emergent_spawn` + `detect_emergent_subtask` (three regex patterns) |
| 2.6 finalize | `e26cd7b` | Python delegation wiring, `__setattr__` mirror, ADR-012, architecture.md + CLAUDE.md |

## Why this worked in 1 session

1. **User authorized it explicitly.** The "one item per session" rule
   came from the plan's conservative pacing; user lifted it mid-flight.
2. **Phase 1 items chained cleanly.** 1.1 → 1.2 → 1.3 → 1.4/1.4a → 1.5
   → 1.6 each built on the prior, all in `pipeline.py` or docs.
3. **Phase 2 was incremental port.** Each path had its own commit
   with 20-sample equivalence test. Rust rebuild via `maturin develop
   --release` took ~16s per cycle.
4. **Advisor caught 2 real risks mid-flight.** H4-pattern on 1.1
   (`max_steps` mutation). H10 template-branch-bypass caught when my
   test forced only the engine branch; expanded fix to cover all four
   topology-building branches.
5. **Empirical tests over mocks.** Every Phase 2 commit had a real-
   `sage_core` equivalence test (`_HAS_SAGE_CORE` gated).

## Surprises / findings

- **H9 + H10 were REAL live bypasses** — not anticipated by the plan
  (plan said 1.4 might find bypasses; it did, worse than expected).
  Pipeline archive stayed at 0 cells in production-dominant path
  regardless of quality → SA-3 online evolution claim was materially
  broken. Fixed in `c65659b`.
- **User feedback mid-session:**
  - "N'utilise pas litellm nous l'avons remplacé par pydanticAI
    déjà." → `feedback_no_litellm.md`, verified migration 2026-04-18.
  - "Tu devrais lire doc et le vault obsidian" → read plan, spec,
    bypass-patterns.md, ADR-011 placeholder, Changelog.
  - "Qu'est-ce que c'est que ça? 'TODO: implement robust retry
    logic' Je ne veux pas de todo en plein milieu du rust..." →
    refactored test fixtures to `const LLM_OUTPUT_*` named constants
    with header comment. Saved as `feedback_no_todos_in_test_strings`.
- **ADR-012 scope divergences documented honestly.** Python didn't
  shrink to 50 lines (optimistic plan wording); helpers stayed Python
  because embedder/SmtVerifier/topology-graph accessors are Python-
  held. The Critical-Directive-#1 goal ("decisions live in Rust") is
  satisfied; "Python topology_controller.py becomes a thin wrapper"
  is not (~730 lines).

## State after session

- **Rust tests:** 478/478 PASS (was 441 at session start; +37 for
  RustTopologyController module).
- **Python tests:** 1939 passed (was 1927 at session start; +12 new).
  5 errors + 1 failure pre-existing asyncio-fixture pollution from
  session 1 baseline, not caused by any commit this session.
- **Bypass-patterns catalog:** 10 bypasses total across the G + H
  sweeps (G-series, H1, H4, H5, H6, H7, H8, H9, H10 — plus 1.3 false
  positive documented).
- **Architecture claims:** SA-3 online evolution now *empirically
  validated* at pipeline level (was previously validated only at
  engine level, hence H9+H10 latent). TopologyController Rust-primary
  closes the last Critical-Directive-#1 violation catalogued in
  `2026-04-18-astropy-14995-decision-path.md` §5.1.

## What's next (not this session)

- Real SWE-bench smoke to complement the pipeline-level archive-growth
  validation. Archive grows in the integration test; production run
  would confirm it grows on realistic tasks and `should_evolve` fires
  at 5+ outcomes (may need larger limit than `--limit 5`).
- The F-series + F7 routing follow-ups + Sprint 5 ablation execution
  from the prior plan are still pending (unrelated to this Rust-First
  plan).
- Consider removing `_evaluate_and_decide_legacy` after 2-3 sessions
  of Rust-primary stability (ADR-012 notes this as deferred).
