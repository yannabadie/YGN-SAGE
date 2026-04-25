---
name: April 19 (evening) — Bypass-audit architectural sweep
description: 7 commits in one night closing "Rust built, Python doesn't call it" bypasses. G-series + H1-H6 + methodology doc. Fully validated empirically.
type: project
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
# April 19, 2026 (evening) — Bypass-audit sweep

**Duration:** ~6h, one autonomous session
**Commits:** 7 pushed to origin/main (45dd81b → aa348e1)
**Theme:** Rust architectural APIs that exist but Python doesn't call at runtime.

## Commits

- `55043e6` — Audit TL;DR cites true 47-53% rate, not proxy 73%
- `c905d06` — **G-series**: CompositeWriteGate wired into phases/act.py (3 memory writes gated, pipeline-scoped state)
- `0b00abd` — Factory regression test (pins gate wiring so it cannot silently un-wire)
- `02130e8` — Audit §2.5 + §5.1 sync (gate "bypassed" → "wired", Controller TODO documented)
- `2cd840e` — **H1**: engine.should_evolve()/evolve() wired into LEARN stage
- `dc51976` — **H4**: cache_topology() after generate — **caught a silent bypass in H1's own code**. Without it, record_outcome no-ops on MAP-Elites archive. Empirically proven.
- `27a9a4c` — **H5**: write_gate wired onto singleton AgentLoop (single-agent bypass path — G-series only caught multi-node)
- `aa348e1` — **H6**: _on_drift wired onto singleton + `docs/audits/bypass-patterns.md` methodology catalog

## Key learning: the bypass pattern

Every bypass looked identical:
1. Rust impl exists + PyO3-exposed
2. Python imports the symbol
3. Unit tests on the Rust class pass
4. **No runtime code actually invokes it** during pipeline.run()
5. Python call site has `if X is not None: X.do_work()` — the None guard silently defaults to off

The insidious part: every test keeps passing, every lint keeps passing, architecture.md keeps claiming "wired", but the feature does nothing.

## The advisor prediction that paid off

On H1, advisor warned: *"If record_outcome silently fails, you'd wire engine.evolve() perfectly and observe zero effect."* H4 empirical poke proved it — 8 outcomes with diverse IDs → cell_count stayed at 0. Without H4, H1 would never fire on a real run.

## Methodology now catalogued

`docs/audits/bypass-patterns.md` — systematic checklist:
- PyO3 surface inventory
- Runtime-call-site grep (distinguish from imports + factory)
- TWO-path check (multi-node factory vs single-agent singleton)
- Empirical validation against real Rust state (not mocks alone)
- Re-audit after each wiring commit (compound dependency chain)

Plus red-flag commit-message phrases and false-positive list (RustQualityEstimator deliberately stale, HardwareProfile trivially optional).

## Remaining queued (for 2026-04-20+ plan)

- `max_steps` / `stall_cap` / `tools` on singleton AgentLoop — asymmetry flagged but not verified
- `TopologyController` Rust port — 6 decision paths in Python, zero in Rust, Critical Directive #1 violation
- `RustEntityGraph` ↔ `CausalMemory` consolidation — refactor, not bypass

## Why I stopped at 7 commits

Self-imposed criterion: 2 commits + 1 deferral = session done. Exceeded because the pattern compounded naturally (H1 → H4 → H5 → H6 all same shape). Stopped when the next fix would be repetitive rather than additive. Controller port deliberately deferred — multi-session scope, requires dedicated sprint.

## How to apply

Before declaring any "wires X" commit done, run the `bypass-patterns.md` checklist. Mock-only tests never catch the silent-None bypass.
