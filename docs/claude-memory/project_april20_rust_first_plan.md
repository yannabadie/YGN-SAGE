---
name: April 20 (AM) — Rust-First Plan written, session closing
description: Strategic plan post G+H-series written to docs/superpowers/. Phase 1 audits + Phase 2 Controller port, ~6-8 sessions budget, designed for autonomous fresh-session execution.
type: project
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
# April 20, 2026 (AM) — Rust-First architectural completion plan

**Context.** Planning session AFTER the 2026-04-19 bypass sweep. User requested a detailed plan + framework to avoid context loss across future sessions.

## Output

Two docs under `docs/superpowers/` form the full package:

- **Spec:** `docs/superpowers/specs/2026-04-20-rust-first-plan-design.md` — 5 sections (Context / Phase 1 / Phase 2 / Session-hygiene protocol / Bootstrap). Approved via superpowers:brainstorming skill.
- **Plan:** `docs/superpowers/plans/2026-04-20-rust-first-plan.md` — exec-ready checklist. Each of the 12 items has touch points, done-when criteria, commit template. The **progress tracker table** at the top is the visible state.

## Two phases (menu, each phase ~1-2 sessions per item)

**Phase 1 — Audit-Complete (weeks 1-2, 4 sessions)**
- 1.1 max_steps singleton audit
- 1.2 stall_cap singleton audit
- 1.3 tools filter singleton audit (pre-analysis first — may be a false positive)
- 1.4 MAP-Elites archive growth smoke (empirical validation of H4 on real run)
- 1.5 PyO3 inventory sweep (find bypasses we missed)
- 1.6 ADR-011 Singleton vs Factory Asymmetry

**Phase 2 — TopologyController Rust port (weeks 3-4, 6 sessions)**
- 2.1 Rust scaffold + PyO3 + constants
- 2.2–2.5 Port 6 decision paths, 1-2 per commit, Rust-vs-Python equivalence tests
- 2.6 Finalize + ADR-012

## Session-hygiene protocol (critical)

Documented in both spec + plan files. Core rules:
- Max 1-2 commits per session
- Session-start routine: read plan → read bypass-patterns.md → git log → state next item
- Session-close routine: push, update plan file, update MEMORY.md, update Obsidian if major milestone, kill zombie shells
- Golden rule: "If a fact isn't in a file at end of session, it ceases to exist in the next."

## Bootstrap for a fresh session

A new Claude session just needs to be told: *"Execute the plan at `docs/superpowers/plans/2026-04-20-rust-first-plan.md`."*

The plan file opens with a "IF YOU ARE A FRESH SESSION, READ THIS FIRST" block that triggers the startup routine. No conversation history needed.

## How this memory applies

When a future session asks "what's next?" or "continue the work", point them to the plan file. The plan file supersedes any earlier MEMORY.md "Next" line that may have drifted.
