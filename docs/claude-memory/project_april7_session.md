---
name: April 7, 2026 Sessions — Improvement Loop + Strategic Plan + Phase A
description: 2 sessions — 69→0 failures, P0 solver, Codex removal, strategic plan, then Phase A complete (A2A, ToolForge E2E, 17 test fixes)
type: project
---

## Session 1 Results (April 7, 2026 — Morning)

### 10 commits pushed to main (632d3a6)

1. `6ef60cf` — 14 audit fixes (69→0 test failures, sandbox bash blocklist, race condition, boot crashes)
2. `3b19359` — P0 solver equality chains via union-find (10/14 → 14/14 vars)
3. `1b1de51` — Remove Codex CLI from model source of truth — all models via API
4. `bdb8064` — Formalizer prompt with 2 iGSM-style few-shot examples
5. `8b83a10` — Answer extraction via word-overlap scoring (no regex)
6. `63337aa` — Test fix for codex→openai tier
7. `dc4cbcb` — Domain inference (math phrases), DeepSeek formalizer pin, model assigner respects pre-assigned model_id
8. `a1f3dcf` — Remove formal_solver for S1 (reverted in next commit)
9. `12fb96b` — Hybrid formal_solver: Rust exact solving + LLM CoT fallback
10. `632d3a6` — Docs API mismatch fix (boot() → boot_agent_system())

## Session 2 Results (April 7, 2026 — Afternoon)

### Phase A Complete — commit 23ab78b

**A.2 A2A protocol**: Migrated a2a_server.py from phantom v1.0 API to a2a-sdk 0.3.25.
- 10 import paths, TaskUpdater signature, context.message → context.request.message
- 6 A2A tests unblocked (were all skipped)

**A.3 ToolForge E2E**: Proved gap→synthesis→registration→use pipeline.
- Fixed bug: forge.py called Tool object as function (not callable)
- Added Tool.run() method, 4 E2E tests

**A.4 Flaky tests**: Fixed 17 failures + 3 errors across 8 files.
- Codex CLI removal fallout (3 tests), SSL bypass violations (2 files)
- Env leaks: os.environ → monkeypatch (2 files), boot() → boot_agent_system()
- _provider_pool → provider_pool attribute rename

### Key Metrics After Both Sessions

| Metric | Session 1 Start | Session 1 End | Session 2 End |
|--------|----------------|---------------|---------------|
| Python tests | 1940p, 69f | 1991+p, ~4 flaky | **2001p, 0f** |
| A2A tests | 6 skipped | 6 skipped | **6 passed** |
| ToolForge | unit tests only | unit tests only | **4 E2E + units** |

### Strategic Plan Progress

Plan: `~/.claude/plans/witty-honking-sky.md`

- **Phase A — Solidify** ✅ COMPLETE
  - A.1 Docs ✅ | A.2 A2A ✅ | A.3 ToolForge E2E ✅ | A.4 Flaky tests ✅
- **Phase B — Prove delta** ← NEXT
  - B.1 BigCodeBench official submission
  - B.2 Ablation N≥50 with McNemar + Cohen's d
  - B.3 Evolution statistical validation
  - B.4 MASBENCH 5 axes with CIs

**How to apply:** Start next session with B.1 (BigCodeBench). Phase A is complete and pushed.
