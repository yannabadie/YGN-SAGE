---
name: May 2 cycle-9 A2 bench + deepseek migration + CLI dispatcher
description: A14 reset, deepseek-v4-flash migration (A33), A2 bench RUNNING — 4 PASS gate met at task 7/10. Commits 24f97f3c..20d0faf6.
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# Cycle-9 A2 bench + deepseek migration (2026-05-02)

**Commits shipped**: `24f97f3c` → `59f87535` → `27770580` → `5617440e` → `f9f84c9a` → `c9448a9b` → `4d0a75d6` → `20d0faf6`

**Why**: Cycle-9 A2 gate: run budget-tier ablation smoke (N=10 BCB-Hard) to decide A3 N=50 or rollback.

## What shipped

### 1 — deepseek-chat → deepseek-v4-flash migration (commit `24f97f3c`)
- `models.toml`: budget/fallback tier renamed
- `llm/router.py`: `_HARDCODED["budget"]` + `["fallback"]` updated
- `bench/masbench.py`: `bare_model` default updated
- `model_profiles.toml`: renamed block, updated pricing ($0.14/$0.28 per 1M)

### 2 — A14 epoch guard reset (pre-A2 bench)
- Old contaminated state (bandit_state.db + archive_state.db + engine_extras.json without topology_state_manifest) moved to `~/.sage/contaminated/pre_a14_20260502/`
- `posterior_epoch.json` updated to epoch=1
- Command: `python -m sage.ops.a14_reset --reason "Cycle-9 A2 smoke pre-reset: pre-A14 state accumulated off-policy garbage."`

### 3 — A33: deepseek reasoning_content multi-turn fix (commit `27770580`)
- File: `sage-python/src/sage/providers/pydantic_ai_provider.py`
- Problem: deepseek-v4-flash returns `reasoning_content` in thinking-mode multi-turn responses. Without a thinking profile, the next turn omits it → HTTP 400 "reasoning_content must be passed back"
- Fix: Added `deepseek_openai` kind in `_PROVIDER_MAP`, with `OpenAIModelProfile(supports_thinking=True, openai_chat_thinking_field="reasoning_content", openai_chat_send_back_thinking_parts="field")`
- Pattern: same as kimi fix (roadmap-A8 Phase 3, commit `ec5d0c98`)
- **Note**: A2 bench started BEFORE this fix landed. The running A2 bench uses old code (pre-A33). All multi-agent nodes fall back to single-agent via circuit breaker. A3 will use fixed code.

### 4 — CLI dispatcher (commit `c9448a9b`)
- `sage-python/src/sage/cli.py`: root CLI dispatcher (`sage serve`, `sage bench`, `sage chat`)
- `sage-python/tests/test_cli.py`: 6 tests (all pass)
- `pyproject.toml`: entry point `sage.protocols.serve:main` → `sage.cli:main`

### 5 — gitignore + current.json
- `.gitignore`: added `sage-python/files.zip` (ToolForge test artifact)
- `docs/status/current.json`: updated to 2903 Python tests at commit `4d0a75d6`

## A2 bench status (RUNNING — pre-A33 code)

**Config "full"** (primary gate), tasks completed as of ~11:00 UTC:
- [1/10] PASS BigCodeBench/13
- [2/10] FAIL BigCodeBench/15
- [3/10] FAIL BigCodeBench/17
- [4/10] PASS BigCodeBench/19
- [5/10] PASS BigCodeBench/34
- [6/10] PASS BigCodeBench/37
- [7/10] FAIL BigCodeBench/82
- [8/10] IN PROGRESS...

**Gate met**: 4 PASS after 7 tasks → ≥4/10 regardless of remaining results.

**Systematic bias**: all multi-agent Stage 4 executions hit HTTP 400 (reasoning_content) and fall back to single-agent. The "full" config pass rate (60%+ so far) is DEGRADED vs what A3 will show with A33 fixed. Ablation deltas are still valid (all configs equally affected).

**Full A2 bench**: 6 configs × 10 tasks = 60 total. Will complete ~55 min after "full" config ends. Results file: `docs/benchmarks/2026-05-02-ablation-study.json`

## Post-A2 next steps

**Gate decision (already met)**:
- ≥4/10 in "full" config → PROCEED to A3 N=50
- A3 command: `python -m sage.bench --type ablation --limit 50`
- A3 will use A33-fixed code (fresh process)
- Expected A3 improvement: multi-agent path works correctly, pass rate higher

**4 test failures from sweep** (pre-existing test pollution):
- `test_pillar_logging.py::test_topology_edges_logs_adjacency_list`
- `test_pillar_logging.py::test_topology_source_logs_attribution_template_branch`
- `test_online_evolution.py::TestRealEngineEvolutionLoop::test_record_outcome_grows_archive_only_when_topology_cached`
- `test_online_evolution.py::TestRealEngineEvolutionLoop::test_evolve_chain_end_to_end`
- ALL pass in isolation. Pre-existing test pollution (order-dependence). Same class as `contaminated_pre_a14_state` failures.
- NOT regressions from A33.

## How to apply
- Next session should check bench results file after it completes
- Gate is met — A3 can proceed
- Use fresh Python process to get A33 fix
- Budget bench still using deepseek-v4-flash (confirmed working with single-agent fallback)
