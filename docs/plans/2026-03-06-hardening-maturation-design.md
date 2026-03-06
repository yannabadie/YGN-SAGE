# YGN-SAGE Hardening & Maturation Design

**Date:** 2026-03-06
**Author:** Yann Abadie + Claude Opus 4.6
**Status:** Approved
**Trigger:** Dual expert review (structured pragmatic + destructive) verified against actual codebase

## Context

Two independent expert reviews identified issues in YGN-SAGE. All claims were rigorously verified against the codebase. 4 bugs were confirmed in `agent_loop.py`, plus medium-priority improvements for CI/CD, config externalization, observability, and cleanup.

## Section 1: Agent Loop Bug Fixes

### Bug 1 — Self-brake memory gap

**File:** `sage-python/src/sage/agent_loop.py:351-357`
**Problem:** When CGRS self-brakes, `working_memory.add_event("ASSISTANT", content)` at line 357 is never reached because `break` is at line 355. Braked responses are lost from working memory.
**Fix:** Add `self.working_memory.add_event("ASSISTANT", content)` before the `break`.

### Bug 2 — Shared retry counter

**File:** `sage-python/src/sage/agent_loop.py:117`
**Problem:** Single `_prm_retries` governs both S3 PRM retries (line 259) and S2 AVR retries (line 291). Shared state makes retry semantics confusing and counter can carry state across phases.
**Fix:** Split into `_s3_retries` and `_s2_avr_retries` with independent counters and independent max values (`_max_s3_retries`, `_max_s2_avr_retries`). Matches SOTA evaluator-reflect-refine patterns (AWS, Google ADK LoopAgent).

### Bug 3 — Heuristic-only metacognition in loop

**File:** `sage-python/src/sage/agent_loop.py:151`
**Problem:** Loop calls sync `assess_complexity()` (heuristic-only). `AgentSystem.run()` in `boot.py:51` calls async `assess_complexity_async()` (LLM-based). Direct `loop.run()` callers never get LLM routing.
**Fix:** Call `await self.metacognition.assess_complexity_async(task)` in the loop. Log `routing_source=llm|heuristic` in event stream.

### Bug 4 — S2 AVR first-block-only

**File:** `sage-python/src/sage/agent_loop.py:288`
**Problem:** Only `code_blocks[0]` is validated. Multi-block responses have subsequent blocks unvalidated.
**Fix:** Validate the last code block (most likely complete solution). Log which block index was validated.

## Section 2: CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

3 jobs triggered on push/PR to master:

1. **`rust`** — `ubuntu-latest`, Rust stable: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test` in `sage-core/`
2. **`python-sage`** — `ubuntu-latest`, Python 3.12: `pip install -e ".[all,dev]"`, `ruff check src/`, `pytest tests/ -v` in `sage-python/`
3. **`python-discover`** — `ubuntu-latest`, Python 3.12: `pip install -e .`, `pytest tests/ -v` in `sage-discover/`

No Windows job initially (solana_rbpf/wasmtime cross-compile complexity). No maturin build in CI (sage_core has Python fallbacks).

## Section 3: Cleanup + License

- `.gitignore`: Add `research_journal/H*.json`, delete `research_journal/deep_self_awareness_report.md`
- Delete `memory-bank/` (7 stale Cline/Roo Code files)
- `sage-core/Cargo.toml`: Add `license = "Proprietary"` to match README
- `conductor/tracks/asi_convergence/plan.md`: Rewrite aspirational language as factual engineering tasks

## Section 4: Model Router Config Externalization

**New file:** `sage-python/config/models.toml`

```toml
[tiers]
fast = "gemini-3.1-flash-lite-preview"
mutator = "gemini-3-flash-preview"
reasoner = "gemini-3.1-pro-preview"
codex = "gpt-5.3-codex"
codex_max = "gpt-5.2"
budget = "gemini-2.5-flash-lite"
fallback = "gemini-2.5-flash"

[defaults]
temperature = 0.7
```

Loading priority (highest wins):
1. Env var `SAGE_MODEL_{TIER}` (e.g., `SAGE_MODEL_FAST=gemini-2.5-flash`)
2. `models.toml` file (searched: `./config/`, `~/.sage/`, package default)
3. Hardcoded fallback in `router.py` (current behavior, unchanged)

Zero breaking changes to `ModelRouter.get_config()` API.

## Section 5: Observability Event Schema

Versioned event schema replacing ad-hoc kwargs:

```python
@dataclass
class AgentEvent:
    schema_version: int = 1
    type: str                         # PERCEIVE, THINK, ACT, LEARN
    step: int
    timestamp: float
    latency_ms: float | None = None
    cost_usd: float | None = None
    tokens_est: int | None = None
    model: str | None = None
    system: int | None = None         # 1, 2, or 3
    routing_source: str | None = None # "llm" or "heuristic"
    validation: str | None = None     # s2_avr_pass, s2_avr_fail, etc.
    meta: dict[str, Any] = field(default_factory=dict)
```

Changes:
- Replace ad-hoc `_emit()` kwargs with `AgentEvent` construction
- Every event tagged with `schema_version=1`
- Dashboard detects schema version and adapts
- Per-step cost tracking (not just cumulative)
- `routing_source` emitted on PERCEIVE events

## Task Summary

| # | Category | Task | Priority |
|---|----------|------|----------|
| 1 | Bug | Self-brake memory gap fix | High |
| 2 | Bug | Split retry counters (S2/S3) | High |
| 3 | Bug | Async metacognition in loop | High |
| 4 | Bug | S2 AVR validate last block | High |
| 5 | Observability | AgentEvent schema dataclass | Medium |
| 6 | Observability | Migrate _emit() to structured events | Medium |
| 7 | Config | Create models.toml + loader | Medium |
| 8 | Config | Wire ModelRouter to TOML + env overrides | Medium |
| 9 | CI/CD | GitHub Actions workflow | Medium |
| 10 | Cleanup | Gitignore + delete stale artifacts | Low |
| 11 | Cleanup | License alignment Cargo.toml | Low |
| 12 | Cleanup | Conductor ASI language cleanup | Low |

Constraints: All 265 existing tests must pass. Each task gets TDD + commit.
