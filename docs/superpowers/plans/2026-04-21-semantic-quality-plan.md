# Semantic-quality improvements — Plan (post 2026-04-21 session)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift SWE-bench Lite resolved rate from the current ~10% (N=10, noise-dominated) baseline by attacking the two failure modes that infra fixes can't touch: semantic-miss patches (agent emits a syntactic fix that doesn't implement the feature) and patch-emission quality on large files.

**Architecture:** Three independent tracks, each measurable in isolation. Work order reflects lowest-effort → highest-effort. Each track ends with a Docker-graded smoke to confirm lift.

**Tech Stack:** Python (sage-python), pydantic-ai (providers), swebench harness. No Rust changes.

---

## Context (what's already done — don't redo)

Session 2026-04-21 landed on main:
- `efb8afd` CRLF fix, `bcade10` UTF-8 fix (Windows infra — unlocked 0/10 → 1/10)
- `24e51af` patch validator + two-stage repair (code retained, 0 lift in v16)
- `4f90a98` Stage 4 fallback → healthy provider + raise-on-empty (3/5 EMPTYs recovered in v17)
- `8bb923b` docs (v16 post-mortem + v17 results)

Agent `af688789c50e00ce0` in a worktree is applying follow-ups already specified (validator relax to `patch --fuzz=5`, LLM-repair prompt enrichment with source-file snippets). **Check the worktree status before starting this plan** — those tasks may already be committed.

## Track 1 — Generation-quality smoke at N=50

**Why first**: without a statistically meaningful baseline, every subsequent change is wishful thinking. At N=10 variance is ±10pp per task flip — impossible to distinguish a real lift from a lucky run.

### Task 1.1: Run the N=50 smoke

**Files:**
- Invoke: `sage-python/scripts/swebench_n50_smoke.py` (already written)

- [ ] Load env: `set -a && source .env && set +a`
- [ ] Launch in background, tee log:
  ```bash
  python sage-python/scripts/swebench_n50_smoke.py 2>&1 | tee \
    docs/benchmarks/$(date +%Y-%m-%d)-swebench-n50-full.log &
  ```
- [ ] Wallclock ~2-4 h. Don't block on it; come back when notified.
- [ ] When complete, capture the three artifacts under `docs/benchmarks/*-n50-*`.

### Task 1.2: Analyze the N=50 result

**Files:**
- Write: `docs/benchmarks/<date>-swebench-n50-analysis.md`

- [ ] Resolved rate with 95 % binomial CI (Python: `statsmodels.stats.proportion_confint` or `scipy.stats.binomtest`).
- [ ] Per-bucket breakdown: EMPTY, apply-error, unresolved-applied, resolved, timeout.
- [ ] Attribute each failure to: provider-health, malformed-diff, semantic-miss, timeout.
- [ ] Compare buckets to the v13 10-task distribution (proportion test).
- [ ] Decide next track: if `unresolved-applied` > 30%, go Track 3 first. If it's <10%, infra is still dominant, go Track 2 first.

## Track 2 — Emission format change (search-and-replace)

**Why**: unified-diff line numbers are a hallucination trap on large files. astropy-7746 (3375 chars, malformed `@@ -1264,28 +1279,35 @@`) is the canonical case. Switching to search-and-replace blocks sidesteps the problem entirely. Aider / OpenHands / SWE-agent all use this pattern.

### Task 2.1: Failing test for the new extractor

**Files:**
- Create: `sage-python/tests/test_search_replace_extraction.py`

- [ ] Write a test that builds an LLM response containing a search-and-replace block:
  ```
  <<<<<<< SEARCH
  def foo(x):
      return x
  =======
  def foo(x):
      return x + 1
  >>>>>>> REPLACE
  ```
- [ ] Assert that a new helper `_extract_search_replace_blocks(response)` returns `[(file, search, replace), ...]`.
- [ ] Run: `pytest sage-python/tests/test_search_replace_extraction.py -v` — expect FAIL (fn doesn't exist yet).

### Task 2.2: Implement the extractor

**Files:**
- Modify: `sage-python/src/sage/bench/swebench_bench.py` (add module-level helper)

- [ ] Add `_extract_search_replace_blocks(response, repo_dir)` that parses the markers and returns `[(path, search_text, replace_text)]`. The file path is captured from a preceding line like `## File: path/to/x.py` or falls back to searching the repo for a match.
- [ ] Add `_blocks_to_unified_diff(blocks, repo_dir)` that:
  - For each block, reads the actual file.
  - Locates the `search_text` in the file (exact match first; `difflib.SequenceMatcher` fuzzy match if exact fails with confidence ≥ 0.95).
  - Computes the hunk line numbers from the match location.
  - Emits a well-formed unified diff with correct counts.
- [ ] Run the test from 2.1 — should pass.

### Task 2.3: Update the agent task prompt

**Files:**
- Modify: `sage-python/src/sage/bench/swebench_bench.py` `_TASK_TEMPLATE`

- [ ] Change the "Patch Format — Strict" section to require search-and-replace blocks instead of unified diff. Include an example.
- [ ] Add: "The framework converts your blocks to a unified diff automatically — you don't need to count lines or format hunks."
- [ ] Keep the "Mandatory Workflow" (exploration before emitting).

### Task 2.4: Wire the extractor into generate_patches

**Files:**
- Modify: `sage-python/src/sage/bench/swebench_bench.py` `generate_patches`

- [ ] After `_extract_patch` returns empty (no unified diff found), try `_extract_search_replace_blocks` → `_blocks_to_unified_diff`.
- [ ] Record which path produced the patch in `_extraction_method` metadata.
- [ ] Run: `pytest sage-python/tests/test_swebench_bench.py -v` — all existing tests still pass.

### Task 2.5: Commit + smoke

- [ ] Commit with message `feat(bench): search-and-replace patch emission format`.
- [ ] Run N=10 smoke to verify no regression on the infra side.
- [ ] Run N=50 smoke. Compare resolved rate to Track 1.2 baseline.

## Track 3 — Semantic-miss bucket (planner depth + ExoCortex retrieval)

**Why last**: this is the hardest and least predictable. astropy-14182 is the canonical case — SAGE added `header_rows=None` parameter but didn't implement the row-skip logic. The agent had tools, had the issue text, and still emitted a syntactic-only fix. Fixing this requires changes to the agent's reasoning depth, not just its I/O format.

### Task 3.1: Investigate astropy-14182 in isolation

**Files:**
- Create: `docs/audits/2026-04-??-astropy-14182-semantic-miss.md`

- [ ] Pull the v15 agent log for astropy-14182 (`sage-python/logs/run_evaluation/sage-20260421-.../astropy__astropy-14182/run_instance.log` exists from v13; agent trace from v17 if still around).
- [ ] Read: which tools did the agent call? How many? Did it read the test file `astropy/io/ascii/tests/test_rst.py`?
- [ ] Read the failing test. It expects `RST(header_rows=['name', 'unit'])` to parse a table with a unit row. The ground-truth fix is in a specific commit — check `git log --grep header_rows` on astropy upstream.
- [ ] Hypothesis: the agent never looked at the test file, so it didn't know the feature contract. Without the test, `header_rows=None` looks sufficient.

### Task 3.2: Add test-first exploration step to the task prompt

**Files:**
- Modify: `sage-python/src/sage/bench/swebench_bench.py` `_TASK_TEMPLATE`

- [ ] Insert a new mandatory step in "Mandatory Workflow": **"Step 0 (before steps 1-5): find and read the test files that reference the functionality in the issue. Use `grep -RIn 'feature_name' tests/` and read the matching tests in full."**
- [ ] Rationale: tests encode the feature contract. If the agent reads the test BEFORE writing the fix, shallow param-add patches become visibly insufficient.

### Task 3.3: Wire ExoCortex retrieval as a default agent tool

**Files:**
- Check: `sage.agent_loop` tool registration; `search_exocortex` tool already exists.
- Modify: task prompt to mention the tool is available for library-specific documentation.

- [ ] Verify `search_exocortex` is registered for SWE-bench runs (read the run_instance log from v13/v17 to see if it was called).
- [ ] If not registered, add it to the SWE-bench-specific tool list.
- [ ] If registered but unused, add a hint in the task prompt: "For non-trivial fixes, searching the project-wide knowledge base via `search_exocortex` may surface relevant API documentation."

### Task 3.4: Planner budget — 20 steps is too few for large repos

**Files:**
- Investigate: `sage-python/src/sage/singleton.py` (max_steps wiring)

- [ ] Check current S3 `max_steps`: was raised to 20 in `b7ced9d` (2026-04-20 H7 fix).
- [ ] For django-11001 (v17 timed out at 300s), was the agent close to emitting? Check the last 3 agent turns in its log.
- [ ] If budget-exhausted before finalizing, increase to 30 for SWE-bench specifically (override at bench level, not globally).
- [ ] Re-run the failing tasks individually — NOT a full smoke yet.

### Task 3.5: Measure

- [ ] Run N=50 smoke after 3.1-3.4 changes.
- [ ] Compare `unresolved_applied` count to Track 1.2 and Track 2.5 baselines.
- [ ] Target: drop from current ~20-30% `unresolved_applied` to ~10-15%.

## Track 4 — Follow-up housekeeping

- [ ] Revert or formally retire `24e51af` (v16 repair pipeline) if Tracks 2-3 make it redundant.
- [ ] Generate a CLAUDE.md §Benchmarks update with the new N=50 number as the reference pass-rate (supersedes the v15 1/10).
- [ ] Add a `swebench-infra-invariants.md` doc listing the Windows-specific patches (CRLF, UTF-8, fallback, repair) and when each is exercised — easy on-ramp for future contributors.

## Non-goals for this plan

- Training (parked since 2026-04-15).
- Path 6c redesign (pending a separate spec).
- Dataset expansion beyond SWE-bench Lite (Pro / Verified are future).

## Expected outcomes at end of plan

- **N=50 baseline + 95% CI** for resolved rate.
- **Search-and-replace emission** shipped, measured.
- **Semantic-miss attribution** doc + one planner-depth improvement.
- Paper trail in `docs/benchmarks/` for each track.

## Checkpoint policy

After every track, pause + re-evaluate. If the measured lift is below noise floor (Track 1.2 CI half-width), consider whether the next track is worth the complexity before proceeding.
