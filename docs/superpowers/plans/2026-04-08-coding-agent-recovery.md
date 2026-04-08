# Coding Agent Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current SWE-bench preparation work into a credible coding-agent loop, refresh the public benchmark story to match April 8, 2026 reality, and only then resume Path 6 training work.

**Architecture:** Keep the current pipeline direction. The codebase already has the right building blocks: `execute_bash`, ToolForge, repo checkout for SWE-bench, adaptive bypass, and a multi-turn single-agent tool loop. The immediate problem is the connection between them: coding tasks still bypass into weak behavior too often, tool use is under-signaled, and the public docs lag behind the current evidence. Fix the execution loop first, then publish updated evidence, then reopen the training branch.

**Tech Stack:** `sage-python` pipeline/boot/bench modules, pytest, SWE-bench harness, benchmark artifacts under `docs/benchmarks`, Claude project memory, superpowers planning workflow.

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `sage-python/src/sage/pipeline.py` | Routing, adaptive bypass, single-agent tool-call loop, execution tracing | Exists, active hotspot |
| `sage-python/src/sage/boot.py` | Tool registration, `execute_bash`, coding-agent capabilities exposed at boot | Exists, active hotspot |
| `sage-python/src/sage/bench/swebench_bench.py` | SWE-bench prompt, repo checkout at `base_commit`, metadata capture | Exists, active hotspot |
| `sage-python/tests/test_pipeline.py` | Regression coverage for bypass, routing, tool-loop execution | Exists, expand |
| `sage-python/tests/test_agent.py` | Direct tool-call loop behavior in the agent layer | Exists, expand if needed |
| `sage-python/tests/test_agent_loop.py` | Retry / loop semantics for multi-turn execution | Exists, expand if needed |
| `README.md` | Public project positioning and benchmark claims | Exists, stale against April 8 state |
| `CLAUDE.md` | Contributor-facing current state and benchmark snapshot | Exists, stale against April 8 state |
| `docs/benchmarks/results.md` | Human-readable benchmark summary | Exists, stale against April 8 state |
| `docs/benchmarks/2026-04-08-swebench-lite-diagnostic.json` | New diagnostic artifact for the repaired coding-agent loop | Create |
| `docs/benchmarks/2026-04-08-coding-agent-status.md` | New short narrative benchmark/status note for April 8 | Create |

---

### Task 1: Lock the SWE-bench coding loop with regressions

**Files:**
- Modify: `sage-python/src/sage/pipeline.py`
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-python/src/sage/bench/swebench_bench.py`
- Modify: `sage-python/tests/test_pipeline.py`
- Modify: `sage-python/tests/test_agent.py`
- Modify: `sage-python/tests/test_agent_loop.py`

The April 7-8 diagnostic already narrowed the root cause: SWE-bench tasks route too softly, do not reliably use tools, and behave like one-shot patch generators. Do not add heuristics without evidence. Add regressions first, then only the smallest changes needed to make the existing loop behave like a coding agent.

- [ ] **Step 1: Add a failing regression for the current benchmark failure mode**

Target behavior to encode:
- SWE-bench benchmark tasks must not silently end with `0` tool calls.
- The execution trace must expose tool-call count, executed commands, and total turns.
- A complex repo-fix task must not look indistinguishable from a plain one-shot code generation request.

Run:

```bash
cd sage-python
python -m pytest tests/test_pipeline.py tests/test_agent.py tests/test_agent_loop.py -q
```

Expected before the fix: at least one new/updated regression fails and demonstrates the current gap.

- [ ] **Step 2: Tighten the benchmark execution path without inventing a parallel stack**

Implement the minimum change set in the existing flow:
- preserve the current single-agent tool-call loop in `pipeline.py`
- ensure SWE-bench benchmark execution reaches the tool-capable path
- keep `execute_bash` visible and usable from the first coding turn
- capture loop observability in benchmark metadata so the next diagnosis is data-driven

The preferred direction is to reuse the current path, not to build a second ad hoc benchmark-only agent.

- [ ] **Step 3: Re-run focused tests and verify the loop is observable**

Run:

```bash
cd sage-python
python -m pytest tests/test_pipeline.py -q
python -m pytest tests/test_agent.py tests/test_agent_loop.py -q
```

Expected after the fix:
- tests pass
- at least one regression explicitly asserts tool-loop activity or turn-count behavior

- [ ] **Step 4: Commit**

```bash
git add sage-python/src/sage/pipeline.py sage-python/src/sage/boot.py sage-python/src/sage/bench/swebench_bench.py sage-python/tests/test_pipeline.py sage-python/tests/test_agent.py sage-python/tests/test_agent_loop.py
git commit -m "fix(bench): make SWE-bench use the real coding-agent loop"
```

---

### Task 2: Re-run the SWE-bench Lite diagnostic and save artifacts

**Files:**
- Output: `docs/benchmarks/2026-04-08-swebench-lite-diagnostic.json`
- Output: temporary predictions JSONL and metadata generated by the runner

This task is not about claiming SOTA. It is a credibility gate. The repaired loop must show different behavior from the March one-shot baseline before any broader benchmark campaign.

- [ ] **Step 1: Run the repaired 5-task pilot in generate-only mode**

Run:

```bash
cd sage-python
python -m sage.bench --type swebench --dataset lite --limit 5 --generate-only
```

Expected:
- predictions file produced
- metadata file produced
- tool usage appears in metadata
- the run is no longer "100% S2 routing, 100% bypassed, 0 tools used"

- [ ] **Step 2: Convert the pilot into a tracked benchmark artifact**

Save a compact JSON artifact under `docs/benchmarks/2026-04-08-swebench-lite-diagnostic.json` with:
- task ids
- routed system / path used
- tool call count
- turn count
- patch produced or not
- any benchmark error field

This file becomes the reproducible evidence for the next session.

- [ ] **Step 3: If Docker/Linux evaluation is available, run official grading**

Run:

```bash
python -m sage.bench --type swebench --dataset lite --limit 5
```

Expected:
- official harness runs cleanly
- resolved count may still be low, but patch quality should improve over the March "apply errors everywhere" baseline

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/2026-04-08-swebench-lite-diagnostic.json
git commit -m "results(bench): add April 8 SWE-bench Lite diagnostic artifact"
```

---

### Task 3: Refresh the public benchmark story

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/benchmarks/results.md`
- Create: `docs/benchmarks/2026-04-08-coding-agent-status.md`

The current public docs still describe the project as if BigCodeBench were `37.8%` and MASBENCH were a single headline delta. April 8 state is stronger and more nuanced: BigCodeBench improved to `45.9%`, MASBENCH breadth is the only statistically significant axis on the current sample, and SWE-bench work has moved from infrastructure to loop-quality diagnosis.

- [ ] **Step 1: Update the benchmark numbers and caveats**

Refresh the public-facing docs so they consistently state:
- BigCodeBench Hard Instruct: `45.9% (68/148)` as the latest recorded result
- MASBENCH: breadth is the significant win on the current analysis, not a blanket "all axes improved"
- SWE-bench: infrastructure is working, coding-agent loop is the next bottleneck, and current/last pilot results are still diagnostic rather than claim-worthy

- [ ] **Step 2: Add a short April 8 status note**

Write `docs/benchmarks/2026-04-08-coding-agent-status.md` with:
- what improved since March
- what remains blocked
- what the next hard evidence gate is

- [ ] **Step 3: Re-read for honesty and consistency**

Verify there is no stale contradiction across:
- `README.md`
- `CLAUDE.md`
- `docs/benchmarks/results.md`
- `docs/benchmarks/2026-04-08-coding-agent-status.md`

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md docs/benchmarks/results.md docs/benchmarks/2026-04-08-coding-agent-status.md
git commit -m "docs: refresh benchmark story for April 8 state"
```

---

### Task 4: Resume Path 6 / Phase C only after the coding loop clears the gate

**Files:**
- Reference: `docs/superpowers/plans/2026-03-30-sage-v2-completion.md`
- Reference: `sage-python/scripts/verl/train_phase_c_custom.py`
- Reference: `sage-python/scripts/verl/train_topology_phase_c.sh`

Phase C remains strategically important, but it is not the next unblocker. The current memory says Phase C / GRPO is blocked, while Phase D.3 SWE-bench is explicitly next. Follow that sequencing.

- [ ] **Step 1: Define the gate**

Do not reopen training until all of the following are true:
- the 5-task SWE-bench pilot uses tools
- the loop is multi-turn and observable
- the public docs match current evidence

- [ ] **Step 2: Revisit the Phase C plan only after Task 1-3 land**

When the gate is cleared, resume from:

```bash
cd sage-python
bash scripts/verl/train_topology_phase_c.sh
```

or the documented custom fallback:

```bash
cd sage-python
python scripts/verl/train_phase_c_custom.py
```

Expected:
- Path 6 work resumes with a credible runtime target
- benchmark work and training work stop competing for the same attention budget

- [ ] **Step 3: Commit only when a real training artifact exists**

```bash
git add sage-python/scripts/verl/train_phase_c_custom.py sage-python/scripts/verl/train_topology_phase_c.sh docs/benchmarks/
git commit -m "feat(training): resume Phase C after coding-loop validation"
```

---

## Exit Criteria

- SWE-bench benchmark execution no longer looks like one-shot patch generation with zero tool use.
- A reproducible April 8 diagnostic artifact exists under `docs/benchmarks/`.
- `README.md`, `CLAUDE.md`, and `docs/benchmarks/results.md` tell the same story.
- Path 6 / Phase C stays blocked until the coding-agent loop is empirically credible.

## Priority Order

1. Task 1: coding-agent loop
2. Task 2: reproducible SWE-bench evidence
3. Task 3: public benchmark honesty
4. Task 4: Path 6 / Phase C resume
