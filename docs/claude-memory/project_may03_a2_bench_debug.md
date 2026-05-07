---
name: May 3 A2 bench debug — BCB tool loop root cause + fixes + v6 final results
description: A2 v1-v6 debugging history, BCB tool loop root cause, v6 final 3/10 PASS gate MISSED, two new failure modes
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# May 3 — A2 Bench Debug + Doc Updates

## A2 Bench History

**v1 (A2):** 6/10 PASS "full" config — gate MET. But accidental: HTTP 400 (A33 bug) → circuit breaker → single-agent fallback → direct code gen. NOT the intended path.

**v2:** 0/2 PASS — semantic.db from v1 inflated entity hits (360+) → omega=2 → robust 5-node topology → drift RESET_AGENT → 120s timeout. Fix: clear session memory files between configs (`1ede89f9`).

**v3:** 0/4 PASS — memory isolation worked (omega=1). But A33 fixed deepseek-v4-flash multi-turn → proper multi-agent path → `search_repo`/`read_file` tool loops on BCB's empty codebase → 120s timeout. Attempted fix: `instructions=` in `normalize_bcb()` (`9a2de590`).

**v4:** Still 120s timeout — `TaskInput.instructions` NOT propagated to topology nodes. `runner.run(str)` takes a plain string; each node gets a fresh minimal `TaskInput(instructions="")`. Instructions only reach single-agent bypass path via `perceive.py:177`. Text instruction ignored by deepseek-v4-flash when tools available anyway.

**v5:** Disabled repo tools (`register_repo_tools=False` in `boot_agent_system`, `ad79d70f`). Still 120s timeout — model shifted to `create_bash_tool`, `create_python_tool`, `search_memory`, `search_exocortex` loops.

**v6 (COMPLETE 2026-05-03): 3/10 PASS — gate MISSED (diagnostic path)**
- Tool loop fix worked: `system.tool_registry._tools.clear()` → tool_calls=0 on ALL nodes
- BUT two new failure modes emerged (see below)
- Crashed on `baseline` config boot (PermissionError on episodic.db)
- Only `full` config ran; other 5 configs (baseline/no-memory/no-avr/no-routing/no-guardrails) never executed

## v6 Final Results (full config, 10 tasks)

| # | Task | Result | Time | Root cause |
|---|------|--------|------|------------|
| 1 | BCB/13 | FAIL | 67s | No output (synthesizer echoed plan) |
| 2 | BCB/15 | FAIL | 120s | 5-node robust + reroute → 120s cap |
| 3 | BCB/17 | FAIL | 120s | S2 CoT retry + model upgrade → 120s cap |
| 4 | BCB/19 | PASS | 90s | Clean 3-node sequential |
| 5 | BCB/34 | PASS | 69s | Clean 3-node sequential |
| 6 | BCB/37 | FAIL | 120s | Model upgrade → extra API call → 120s cap |
| 7 | BCB/82 | FAIL | 70s | No output (synthesizer echoed plan) |
| 8 | BCB/89 | FAIL | 120s | Cross-provider model upgrade (gemini→deepseek endpoint) → 400 → fallback → 120s |
| 9 | BCB/92 | PASS | 57s | Clean 3-node sequential |
| 10 | BCB/93 | FAIL | 76s | No output (synthesizer echoed plan) |

**Stats:** Pass rate 30% (3/10), avg latency 90913ms, avg cost $0.0046/task, routing S3=6 unknown=4 (S1=0 S2=0)

## Root Cause (key insight — tool loop, fixed in v6)

`TaskInput.instructions` and `TaskInput.tools_filter` ONLY work for single-agent bypass paths (via `perceive.py`). Topology runner (`runner.run(str)`) passes a raw string; each node gets `tools_filter=None` → all tools available. For multi-agent BCB paths, must disable tools at the boot/registry level.

**Why:** `pipeline._stage_execute()` calls `runner.run(ctx.task)` with a string. Runner creates per-node `AgentLoop` via `_agent_loop_factory(**_factory_kwargs)` — factory doesn't pass `task_input`. Each node's `loop.run(full_task)` wraps the string in a minimal `TaskInput`.

## New Failure Modes Discovered in v6

### Failure Mode A: "No output" (3/10 tasks)
**Symptom:** Synthesizer node outputs same char count as planner (1864, 2279 chars). Internal BCB evaluator finds no `def task_func(` → reports "no output".
**Root cause:** Synthesizer node system prompt says "synthesize outputs of all nodes" — it paraphrases/summarizes the plan instead of returning the Python code. When planner produces 1864 chars of planning text and coder produces another 1864 chars (also text), synthesizer produces another 1864 chars of synthesis summary.
**Fix needed:** Synthesizer node system prompt for BCB must say "return the Python code implementation verbatim, not a summary."

### Failure Mode B: Adaptive controller overhead (4/10 tasks)
**Symptom:** Tasks hit exactly 120s hard cap. Log shows `Node N model upgraded to X` or `Topology reroute triggered` AFTER node N already completed → controller triggers extra API calls that push budget-tier tasks over 120s.
**Specific variants:**
- Model upgrade: `Node 1 model upgraded to gpt-5.4` → re-runs node 1 with GPT-5.4 → extra 30-40s
- S2 CoT retry: `S2 validation: missing reasoning, requesting CoT` → extra API call
- Reroute: `Topology reroute triggered → REBUILDING full topology` → runs node 0 twice
- Cross-provider upgrade: `Node 2 model upgraded to gemini-3.1-flash-lite-preview` → sent to deepseek endpoint → 400 error → Stage 4 failure → single-agent fallback → still hits 120s
**Fix needed:** For budget tier, disable adaptive model upgrades and reroutes. OR raise timeout from 120s to 180s+.

### Failure Mode C: PermissionError on episodic.db (crash, 1 occurrence)
**Symptom:** After `full` config completes, `_clear_a14_topology_state()` tries `p.unlink()` on `episodic.db` → WinError 32 (file in use). The atexit handler from the `full` config's system boot still holds the SQLite connection.
**Location:** `__main__.py:193` `_clear_a14_topology_state()` → `p.unlink()` → `os.unlink(self)` → PermissionError
**Fix needed:** `_clear_a14_topology_state()` should only delete topology JSON files (manifest, state files), NOT `episodic.db`. The function name implies topology state, not episodic DB.

## Commits (May 3)

- `9a2de590` — fix(bcb): instructions in normalize_bcb (partial fix — single-agent only)
- `ad79d70f` — fix(bench): disable repo tools for BCB (insufficient — model loops on other tools)
- `639e1a34` — fix(bench): clear ALL tools for BCB ablation
- `1dc7634e` — docs: update READMEs (tests 2887→2903, deepseek-chat→v4-flash, feature flag defaults, sage-python README rewrite, bench README rewrite)
- `ad00e9e9` — fix(bench): clear A14 topology state between ablation config iterations

## v7 FINAL RESULTS (2026-05-03)

**Full config: 4/10 PASS — GATE MET** ≥4/10 → A3 N=50 authorized.

| # | Task | v7 | v6 | Notes |
|---|------|----|-----|-------|
| 1 | BCB/13 | FAIL 74s | FAIL 67s no-output | still wrong code |
| 2 | BCB/15 | FAIL 47s | FAIL 120s TIMEOUT | Fix 3: no longer timeout — correctness fail |
| 3 | BCB/17 | FAIL 102s | FAIL 120s TIMEOUT | Fix 3: no longer timeout — correctness fail |
| 4 | BCB/19 | PASS 66s | PASS 90s | faster |
| 5 | BCB/34 | PASS 55s | PASS 69s | faster |
| 6 | BCB/37 | PASS 90s | FAIL 120s TIMEOUT | Fix 3: synthesis under 120s cap |
| 7 | BCB/82 | FAIL 120s | FAIL 120s TIMEOUT | Still 5-node robust + controller overhead |
| 8 | BCB/89 | FAIL 120s | FAIL 120s cross-provider | Cross-provider fix NOT active in v7 (loaded before fix) |
| 9 | BCB/92 | PASS 69s | PASS 57s | Clean sequential |
| 10 | BCB/93 | FAIL 66s | FAIL 76s no-output | Fix 3: no longer "no output" — correctness fail |

Stats: Pass rate 40% (4/10), gate PASSED.

**v7 ALL 6 CONFIGS COMPLETE (60/60 tasks, 2026-05-03 ~20:00)**

| Config | PASS | PASS tasks |
|--------|------|------------|
| full | 4/10 | /19, /34, /37, /92 |
| baseline | 8/10 | /13, /15, /17, /19, /34, /37, /89, /92 |
| no-memory | 4/10 | /19, /34, /37, /92 |
| no-avr | 4/10 | /13, /19, /34, /37 |
| no-routing | 4/10 | /13, /34, /37, /92 |
| **no-guardrails** | **7/10** | /13, /17, /19, /34, /37, /89, /92 |

**ORIGINAL FRAMING (INCORRECT — kept as historical record)**: "`no-guardrails` 7/10 >> other pipeline configs 4/10. Guardrails (adaptive controller: model upgrades, reroutes, CoT retries) add ~50s overhead per task → timeouts. Removing guardrails nearly matches baseline 8/10. This directly validates Fix C (budget-tier guard = skip guardrails for budget tier)."

**CORRECTED FRAMING (cgpro 2026-05-04, verified against `bench/ablation.py`, `phases/{act,learn,perceive}.py`, `pipeline.py`)**: `AblationConfig(guardrails=False)` only sets `loop._skip_guardrails=True`. That flag short-circuits the `guardrail_pipeline` rule-based input/output checks (`phases/act.py:175-179`, `phases/learn.py:57`, `phases/perceive.py:119`). It does NOT touch the `TopologyController` Phase C runtime adaptation. **The v7 4/10→7/10 gap is the effect of disabling rule-based guardrail_pipeline checks**, NOT the effect of disabling adaptive controller. Fix C (`a23e196b`) disables the TopologyController via `_effective_controller=None` when `tier=="budget"` — that's a different lever, orthogonal to the ablation flag. Fix C may still be valuable (controller does add overhead) but its empirical validation requires running paired full vs no-guardrails *with Fix C active* and observing whether the guardrail_pipeline gap persists. A3 partial (11/33 = 33% on `full` with Fix C active) is within sampling noise of v7 `full` 4/10 = 40% and tells us nothing yet.

Persistent fails across all configs: BCB/82 (5-node robust → always timeout), BCB/93 (correctness). BCB/15 fails all pipeline configs (correctness).

A3 N=50 launched immediately after v7 exit (PID 23228 held episodic.db lock until ~20:00).

**Fixes confirmed by v7 data:**
- Fix 1 (episodic close): WORKS — baseline config started cleanly (no PermissionError between configs)
- Fix 3 (OUTPUT REQUIREMENT): WORKS — BCB/15, 17, 37, 93 no longer "no output"; BCB/37 TIMEOUT→PASS
- Cross-provider fix: NOT ACTIVE in v7 (module cached). Active in A3+.

**Commits shipped (all on GitHub main):**
- 2792b44f: Fix 1 (episodic close) + Fix 3 (OUTPUT REQUIREMENT)
- 9715ed4e: cross-provider fix (_is_cross_provider, returns None)
- 99fd1c31: 4 regression tests for cross-provider guard
- f7a8bc47: cgpro-validated: topology revert + runner guard + entry_point fix
- 6e79bf84: CLAUDE.md + current.json updated
- 7c6ab507: architecture.md + sage-discover README updated
- ad6cd78f: docs: A2 v7 gate MET (4/10 PASS) — A3 N=50 launched

**Pending (Fix C — budget-tier guard)**: Skip model upgrades/reroutes for budget tier. Not implemented — A3 data will show if BCB/82 + remaining TIMEOUTs justify the context plumbing work.

## BCB Ablation Design Note

Framework value for BCB comes from routing (kNN S1/S2/S3 classification), topology selection (sequential 3-node vs simpler), and automatic memory context injection (`perceive.py` semantic query — no tool call needed). NOT from agent tool calls. Clearing tools is correct for BCB; SWE-bench and other repo-based benches must keep tools.

**Why:** After 5 fix iterations revealed that any available tool causes deepseek-v4-flash to loop. BCB tests whether SAGE's orchestration ABOVE the tool layer adds value.

## Doc Updates (May 3)

- Root README: tests badge 2887→2903, DeepSeek model deepseek-chat→deepseek-v4-flash
- sage-core/README: feature flags corrected (sandbox/cranelift/tool-executor/cognitive now DEFAULT)
- sage-python/README: full rewrite (was 1426 tests, one provider, wrong bench commands)
- sage-python/src/sage/bench/README: full rewrite (add BCB, SWE-bench, routing_gt, ablation --tier)
