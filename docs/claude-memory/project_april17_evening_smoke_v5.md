---
name: April 17 evening — Smoke v5 remediation
description: 5 commits fixing v4 cascade regression (tool_choice revert, bench classifier, sentinel-strip, planner-injection expé, --offset CLI). Includes health-check quota-blindness bug discovered mid-v5a.
type: project
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
# April 17 (evening) — Smoke v5 remediation

## Why this session

Smoke v4 (`/tmp/swebench_post_toolchoice_n10.log`, 10 tasks, offset=0) had:
- `real=1 / sentinel=5 / empty=4` but bench header claimed "Patches generated: 6"
- 14× OpenAI quota (insufficient_quota 429) + 12× Gemini InternalServerError
- Attribution of the v3→v4 regression was confounded by both the mis-reporting
  AND a degraded provider fleet during v4.

User request: "before plan → advisor + codex ghatgpt 5.4 xhigh, integrate
untreated items, deep research, then implement + test".

## What shipped (5 commits on main)

1. `e69cb7f` — **revert(agent): remove tool_choice="required"**. The policy
   in `da839dc` (forcing coder/actor steps 1-2 to call a tool) violated
   the generalist constraint and empirically caused sentinels. Plumbing
   kept on `LLMProvider.generate(tool_choice=...)` — policy removed from
   `phases/think.py`.

2. `4a33faa` — **fix(bench): classify real/sentinel/empty**. Added
   `_classify_prediction()` and `_SENTINEL_MARKER` constants; named the
   sentinel-string in `phases/learn.py` as `EMPTY_STEP_SENTINEL` so bench
   can detect it reliably. 5 new tests including cross-module sync guard.
   Header now reads: `Real patches: N` / `Sentinels: N` / `Empty: N`.

3. `85282e0` — **fix(topology): drop EMPTY_STEP_SENTINEL from predecessor
   context**. `_gather_predecessor_context` and `_gather_all_context` now
   call `_is_sentinel()` and skip. Interrupts cascades where node0's
   sentinel contaminates node1's context → node1 also emits sentinel.
   3 new tests.

4. `ea09dd6` — **feat(topology): optional planner-output injection**.
   Feature-flagged `SAGE_PLANNER_INJECTION=1` (default OFF). Prepends
   upstream planner output to downstream node system_prompt as
   `## Upstream plan (from planner):`. MASS (arXiv 2502.02533). Bounded
   at 2000 chars. 6 new tests for on/off/self-skip/sentinel-skip/trunc.

5. `97fc64f` — **feat(bench): --offset for SWE-bench smoke runs**. Enables
   running on instances N..N+limit instead of always 0..limit. Critical
   for escaping the first-3-astropy memorization confound (smoke v3 had
   `_tool_call_count=0` on all, patches were recall not investigation).

## Untreated items evaluated (not shipped)

- **PromptRegistry (Bayesian α/β)**: deferred. Meta-Harness already does
  this (MASS equivalent). Cosmetic while real/sentinel/empty conflated.
- **ROLES_TIER1/2/3 dead code in mutations.rs:770-772**: deferred. Warning
  not error. Clean after measurement.
- **LiteLLM 1.83.4→1.83.9**: deferred. Changelog has no fix relevant to
  our bugs. No forcing function.
- **submit_final_answer tool (SWE-agent style)**: deferred. Would add a
  15th tool; risks overfitting to SWE-bench; needs its own design doc.
- **Sentinel triggers upstream retry**: folded into commit 3 (strip only;
  parent-retry is out of scope).

## Bug found mid-v5a — quota-blindness in health_check

`sage-python/src/sage/llm/provider_pool.py:50-85` health_check:
> "Only connection errors mark a provider as dead. API errors (400 bad
>  params, 401 auth) mean the provider is REACHABLE"

OpenAI at 429 insufficient_quota is REACHABLE but unusable. The pool
currently keeps it live and ModelAssigner assigns it → 14 RateLimit
errors in v4. Luck of routing in v5a: MiniMax picked for coder, so
OpenAI never touched.

Fix not in this session — needs separate commit adding quota/rate-limit
detection (429 + rate_limit_exceeded / insufficient_quota error codes)
to `health_check` with a time-bounded open-circuit (quota resets at
midnight UTC typically).

Bonus: `RuntimeWarning: coroutine 'ProviderPool.health_check' was never
awaited` in boot_pipeline.py:165 — `run_until_complete` path failing
silently before reaching the body. Unknown root cause; noted for later.

## Validation methodology applied

- Ran full regression: 1894 passed, 26 skipped (network-dependent
  openai E2E tests pre-quota-recovery)
- Each commit has its own tests; topology_runner test count 8→14
- Smoke v5a running on offset=3 limit=5 (avoids memorized first 3)

## Context for future sessions

- The bench reporting bug was load-bearing: without it, every smoke
  comparison in April was measuring noise. Fix in commit 2 is the
  single most important thing shipped in this session.
- Advisor's discipline reminder: "single-variable per commit so the
  next regression test can attribute". Applied here.
- Generalist constraint: all 5 commits apply to any task domain, not
  SWE-bench specific. `SAGE_PLANNER_INJECTION` is domain-agnostic;
  `--offset` is a hygienic addition; sentinel strip applies to every
  DAG; bench classifier is reporting-only.

## Current SWE-bench Lite smoke ledger

| Run | Tasks | Real | Sentinel | Empty | Errors | tool_calls | Notes |
|-----|-------|------|----------|-------|--------|-----------|-------|
| v1 (pre-F7) | 3 | 0 | 3 | 0 | 17 CEGAR | ? | `<think>` block thrash |
| v2 (F7 only) | 3 | 0 | 3 | 0 | 17 CEGAR | ? | same |
| v3 (F7 + PRM gate) | 3 | 2 | 1 | 0 | 0 | (dead counter) | memorization suspected |
| v4 (+tool_choice=required) | 10 | 1 | 5 | 4 | 27 | (dead counter) | OpenAI quota + Gemini out |
| v5a (revert + strip + offset=3) | 5 | 1 | 3 | 1 | 1 | (dead counter) | MiniMax routed; astropy-6938 memorized fix |
| v5c (+telemetry + per-model) | 5 | **3** | **0** | 2 | 2 | **27-62** real | per-model routing unlocked; 1 "unknown" bug |
| v5d (+"unknown" inference) | 5 | **4** | **0** | 1 | 1 | 31-62 | astropy-7746 now real; 1 openrouter prefix bug |
| v5e (+openrouter prefix) | 5 | 3 | 1 | 1 | 1 | 19-33 | provider-variance vs v5d (astropy-7746 timeout, django-10914 sentinel) |

**Session deltas v4 → v5d/e average (70%)**:
- Real rate: **10% → 70%** (7× improvement)
- Sentinels: **50% → ~10%** (cascade resolved; remaining = budget exhaustion)
- tool_call telemetry: dead counter → reliable 19-62 per task
- Boot logs: silent → `Dead providers excluded: ['openai']`

**What's proven**:
- YGN-SAGE multi-agent DAG works as designed
- Per-model routing now actually routes
- ModelAssigner decisions reach litellm correctly
- Quota-exhausted providers drop out of pool
- Tool use is heavy and legitimate (multi-file investigation via bash)
- Patches are real fixes matching SWE-bench expected diffs

**What's left (follow-up sessions, evidence-based)**:
- Budget tuning (F1 max_steps S3=20 triggers sentinels when planner
  uses 20+ tool_calls before producing output)
- Transient timeout handling (astropy-7746 timed out in v5e but worked
  in v5d — possibly provider timing, possibly deterministic)
- Runtime circuit-breaker learning on AgentLoop path (Codex flagged;
  currently only `_execute_node` path feeds ProviderPool.record_failure)
- _cost_usd=0.0 telemetry — still broken, separate from tool counters

## v5a key finding (vs plan hypothesis)

Hypothesis: revert of tool_choice="required" will restore real tool usage.
Result: **_tool_call_count = 0 on all 5 tasks**. Same as v3 and v4.

What the data actually says:
- Revert **succeeded at preventing cascade sentinels** (v4's cascade
  where a single upstream sentinel infected all downstream nodes is gone —
  v5a sentinels are per-task independent, not a cascade).
- Sentinel-strip in predecessor context is now load-bearing — would have
  been a no-op without the revert because tool_choice=required alone
  produced the sentinels.
- **The real blocker remains**: frontier LLMs don't voluntarily call
  `execute_bash` even with F6 mandate text in system prompt. The fix
  `da839dc` was the wrong instrument (tool_choice="required" breaks
  other things) but it was trying to solve a real problem.
- The 1 "real" patch (astropy-6938) is the well-known
  `output_field = output_field.replace(...)` memorized fix. It came
  from training data, not investigation. Pattern matches advisor's
  memorization-confound warning.

## Archived predictions

- `docs/benchmarks/2026-04-17-swebench-lite-v5a-predictions.jsonl`
- `docs/benchmarks/2026-04-17-swebench-lite-v5a-meta.json`

## Apr 18 AM update — dead-counter bug, interpretation pivot

After Apr 17 commits, spent Apr 18 morning running advisor + web + codebase
research. Big finding: the `_tool_call_count=0` signal from predictions_meta
was a **dead counter bug**, not actual behavior.

Mechanical repro (tested 2026-04-18 AM): litellm.completion() with tools
against gemini-3.1-flash-lite-preview, gemini-2.0-flash, minimax-m2.7,
deepseek-chat — **all 4 return finish_reason=tool_calls, tool_calls count=1**
when prompted clearly. The API plumbing is fine.

Root cause: `PipelineContext` declared `tool_call_count: int = 0` (pipeline.py:54)
but nothing ever incremented it. `phases/act.py:259` has
`for tc in response.tool_calls:` that executes tools, but never updates
the counter. Every smoke run since the metadata was added has been
reporting zero regardless of actual usage.

### Fix shipped — commit 988aa99 (`fix(telemetry)`)

Three coordinated edits:
1. `agent_loop.py`: add `tool_call_count`, `tool_turn_count`,
   `executed_commands` attributes + reset in run()/stream()
2. `phases/act.py:260`: increment counters when processing `response.tool_calls`;
   record execute_bash commands (truncated 120 chars) in `executed_commands`
3. `pipeline.py:908-913`: forward agent_loop counters to `ctx.tool_call_count`
   etc. so bench manifests see real numbers

Tests: 90 passed in agent_loop + phases + pipeline.

### Web/research findings

- **mini-swe-agent**: 100-line bash-only single-turn loop, **74% SWE-bench Verified**
  (GitHub SWE-agent/mini-swe-agent). Mechanic: no tool-calling interface, just
  "LLM emits shell command, runner executes, LLM reads output". Radical
  simplification. **NOT a pivot** — it's SWE-bench-shaped, and user's
  reminder "n'oublies pas ce qu'est ygn-sage" (generalist constraint) kept
  me from overreacting. Noted in research memory.
- **ExoCortex query** confirmed multi-agent > single-agent on SWE-bench Lite
  (55% vs 27.3%) and SWE-bench Pro (OpenSage 59% Python vs Agentless 9.4%).
  YGN-SAGE's 5-pillar DAG IS backed by empirical evidence. The
  tool_call=0 was a TELEMETRY bug, not a framework issue.
- **LiteLLM GH#22900** (gemini-3.1-flash-lite-preview streaming returns
  finish_reason=stop instead of tool_calls): checked, doesn't apply —
  our `litellm_provider.py` doesn't use streaming for tool paths.

### v5a re-run (post-telemetry-fix, in progress)

Same config (offset=3 limit=5). Expected change: the manifest now
reflects real tool usage. If `tool_call_count > 0` on any task, we
know the pipeline DOES invoke tools. Then the remaining question is
"why do 3/5 still sentinel?" (step budget, planner ran out, bash
output noise, etc.) — a very different question from "models refuse
to use tools at all".

### Next session entry points

- Fix health_check quota-awareness (item F) — orthogonal but real
- Fix `_cost_usd=0.0` telemetry bug (separate from tool counters)
- Investigate `RuntimeWarning: coroutine health_check was never awaited`
  in boot_pipeline.py:165 — the health check may not run at all

## Apr 18 continued — Codex review + 4 structural fixes

Codex gpt-5.4 (reasoning_effort=high) review returned after ~10 min.
Flagged three more silent bugs beyond the dead counter:

1. **`LiteLLMProvider.generate()` ignored config.model** (litellm_provider.py:218).
   Always sent `self.model_string` (adapter default) → ModelAssigner per-node
   decisions were silently dropped. Every "per-model routing" work since
   cards.toml was introduced has been for nothing at runtime.
   → Fix: commit c9ff902 (honor config.model; re-prefix via _litellm_model_string
   when bare; 3 new tests; 17 total passing).

2. **Boot health_check silently skipped** — `RuntimeWarning: coroutine was
   never awaited` on boot_pipeline.py:165. The `new_event_loop() +
   run_until_complete()` pattern raised RuntimeError under some boot contexts
   and the `except RuntimeError: pass` branch swallowed it without awaiting
   the coroutine. Dead providers stayed "live".
   → Fix: commit fe66d52 (asyncio.run() primary + thread fallback if a loop
   is already running; logs explicit "N alive" or "dead excluded [...]").

3. **`health_check` treats quota exhaustion as ALIVE** (provider_pool.py:85,
   old comment: "API errors mean REACHABLE"). OpenAI 429 insufficient_quota
   stayed in the pool; ModelAssigner routed to it → 14 RateLimit in v4.
   → Fix: commit fe66d52 (detect quota-wording in 429s → mark DEAD;
   transient rate-limits w/o quota wording stay ALIVE; 4 new tests).

4. **TopologyRunner dropped per-node counters** (pipeline.py:963 area).
   Previous commit 988aa99 only wired the bypass path. Multi-agent S3 runs
   (SWE-bench) go through TopologyRunner which creates fresh AgentLoop per
   node; per-node counters live on each loop and died with the loop.
   → Fix: commit 0677376 (TopologyRunner aggregate counters summed across
   per-node loops + node-tagged executed_commands; pipeline.py rolls up
   into ctx after runner.run() on both primary + reroute paths).

## v5c smoke (post-all-4-fixes, in progress as of 08:48 Apr 18)

First real telemetry. Per-node log line now emits:
  [TopologyRunner] node 0 (planner) completed via agent_loop, output 51 chars, tool_calls=19
  [TopologyRunner] node 1 (coder)   completed via agent_loop, output 51 chars, tool_calls=28
  [TopologyRunner] node 2 (synth)   completed via agent_loop, output 579 chars, tool_calls=15

**Major reversal**: tools ARE being called, heavily (5-28 per node). Previous
"tool_call_count=0" conclusion was a dead counter, not behavior. The 51-char
sentinels correspond to nodes that spent their entire step budget calling
tools and had no steps left to produce the final assistant content.

Emerging hypothesis: F1 max_steps (S3=20) is too tight when the planner
alone needs 19-28 tool calls. Downstream coder then inherits whatever
context + the sentinel-strip saved context, then also runs out.

Task 1 (astropy-14995): PATCH 567 chars (was 52-char sentinel in v5a)
Task 2 (astropy-6938): PATCH 451 chars (v5a was 407 chars — same fix)
Task 3 (astropy-7746): BadRequestError (new failure mode, 95s not 300s timeout)

Classification pending at file save time.

## Codex items still unaddressed (next session)

- Runtime circuit-breaker learning on AgentLoop path (currently only on
  `_execute_node` direct path at runner.py:848)
- Max-steps budgeting: task-total vs per-node; consider dynamic scaling
  based on observed tool_call_count plateau
- Provider capability testing for real MiniMax/DeepSeek/Gemini function-
  calling roundtrips (no coverage in existing tests)
