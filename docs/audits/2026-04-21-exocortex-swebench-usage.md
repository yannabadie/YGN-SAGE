# ExoCortex Usage Audit — SWE-bench Lite Smoke Runs (v13 / v15 / v17)

**Date**: 2026-04-21
**Question**: Was `search_exocortex` actually invoked by the agent during the v13 / v15 / v17 SWE-bench Lite smoke runs?
**Verdict**: **No. Zero invocations across all runs.**
**Type**: Read-only audit (no production code touched).

---

## Executive Summary

- `search_exocortex` is registered at boot (`sage-python/src/sage/boot.py:442-445`) and returned in `get_tool_defs(None)` for planner/coder/actor roles (`sage-python/src/sage/tools/registry.py:36-54`).
- `SAGE_EXOCORTEX_STORE` is set in the local `.env`, so ExoCortex is NOT disabled.
- Across v17 (the only run with per-task tool metadata preserved in `docs/benchmarks/2026-04-21-swebench-v17-full.jsonl`), **every task had `_tool_call_count == len(_executed_commands)`**. Since `_executed_commands.append` is gated by `if tc.name == "execute_bash"` (`sage-python/src/sage/phases/act.py:306,314`), the equality proves 100% of tool calls were `execute_bash`. Zero `search_exocortex`.
- For v13 / v15 (eval-only, reusing v13 predictions), the equivalent jsonl was written to a temp dir (`C:\Users\yann.abadie\AppData\Local\Temp\sage_swebench_kfhxbz7i\predictions.jsonl`, per `2026-04-21-swebench-v15-eval.log:2`) and is not archived. Indirect evidence (v13 smoke log per-node `tool_calls=N` + identical code path + identical role filtering) is strong but not arithmetic.

---

## 1. Tool Registration — **YES** (but never logged)

| Evidence | Location |
|---|---|
| Factory function | `sage-python/src/sage/tools/exocortex_tools.py:10-41` (`create_exocortex_tools`, binds to the `ExoCortex` instance, returns `[search_exocortex, refresh_knowledge]`) |
| Boot-time registration | `sage-python/src/sage/boot.py:442-445` (`for tool in create_exocortex_tools(mem["exocortex"]): tool_registry.register(tool)`) |
| No log line | `sage-python/src/sage/tools/registry.py:14-23` (`register()` does not log). Absence of "ExoCortex tools registered" in the smoke logs is therefore NOT evidence of absence. |
| Wired test | `sage-python/tests/test_exocortex_wiring.py:144-159` asserts `search_exocortex` is in the returned tool list. |
| Store env | `.env` is present; `SAGE_EXOCORTEX_STORE` is non-empty (value redacted). ExoCortex is NOT in silent-no-op mode. |

Boot logs for v13 (`2026-04-21-swebench-smoke-v13-10task-post-phase1-stab.log:19,100`) and v17 (`2026-04-21-swebench-v17-full.log:19,101`) show `execute_bash` and `sage_recurse` registrations — ExoCortex tool registration happens silently in the same boot sequence.

## 2. Tool Availability to the Agent

### Runtime (schema sent to LLM): **YES for planner/coder/actor roles, NO for verifier/formatter/synthesizer**

- `sage-python/src/sage/phases/perceive.py:136-137`: `tool_defs = loop._tools.get_tool_defs(loop.config.tools if loop.config.tools else None)`. When `config.tools is None`, ALL registered tools flow into the LLM schema (`registry.py:36-54`). That includes `search_exocortex`.
- `sage-python/src/sage/phases/think.py:100-104`: `tool_defs` is passed directly to `loop._llm.generate(tools=tool_defs)`.
- Role filtering in `sage-python/src/sage/agent_loop_factory.py:20-21,93-104`:
  - `_VERIFIER_TOOLS = ["execute_bash", "stm_read", "stm_write", "ltm_recall"]`
  - `_FORMATTER_TOOLS = ["stm_read", "stm_write", "ltm_recall"]`
  - **Neither includes `search_exocortex`.** Synthesizer/formatter/verifier roles cannot invoke it (they match these filters; F10 audit line 99 explicitly added synth to the formatter filter).
  - Planner, coder, and default actor get `tools=None` → full tool list → `search_exocortex` IS in their schema.

### Prompt (what the LLM is TOLD it has): **NO**

- `sage-python/src/sage/bench/swebench_bench.py:149-153` (`_TASK_TEMPLATE`) lists tools as:
  ```
  You have these tools available:
  - execute_bash — run any shell command ...
  - Memory + knowledge tools registered at boot (if any)
  ```
  `search_exocortex` is **never named**. The agent discovers it only by inspecting the LLM's own tool-call schema, which is not a substitute for prompt-level encouragement.
- Role prompts in `sage-python/src/sage/topology/role_prompts.py`:
  - `_PLANNER` (line 37-67): mentions `execute_bash, search_memory` only.
  - `_CODER` (line 70-99): mentions `execute_bash` only.
  - `grep -nE "exocortex|search_exocortex|knowledge|refresh_knowledge"` in that file → 0 matches.

## 3. Tool Calls Observed

### v17 (`docs/benchmarks/2026-04-21-swebench-v17-full.{log,jsonl}`) — direct arithmetic proof, N=10 tasks

Parsed metadata from `2026-04-21-swebench-v17-full.jsonl`:

| task_id | tool_call_count | len(executed_commands) | non-bash tool calls |
|---|--:|--:|--:|
| astropy__astropy-12907 | 0 | 0 | 0 |
| astropy__astropy-14182 | 29 | 29 | 0 |
| astropy__astropy-14365 | 0 | 0 | 0 |
| astropy__astropy-14995 | 28 | 28 | 0 |
| astropy__astropy-6938 | 27 | 27 | 0 |
| astropy__astropy-7746 | 33 | 33 | 0 |
| django__django-10914 | 0 | 0 | 0 |
| django__django-10924 | 40 | 40 | 0 |
| django__django-11001 | 0 | 0 | 0 |
| django__django-11019 | 0 | 0 | 0 |
| **Total** | **157** | **157** | **0** |

**Why this proves zero `search_exocortex` calls**:
- `executed_commands.append` is gated by `if tc.name == "execute_bash":` at `sage-python/src/sage/phases/act.py:306-315`.
- `tool_call_count` increments for **every** tool call (`act.py:299`: `loop.tool_call_count += len(response.tool_calls)`).
- The pipeline aggregates both fields across all topology nodes (`sage-python/src/sage/topology/runner.py:567-572`; prefixes commands with `[role]` — the `[planner]`/`[coder]` tags in v17's `executed_commands` confirm aggregation is working).
- `pipeline.py:1212-1214` rolls runner aggregates into `ctx`, then `swebench_bench.py:568-570,606-610` writes them into the jsonl.
- Therefore `tool_call_count == len(executed_commands)` on all 10 tasks → every tool call was `execute_bash` → zero non-bash tools (including `search_exocortex`).

Grep on the v17 log for "exocortex" / "search_exocortex" / "SAGE_EXOCORTEX": **0 matches**.
Grep on all 157 aggregated commands for "exocortex": **0 matches**. First-word distribution: `grep` ×65, `sed` ×61, `cat` ×13, `find` ×8, `git` ×6, `python` ×4. All bash.

### v13 (`docs/benchmarks/2026-04-21-swebench-smoke-v13-10task-post-phase1-stab.log`) — strong circumstantial

No per-task jsonl metadata was archived (`v15-eval.log:2` confirms the v13 predictions lived in `C:\Users\yann.abadie\AppData\Local\Temp\sage_swebench_kfhxbz7i\predictions.jsonl`, now gone). Evidence:
- Per-node log lines (`tool_calls=N`) use the SAME counter (`topology/runner.py:575`) that aggregates into `tool_call_count`. 37 `tool_calls=N` lines total, all >0 values are on `planner`/`coder` nodes; synthesizer always shows `tool_calls=0` (expected — formatter filter).
- No "exocortex" / "search_exocortex" / "SAGE_EXOCORTEX" / "no store configured" string in the log.
- Same code path as v17 (predate v16/v17 fixes were validator/fallback changes, not tool registration changes — see `git log main --oneline` for 2026-04-21).
- Conclusion: no positive evidence of `search_exocortex` invocation; strong structural reason to believe the v17 arithmetic result generalizes.

### v15 (`docs/benchmarks/2026-04-21-swebench-v15-eval.log`) — eval-only

`v15-eval.log:1-2` shows v15 is `swebench-v14 - V14 eval-only run` reusing v13 predictions. No agent generation happened in v15; zero net tool calls are possible in v15. The v15 pass-rate (1/10) is the Docker-grading of v13-generated patches.

## 4. Returned Results

**None observed.** No `search_exocortex` calls occurred, so there are no results to inspect. The `SAGE_EXOCORTEX_STORE` variable IS configured (`.env` has a non-empty value), so the tool would return substantive content IF invoked — the ExoCortex store itself is healthy (ExoCortex rattrapage ran through 2026-04-17; see `MEMORY.md:46`).

## 5. Diagnosis

The tool is registered and wired, but unused. Three causes stack:

1. **Prompt silence.** The SWE-bench task template (`swebench_bench.py:148-153`) only names `execute_bash`; `search_exocortex` is hidden behind "Memory + knowledge tools registered at boot (if any)". The coder role prompt (`role_prompts.py:70-99`) never mentions it. An LLM with a 6-tool schema but a prompt that demands `execute_bash + grep` will almost never try a cold tool unless explicitly invited.

2. **Mandatory workflow over-constrains to bash.** `_TASK_TEMPLATE` lines 158-172 say "You MUST make at least THREE distinct execute_bash calls", then prescribes a 6-step workflow whose every step names `execute_bash`, `grep`, `sed`, `cat`. This is an explicit anti-affordance: the model is punished for deviating from bash.

3. **Policy-intent conflict.** `.env.example:27-30` documents "SWE-bench / code-repair tasks don't depend on ExoCortex" as intentional design. Simultaneously, the v17 post-mortem (`2026-04-21-swebench-v17-full-results.md:83-85`) flags the astropy-14182 / django-11001 "semantic-miss bucket" as needing **"planner depth or ExoCortex retrieval"**. The current prompt encodes the first policy; the findings suggest the second is correct for a specific failure bucket. **This is the real audit finding**: the disclaimer in `.env.example` needs to be revisited for semantic-miss tasks, not just reinforced by silence.

Secondary: **audit observability is fragile.** `executed_commands` only records `execute_bash` (`act.py:306-315`). Non-bash tool calls are invisible to post-hoc forensic analysis. Any future "did X tool fire?" audit has to rely on the arithmetic `tool_call_count - len(executed_commands)` gap (or running a fresh smoke).

## 6. Fix Recommendations (ordered by effort)

### R1 — Low-effort, high-leverage: observability

Record all tool names (not just bash) in `executed_commands`, so future audits are a grep.

- File: `sage-python/src/sage/phases/act.py:306-315`
- Change: replace the `if tc.name == "execute_bash":` guard with logic that records every tool call, prefixed by name. E.g.:
  ```python
  cmd_str = ""
  if tc.name == "execute_bash":
      # ... existing command-extraction
  else:
      try:
          import json as _json
          args = tc.arguments if isinstance(tc.arguments, dict) else _json.loads(tc.arguments or "{}")
          cmd_str = f"{tc.name}({','.join(f'{k}={str(v)[:40]!r}' for k,v in args.items())})"[:120]
      except (ValueError, TypeError, AttributeError):
          cmd_str = f"{tc.name}(<unparsable>)"
  if cmd_str:
      loop.executed_commands.append(cmd_str)
  ```
- Cost: 10-line edit, no semantic change for existing bash behaviour. A passing `tests/test_agent_loop*.py` / `tests/test_swebench_bench.py` suite after the edit is the gate.
- Payoff: any future smoke jsonl becomes self-describing. `grep search_exocortex *.jsonl` is then the audit answer.

### R2 — Low-effort: name tools in the task template

Add `search_exocortex` (and `sage_recurse`, which is similarly orphaned) to the `You have these tools available` list, with a one-line use case.

- File: `sage-python/src/sage/bench/swebench_bench.py:148-153`
- Change the block to:
  ```
  You have these tools available:
  - execute_bash — shell commands (cat, grep, find, git, sed, python, pytest)
  - search_exocortex(query, domain?) — query the research-paper knowledge
    store. Use when the bug touches a concept you don't recognize
    (e.g. "what's the correct serialization protocol for X?"). Returns
    papers with passages; call at most 1-2 times per task.
  - sage_recurse(sub_task) — spawn a narrow sub-agent for a self-contained
    sub-problem. Use sparingly; budget-gated.
  ```
- Cost: 8-line prompt edit. Risk: the LLM may wander into exocortex on tasks where it isn't helpful; mitigate with the "at most 1-2 times" hint.

### R3 — Medium-effort: planner role prompt

Encode the "when to reach for ExoCortex" heuristic in the planner, not the bench template.

- File: `sage-python/src/sage/topology/role_prompts.py:37-67` (the `_PLANNER` constant)
- Add a bullet to the "Your role" list:
  ```
  - If the task mentions a domain concept you cannot verify from code
    alone (serialization protocol, wire format, spec compliance, standard
    naming conventions), call `search_exocortex(query=<concept>)` to pull
    the relevant paper passages into the plan.
  ```
- Cost: 5-line edit + re-run of the planner/topology tests (`tests/test_topology_*.py`, `tests/test_role_prompts*.py` if any).

### R4 — Medium-effort: decide the policy

`.env.example:27-30` says SWE-bench doesn't need ExoCortex. The v17 post-mortem says the semantic-miss bucket does. Resolve in an ADR so R2/R3 land on a stable footing, not as ad-hoc additions.

- File: new `docs/architecture/adr-0XX-exocortex-for-code-tasks.md` (follow the ADR template in `docs/adr/` / `docs/architecture/`).
- Content: decide "ExoCortex is off-path for SWE-bench" vs "ExoCortex is on-path when planner flags a semantic unknown" — then reconcile `.env.example` and prompts.
- Cost: 1 writing session, no code change.

### R5 — Higher-effort: measurement

Run a single SWE-bench smoke with `search_exocortex` named in the prompt (R2) + logged (R1), and a second smoke without (control), on the same 4-task semantic-miss bucket from v17 (`astropy-14182, astropy-14995, django-11001, django-11019`). Success criterion is not overall pass-rate but a causal chain: "on semantic-miss tasks, search_exocortex retrieved passage X, coder's plan changed to Y, patch passed".

- Cost: 2-4 hours run time + 30 min analysis. Requires Docker.
- Gate: do this AFTER R1+R2 land so the data is interpretable.

---

## Artifacts grepped

- `docs/benchmarks/2026-04-21-swebench-smoke-v13-10task-post-phase1-stab.log` (912 lines, 0 exocortex matches)
- `docs/benchmarks/2026-04-21-swebench-v13-eval-final4.log` (52 lines, 0 matches)
- `docs/benchmarks/2026-04-21-swebench-v14-eval.log` (80 lines, 0 matches)
- `docs/benchmarks/2026-04-21-swebench-v15-eval.log` (81 lines, 0 matches; eval-only reuses v13 preds)
- `docs/benchmarks/2026-04-21-swebench-v16-repair-eval.log` (113 lines, 0 matches)
- `docs/benchmarks/2026-04-21-swebench-v17-full.log` (765 lines, 0 matches)
- `docs/benchmarks/2026-04-21-swebench-v17-full.jsonl` (9 records + 1 task without meta, all `tool_call_count == len(executed_commands)`)
- `docs/benchmarks/2026-04-21-swebench-v17-full-results.md:85` (flags ExoCortex as a missing feature for the semantic-miss bucket)

All files read from `git show main:<path>` since this worktree is not on main.

## Constraints honoured

- No production-code edits. This is a read-only audit.
- Worktree branch `worktree-agent-a0b724bb` only; no commits, no touches to `main`.
- `SAGE_EXOCORTEX_STORE` value not printed.
