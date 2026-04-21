# BCB Hard Instruct Failure Analysis — 2026-04-21

Read-only audit of the 45.9 % run to identify the biggest lift buckets.

## Data source

| Field | Value |
|---|---|
| Run report | `docs/benchmarks/2026-04-08-bigcodebench-hard-instruct.json` |
| Predictions JSONL | `sage-python/docs/benchmarks/2026-04-08-predictions-hard-instruct.jsonl` |
| Commit (best match) | `8d7bc6d21e2c` ± (the 2026-04-08 run window; `git_sha` in the report JSON is empty — see §Blockers) |
| Total tasks | 148 |
| Passed | 68 (pass rate 45.95 %) |
| Failed | 80 |
| Errors flagged in report | 8 (all `api_error`) |
| Avg latency | 30 386 ms |
| Routing | 140 tasks S2 / 8 tasks unknown (api_error cases) |
| Split | Hard Instruct |

The Apr 10 and Apr 17 BCB Hard runs are smoke-only (5 and 20 tasks, all
TIMEOUT/provider outage) and **not** representative of the 45.9 % regime.
Apr 07 has the same 148-task volume as Apr 08 — Apr 08 is one iteration
downstream and is the number the project card quotes, so this audit uses
Apr 08.

## Classifier

Script: `scripts/audits/classify_bcb_failures.py`.
Output JSON: `scripts/audits/2026-04-08-bcb-hard-failure-classification.json`.

Classification is **cascaded** (first match wins), not OR'd, to avoid
double counting when a task carries both `generation_error` and an
`eval_error_snippet`. The snippet is capped at 200 characters in
`bigcodebench_bench.py` (line 217), so several rules work on positional
heuristics (`Traceback ... in <module>` followed by an `import` line
within the first 400 chars is almost always `ModuleNotFoundError` with
the keyword truncated off).

## Bucket table (80 failures)

| # | Category | Count | % of fails | Example task | Actionable lever |
|---|---|---:|---:|---|---|
| 1 | `assertion_failure_logical` | 25 | 31.2 % | BigCodeBench/17 (`test_process_found_restarts_process` — `FAIL:`) | **Enrich AVR context**: current retry sends only `eval_stderr[-500:]`, no source of truth, no test body. Feed the reasoner the failing test's source + first 5 asserts. AVR ran on 100 % of these and repaired 0. |
| 2 | `runtime_exception_in_test` | 24 | 30.0 % | BigCodeBench/129 (`test_return_type`) | **Schema-aware generation**: these are `TypeError`/`KeyError`/wrong return-type errors. Prompt must surface the function's return contract. Strong candidate for type-check step before eval (mypy/pyright quick-lint). |
| 3 | `truncated_warning_prefix` | 10 | 12.5 % | BigCodeBench/89 (matplotlib `FigureCanvasAgg`) | **Raise snippet cap from 200 to 2000 chars in bench code** so we can actually classify this bucket next run. Telemetry lever, not model lever. |
| 4 | `import_error` | 8 | 10.0 % | BigCodeBench/108 (`statsmodels`) | **Install the BCB dep set** in the bench venv: `cgi` (x2, needs Py ≤3.12 or `legacy-cgi` shim), `statsmodels` (x2), `flask_login`, `soundfile`, `xlwt`, `pyquery`, `pycryptodome`. This is an env gap, not a codegen miss. |
| 5 | `api_error` | 8 | 10.0 % | BigCodeBench/287 (DeepSeek "Model Not Exist") | Provider-side — already partially addressed by TTL exclusion (Apr 18). Not a SAGE-quality lever. |
| 6 | `import_error_truncated` | 1 | 1.2 % | BigCodeBench/82 (`flask_login`) | Merge with #4 (same root cause, just snippet layout differs). |
| 7 | `other` (SSL) | 2 | 2.5 % | BigCodeBench/177 (nltk `punkt` download, `CERTIFICATE_VERIFY_FAILED`) | Set `NLTK_DATA` with pre-downloaded corpora; outside scope of codegen lift. |
| 8 | `syntax_error` | 1 | 1.2 % | BigCodeBench/1019 (`IndentationError` in `except:` block) | Too small to act on alone. |
| 9 | `timeout_eval` | 1 | 1.2 % | BigCodeBench/1040 (30 s limit hit) | Isolated. Not a lever. |
| — | `test_env_error` (HF Snowflake) | 0 | 0 % | — | Not in this run (hit the Apr 17 re-run instead). |
| — | `empty_or_sentinel` | 0 | 0 % | — | Pipeline is emitting code every task. Not broken. |
| — | `silent_fail` | 0 | 0 % | — | No plumbing gap in this run. |
| — | `timeout_generation` / `generation_exception` | 0 | 0 % | — | Generation side is healthy. |

**Failures that are actually about codegen quality** (not env / provider /
telemetry): 25 + 24 + 1 = **50 of 80 (62.5 %)**.

## Cross-cuts (discriminators)

- **AVR fired on 72/72 non-api failures, repaired 0.**
  That's the single biggest signal in the audit. The reasoner-repair
  step in `bigcodebench_bench.py:152-176` runs on every failed task but
  has a 0 % success rate in this run. Either the repair prompt is too
  thin (only the last 500 chars of stderr and the original NL prompt —
  no test body, no function signature recap) or the repair-model
  routing isn't actually hitting a stronger tier. Root-cause either way
  is **signal to the repair step, not model capability**.
- **Bypass vs topology**: 49 / 63 non-api/non-import failures went the
  single-agent bypass path (`topology_nodes == 0`). Mean omega on those
  is 1.22 — many are decomposable but the bypass heuristic didn't let
  topology run. Escalation ("Step 2: Topology escalation") exists in
  the bench (line 179) but the trace doesn't tell us whether the
  escalated run passed or failed (the `trace.topology_nodes` reflects
  the **first** attempt, not the escalation). Improving visibility
  here is a prerequisite before we can prove topology-escalation lift.
- **Mean omega** by bucket: 1.22 for runtime-exception failures, 1.38
  for import failures (unrelated — imports are env gaps). Neither is
  high enough to be a primary decomposition lever on its own.

## Top 3 levers (ranked by lift_potential × inverse_effort)

### 1. Enrich AVR repair context  — lift: high, effort: ~1 day

**Why this wins.** 25 + 24 = 49 failures (61 % of total failures,
~33 % of 148) are in buckets where a better repair prompt could recover
them. AVR is already wired, already fires, but repairs 0 of them. The
failure is in the prompt, not the model.

**Scope.** `sage-python/src/sage/bench/bigcodebench_bench.py:152-176`
(the `retry_prompt` builder). Add: full `task["test"]` body, `entry_point`
signature re-print, and `task["code_prompt"]` (imports + signature).
Keep the prompt under 8 k tokens.

**Risk.** Low — the repair path is already sandboxed and only runs on
already-failing tasks. Worst case it's a no-op like today.

**Measurement plan.** Re-run the same 148-task BCB Hard Instruct split
on the same commit range. Target: lift `avr_repaired=True` count from
0 to 8–12 (~10 %) on the 49 target bucket. A +5 pp overall pass-rate
gate (45.9 % → 50 %) is the minimum to declare success; +8 pp (→ 54 %)
would be the stretch goal.

### 2. Raise the `eval_error_snippet` cap from 200 to 2000 chars — lift: unknown but UNBLOCKS #1, effort: 5 min

**Why it matters.** `truncated_warning_prefix` (10 tasks, 12.5 %) and
`import_error_truncated` (1 task) are failures we cannot classify
reliably because the warning ate the buffer before the real exception
rendered. Raising the cap is a one-line change; telemetry lift is
immediate. It also strictly improves lever #1 (AVR repair receives the
same snippet).

**Scope.** `bigcodebench_bench.py:217` (`eval_stderr[:200]` → `eval_stderr[:2000]`)
and `self._evaluate_solution_with_stderr` return-size handling if any.
Check `sage-python/src/sage/bench/runner.py` doesn't re-truncate.

**Risk.** Zero — JSONL grows by ~1 KB × 148 tasks = ~150 KB per run.

**Measurement plan.** Re-run any 20-task smoke, count new
`truncated_warning_prefix` cases: should drop to near 0.

### 3. Pre-install the BCB dependency set in the bench venv — lift: ≤ +5.4 pp, effort: ~30 min

**Why it's #3, not #1.** Caps at 8 + 1 = 9 tasks (max +6.1 pp). These
are genuine infra misses, not codegen. The lift is known and bounded —
we cannot get more than those 9 tasks no matter how good the repair.
Still worth doing because the win is essentially free. Grouping:

```
legacy-cgi      # Py 3.13 replacement for stdlib cgi (2 tasks: /273 /274)
statsmodels     # 2 tasks: /108 /917
flask_login
flask_wtf       # probable co-dep (BCB/82)
soundfile
xlwt
pyquery
pycryptodome    # provides Crypto.*
```

**Scope.** `sage-python/pyproject.toml` optional extra `bench-deps`,
then `pip install -e ".[bench-deps]"` before any BCB run. No SAGE
source changes.

**Risk.** Low — dep sprawl, but these are BCB-specified tasks: if BCB
requires a module, the bench environment should have it.

**Measurement plan.** Re-run the 9 tasks in isolation
(`--limit 148 --filter "108,227,273,274,501,583,590,917,82"` if
filter exists; else full re-run): expect all 9 to succeed-or-fail
for a new (logical) reason, not `ModuleNotFoundError`. Net effect on
the pass rate should be +4 to +6 pp.

## What this data does NOT tell us

1. **Single-run snapshot.** Apr 08 is one run of 148 tasks on the full
   Hard split. BCB Hard is ~150 tasks total so this is roughly the
   complete benchmark, but provider-side variance between runs is real
   (Apr 07 was the same size, Apr 10 and Apr 17 were smoke-only and
   both degraded). None of these conclusions are confirmed by N≥2.
2. **Snippet truncation is pervasive.** 10 tasks are classified from
   warnings not errors because the 200-char cap hid the real error.
   Lever #2 addresses this; until applied, any downstream bucket count
   carries ±10 tasks of uncertainty.
3. **Topology escalation outcome is invisible.** `trace.topology_nodes`
   captures the **first** attempt, not whether the escalation (line
   179 of `bigcodebench_bench.py`) ran or helped. We cannot quantify
   "how much did topology rescue" from this JSONL.
4. **Repair routing is not verified.** AVR claims to use the reasoner
   tier via `ModelRouter.get_config("reasoner")` but the trace does
   not store which provider/model actually handled the repair. A 0 %
   repair rate *could* mean the reasoner call is silently falling back
   to the same budget model that generated the broken code.
5. **`git_sha` field is empty.** Commit provenance for the 45.9 % is
   by calendar date only. Any re-run for comparison must be explicit
   about the commit.
6. **No cross-validation with the official BCB harness.** The local
   eval uses `unittest` subprocess (bigcodebench_bench.py:340) with
   matplotlib forced to `Agg`. Minor divergences from the official
   Docker harness are possible but not measured here.

## Blockers — none

All required data was already on disk. No new generation needed.
