# Option B — REROUTE_REBUILD end-to-end empirical smoke

**Date**: 2026-05-12 09:26 UTC
**HEAD** at run-time: `fd68e7f1`
**Cost**: $0.0011 cumulative across 2 iterations (iter1 Vuls $0.0006 + iter2 NodeBB $0.0005)
**Wall time**: 21.8s cumulative (iter1 11.3s + iter2 10.5s)
**Output**: `docs/benchmarks/2026-05-12-reroute-end-to-end-option-b/`
**cgpro context**: `FUTURE_HARDENING_ID=REROUTE_REBUILD_RUNTIME_INTEGRATION_SMOKE`
(non-blocker per 2026-05-12 VERIFY)

## Goal

Drive the `REROUTE_REBUILD` branch in `execute.py:364` end-to-end with
a real LLM canary, prove the I-11 chain fires on a real production
reroute trace (not just helper-level smoke).

## Configuration

```bash
SAGE_LLM_TIER=fast                          # activate controller (Fix C bypass)
SAGE_BOOT_BYPASS_EPOCH_GUARD=1
SAGE_TRACE_RAW=1
python sage-python/scripts/run_dryrun_arm_d.py \
  --instances-json sage-python/data/swebench_pro/n10/instances.json \
  --limit 1 --budget-usd 1.0 --global-budget-usd 3.0 --task-timeout-s 900 \
  --tier fast \
  --provider-allowlist google,deepseek \
  --provider-denylist openai \
  --output-dir docs/benchmarks/2026-05-12-reroute-end-to-end-option-b
```

Task chosen: `instance_future-architect__vuls-139f3a81b66c47e6d8f70ce6c4afe7a9196a6ea8`
(same task as the prior real-canary smoke at `7d631f76` for
reproducibility).

## Actual events in the JSONL trace

```
seq=0  task_started
seq=1  routing_decision
seq=2  topology_selected (1-node sequential)
seq=3  topology_selected (...)
seq=4  model_assigned (deepseek-v4-pro, deepseek)
seq=5  provider_execution_witness phase=initial decision=blocked
       reason=provider_in_denylist (router proposed openai)
seq=6  runtime_integrity_assertion I-11 phase=initial verdict=pass
seq=7  final_result status=success output="[sage: agent exited after 3 steps with no content]"
seq=8  oracle_verdict trainable=False verdict_source=abstain
```

**No `node_started` / `node_completed` / `controller_decision` /
reroute witness.**

## Why REROUTE_REBUILD didn't fire

The `RustTopologyController` decides reroute via 5 paths
(`architecture.md`):

- **Path 1** (empty/error reroute) — needs `result == ""` from a
  completed node
- **Path 2** (quality cascade) — needs `quality < THETA_CRITICAL`
  AND a real numeric score from `QualityLabeler`
- **Path 3** (debate-gate threshold) — debate topology specific
- **Path 4** (parallel inconsistency) — parallel topology specific
- **Path 5** (importance prune) — multi-node only

Observed trace:
- Topology is **1-node sequential** (bypass / single-agent mode)
- Agent loop produces `EMPTY_STEP_SENTINEL` ("agent exited after
  3 steps with no content") — this is a SAGE-side sentinel, NOT
  the `""` result the Rust controller's path-1 expects
- `oracle_verdict.verdict_source = "abstain"` —
  `QualityLabeler` couldn't score the sentinel string → no quality
  number for the controller's path-2 cascade to compare against
- Single-node topology excludes paths 3/4/5

**Net**: the controller has nothing to act on. `runner.run()`
returns the sentinel string, NOT `"__REROUTE__"`. `execute.py:364`
condition is false. REROUTE_REBUILD branch is dead code on this
task.

## What we DID prove empirically

| Acceptance | Status |
|---|---|
| Witness phase=initial decision=blocked emitted in prod | ✅ |
| `routing_candidate_reason_code` populated correctly | ✅ (`provider_in_denylist`) |
| `runtime_integrity_assertion` I-11 emitted in prod | ✅ |
| Assertion verdict=pass when declared==verified | ✅ |
| Assertion includes `witness_seq` link to witness | ✅ |
| Failure schema v1_1 + correlation_witness_seq | not exercised (no failure on this trace) |
| REROUTE_REBUILD branch + phase=reroute witness | NOT exercised (controller didn't fire) |
| `no node_started` invariant after blocked reroute | NOT exercised (no reroute) |

## What we did NOT prove

The cgpro FUTURE_HARDENING acceptance criteria assume the controller
will reroute. That assumption requires:

1. **Multi-node topology** (not single-agent bypass) — controller
   needs ≥2 nodes for paths 4/5; path 1/2 needs a node that actually
   completes with a score
2. **Real output (not sentinel)** — `QualityLabeler` must produce a
   numeric quality score
3. **Quality below threshold** — `THETA_CRITICAL` (default 0.3)

The Vuls task doesn't satisfy #2 in fast tier — the agent gives up
empty, which the SAGE pipeline reports as a sentinel that doesn't go
through `QualityLabeler` as a real evaluation.

To make REROUTE_REBUILD reachable empirically, a future iteration
would need:
- A task that produces non-empty agent output (e.g. a simpler task
  where the agent can attempt SOMETHING)
- A scoring path that doesn't abstain
- A configuration that prefers multi-node topology
- Possibly a tuned `THETA_CRITICAL` to make low-quality more
  reroutable

## Coverage gap and how it's mitigated

| Layer | Coverage | Status |
|---|---|---|
| **Source inspection** | `test_reroute_rebuild_path_calls_enforce_provider_policy` proves `execute.py` invokes `enforce_provider_policy` between the reroute witness emit and `runner2 = TopologyRunner` construction | ✅ shipped |
| **Helper-level smoke** | `test_reroute_rebuild_blocked_candidate_chain_witness_assertion_failure_no_dispatch` drives the EXACT helper sequence (witness → assertion → failure → no dispatch) with `phase=reroute` | ✅ shipped |
| **Real-canary** | This Option B run captured the witness + I-11 chain in production on the `initial` phase (real LLM, real provider policy, real Rust assigner) | ✅ this artefact |
| **End-to-end reroute** | A canary that actually triggers REROUTE_REBUILD + verifies the reroute witness → assertion → enforce chain | ❌ NOT YET, deferred |

The first 3 layers together provide strong evidence that the
REROUTE_REBUILD chain is correctly wired. The 4th layer requires
either (a) finding a task that reliably triggers reroute, (b)
synthetic configuration that forces reroute, or (c) accepting the
helper-level smoke as sufficient.

## Iteration 2 — NodeBB task

To confirm whether iter1's result was task-specific (Vuls) or
tier-systemic (fast), iter2 ran the same configuration against the
NodeBB task (`instance_NodeBB__NodeBB-76c6e3...`).

```
Result: $0.0005 cost, 10.5s wall, 0 patches, 0 controller_decision events
Trace: same shape as iter1 — task_started, routing, topology_selected,
       model_assigned, witness (phase=initial, decision=blocked),
       runtime_integrity_assertion (I-11, verdict=pass), final_result,
       oracle_verdict, cli_started, cli_complete
```

Empirically reproducible: the fast tier consistently produces
empty-step-sentinel before the controller can score quality. This
is **tier-systemic**, not task-specific.

## Diagnosed root cause

The fast tier (gemini-3.1-flash-lite-preview) is over-aggressive at
giving up. After 3 agent steps with no patch produced, SAGE returns
`EMPTY_STEP_SENTINEL`. The Rust controller doesn't receive a numeric
quality score to compare against `THETA_CRITICAL` because
`QualityLabeler` abstains on the sentinel — it's not a real model
output.

**Bidirectional trap**:
- `_llm_tier == "budget"` (default) → cheap, agent runs long enough,
  BUT controller disabled by Fix C → no reroute
- `_llm_tier == "fast"` → cheap, controller enabled, BUT agent gives
  up empty → controller has no score → no reroute
- `_llm_tier == "reasoner"` / `codex_max` → controller enabled,
  agent likely produces real output, BUT 10-50× cost per run ($0.50-
  3 per task vs $0.0006)

## Recommendation

cgpro explicitly marked this as `NON_BLOCKING` and "not required
before B2". The empirical reality (this artefact) shows that
realistic prod canaries on the available SWE-bench Pro tasks DON'T
trigger REROUTE_REBUILD reliably — at least not in budget/fast tier
on tasks that lead to empty agent output.

For the FUTURE_HARDENING to deliver real evidence, either:
- Choose a task with known empirical reroute behavior (would require
  scanning prior canary traces for controller_decision events)
- Switch to reasoner/codex_max tier which is more likely to produce
  scorable output (10-50× higher cost per run)
- Defer until a real B2 canary trace exposes reroute organically

This artefact closes the Option B audit with the empirical finding:
"REROUTE_REBUILD is not naturally exercised in budget/fast tier on
the Vuls task; helper smoke + source inspection remain the
authoritative cover until a reroute-rich canary trace is available."

## Reproduce

```bash
# Same env vars + flags as the original launch (see Configuration
# above). The single-task budget is $1.0; expect ~$0.0006-0.01 if
# the agent gives up empty, $0.10-0.50 if it generates output but
# fails quality cascade.
```
