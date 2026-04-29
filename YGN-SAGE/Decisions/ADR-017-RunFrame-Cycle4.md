---
title: ADR-017 RunFrame v0 (Cycle 4, R7)
type: adr
status: shipped
date: 2026-04-29
commits: ["8082b203", "f1557a1f"]
tags: [runtime, run-frame, evidence, provenance]
---

# ADR-017 — RunFrame v0 (Cycle 4, R7 + R7.0.1+R7.0.2)

## Context

After R6 StateCore added typed channels (cycle 3), the pipeline still threaded ad-hoc per-run state through `runner._node_outputs`, `_node_state_deltas`, `_node_costs` dictionaries plus pipeline-level locals. Stage 6 learning consumed mutable side effects. The A14 incident class (off-policy bandit posteriors with no audit trail) was a symptom of this: there was no single typed record of "what happened, in what state version, from which node, under which feature flags, with which evidence".

cgpro 2026-04-29 cycle-3-reassess: "R6 just changed the runner's execution spine: edge-channel partitioning, strict/legacy split, state frames/deltas, state_applied, mode-sensitive NodeStarted.predecessors_by_channel. That is exactly the point where the next architectural move should be a typed per-run object, not more ad hoc side effects."

## Decision

Behind `SAGE_RUN_FRAME=1` (default OFF byte-identical to R6 baseline; ON adds ONE trailing diagnostic event after `final_result`). NEW module `sage/runtime/run_frame/` with 3 files:

- **frame.py**: public frozen dataclasses
  - `RunFrame` — final immutable snapshot returned by `pipeline.run_with_frame()`
  - `NodeRunRecord` — per-node-execution typed record
  - `TopologyRunRef` — topology history (initial vs reroute vs fallback)
  - + status enums (`NodeRunStatus`, `RunStatus`) + `RUN_FRAME_SCHEMA_VERSION="0"`
- **builder.py**: private `_RunFrameBuilder` mutable hot accumulator. Allowlisted env capture (8 keys hardcoded). `node_run_id = f"{topology_epoch}:{node_id}:{attempt}"` prevents overwrite across retries / open_gate / upgrade_model / reroute. `finalize()` returns frozen RunFrame with deep-copied internals.
- **__init__.py**: public API surface.

### Architecture: private builder + public frozen snapshot

cgpro DESIGN locked option (iii) hybrid (NOT mutable public RunFrame):
- `_RunFrameBuilder` private mutable hot accumulator during the run.
- `RunFrame` public frozen final snapshot returned to callers.
- One frozen RunFrame per run, NOT per transition (avoids hot-path allocation churn).
- The R5 JSONL event log remains the durable immutable record.

### Pipeline API

```python
async def pipeline.run(task) -> str  # UNCHANGED, returns str
async def pipeline.run_with_frame(task, budget_usd=None, system_hint=None) -> tuple[str, RunFrame]  # NEW
```

`run()` UNCHANGED to avoid breaking 25+ existing callers. `run_with_frame()` builds and returns frame regardless of `SAGE_RUN_FRAME`. Builder runs always; the env flag controls only the trailing diagnostic emission.

R7.0.2 corrected: `run_with_frame()` initially had narrower signature than `run()` (missing `budget_usd` + `system_hint`). Bench/traced adapters switching entry points lost per-task budget cap and system tier hint. R7.0.2 restored signature parity + regression test using `inspect.signature` to detect future drift.

### Builder ownership: per-call local

cgpro VERIFY round-trip required: `run_with_frame()` builder is a LOCAL variable inside `_run_internal()` (NOT cached on `self`). Concurrent runs on the same pipeline don't share state. Verified via test_run_frame.py's concurrent runs test.

### NEW event type: `run_frame_summary` (12th)

Trailing DIAGNOSTIC event ONLY when `SAGE_RUN_FRAME=1`. Emitted AFTER `final_result`. `parent_event_id == final_result.seq`. Contains `run_frame_schema_version`, `run_frame_hash`, `status`, `node_record_count`, `final_result_seq`, terminal_failure_seq — refs/hashes/counts only, NEVER raw outputs.

**Best-effort emission**: failures during `run_frame_summary` write must NOT change pipeline result. `final_result` is still emitted exactly once before the diagnostic. Existing event types' `schema_version` stays `"1.0"` ; only `run_frame_summary` uses `RUN_FRAME_SCHEMA_VERSION="0"` to track the new event family separately.

### Allowlisted env capture (8 keys, no wildcard)

```python
_ALLOWED_FEATURE_FLAGS = {
    "SAGE_RUN_FRAME", "SAGE_STATECORE", "SAGE_TRACE_JSONL_DIR",
    "SAGE_TRACE_RAW", "SAGE_TRACE_FAIL_CLOSED", "SAGE_DIFF_VERIFIER_MODE",
    "SAGE_ENABLE_PATH6", "SAGE_ORACLE",  # +SAGE_ORACLE in R9
}
_PATH_LIKE_FLAGS = {"SAGE_TRACE_JSONL_DIR"}  # redacted as "<path>"
```

cgpro trap-closed: SAGE_DASHBOARD_TOKEN and other secrets do NOT leak into RunFrame public surface. Wildcard SAGE_* capture explicitly forbidden.

### NodeRunRecord composite key

```python
node_run_id: str = f"{topology_epoch}:{node_id}:{attempt}"
```

`topology_epoch` increments on reroute (record_topology_selected with reason="reroute"). `attempt` counter per `(epoch, node_id)`. This prevents the overwrite class where retry/open_gate/upgrade_model/reroute would silently clobber prior records.

### Refs are event seqs (not inline payloads)

NodeRunRecord stores `node_started_seq`, `node_completed_seq`, `failure_seq`, `controller_decision_seqs`, `state_applied_seqs`, `event_seqs` — pointers into the JSONL trace. Plus cheap local metadata: `output_sha256`, `output_length`, `input_context_hash`. The JSONL is the ground truth ; RunFrame is the typed in-process view.

`emit_*()` methods extended to return `int | None` (the seq written, or None if disabled). Builder uses returned seqs ; never reads private writer internals.

## R7.0.1 — Doc fix (`f1557a1f`)

Stale comment: `event_type: str  # one of the 11 above` → `event_type: str  # one of EVENT_TYPES above (12 as of R7)`. Future additions reference the constant, not a count.

## Consequences

- 19 R7 acceptance tests + 1 cgpro VERIFY round-trip (NodeStarted predecessors_by_channel ON/OFF asymmetry) + 1 R7.0.2 signature parity regression = 21 total in R7.
- RunFrame becomes the foundation for R9 OracleStack (typed evidence surface) and R6.1a delta producers (`runtime_deltas` field added in cycle 6).
- 1826 LOC in R7 + 55 in R7.0.1+R7.0.2 = 1881 total.
- mypy 0/198 (was 195, +3 for new module).

## Related

- [[ADR-014-RuntimeContracts-Cycle1]] — R2 `_run_core` events feed the builder
- [[ADR-015-RuntimeEventLog-Cycle2]] — `emit_*()` returns seq for builder to record
- [[ADR-016-StateCore-Cycle3]] — `RunFrame.state_frames` exposes R6 per-node frames
- [[ADR-018-OracleStack-Cycle5]] — `RunFrameView` (read-only protocol) is what `evaluate()` consumes; `RunFrame.oracle_verdict` field added in R9
- `docs/contracts/runtime-event-log.md` — run_frame_summary event row
- `tests/golden/runtime_events/run_frame_summary.json` — golden fixture
