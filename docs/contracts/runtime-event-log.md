# RuntimeEventLog event contract

This doc + the JSON fixtures at `sage-python/tests/golden/runtime_events/` form the canonical contract for the JSONL events emitted by `RuntimeEventLog` (R5, cycle 2) under each feature-flag mode.

The contract is enforced by `tests/test_runtime_event_contracts.py`. Breaking changes here require an explicit `schema_version` bump and an updated golden fixture.

## Mode-aware contract matrix

| Field on event | SAGE_STATECORE unset / 0 | SAGE_STATECORE=1 |
|---|---|---|
| `predecessors_by_channel` (NodeStarted.payload) | **forbidden** | required |
| `state_applied` event type | **never emitted** | emitted on each State-channel reduction |
| Failure event with `kind="unknown_edge_type"` | **never emitted** (warning log only) | emitted before raising ValueError |
| Failure event with `kind="controller_reroute"` | emitted | emitted |
| Failure event with `kind="budget_exceeded"` | emitted | emitted |
| All other events (TaskStarted, RoutingDecision, TopologySelected, ModelAssigned, NodeStarted core fields, NodeCompleted, ControllerDecision, Failure, Budget, FinalResult) | byte-identical to R5 baseline | byte-identical to R5 baseline |

| Field on event | SAGE_RUN_FRAME unset / 0 | SAGE_RUN_FRAME=1 |
|---|---|---|
| `run_frame_summary` event type | **never emitted** | emitted as trailing diagnostic AFTER `final_result` |

| Field on event | SAGE_ORACLE unset / true-ish (default-on, cycle 7+) | SAGE_ORACLE=0/false/off/no/disable/disabled |
|---|---|---|
| `oracle_verdict` event type | emitted after `final_result` with `parent_event_id == final_result.seq` | **never emitted** |
| `controller_decision.payload` | forced **allowlist-only** payload (9 keys: `node_id`, `action`, `target_node_id`, `gate_source_id`, `gate_target_id`, `quality_score`, `quality_source`, `threshold_band`, `reason_code`) — no free-form `reason`, slug-constrained `reason_code`, clamped `quality_score` | legacy redacted payload only emitted when `SAGE_TRACE_RAW=1` |

**Cycle-7 default-on flip (2026-04-29, commit `128e1b89`)**: the contract above is the new default. The legacy "byte-identical to R5 baseline" guarantee for `SAGE_ORACLE` survived for `SAGE_STATECORE` and `SAGE_RUN_FRAME` only — those are still opt-in. The `oracle_verdict` and forced `controller_decision.payload` events ship by default; operators turn them OFF via the kill-switch. cgpro 2026-04-30 cycle-7 VERIFY round-1 PUSH BACK closed by tightening the forced payload to allowlist-only (no free-form `reason` leak).

The remaining OFF-mode columns (StateCore / RunFrame) are still **byte-identical baseline guarantees**. Any field added unconditionally there breaks downstream callers (e.g., `protocols/a2a_server.py`, `scripts/run_masbench_traced.py`) that may be parsing the JSONL. Behavior must match the previous milestone exactly when the gating flag is unset/0.

## Final-event semantics (R9+)

`final_result` is the LAST **business** event of any run. Under default-on `SAGE_ORACLE` (unset or any non-kill-switch value, cycle 7+), exactly ONE `oracle_verdict` event is emitted after `final_result` and before any optional `run_frame_summary`; its `parent_event_id` MUST equal `final_result.seq`. Under `SAGE_RUN_FRAME=1`, exactly ONE optional trailing `run_frame_summary` diagnostic event MAY follow `final_result` or `oracle_verdict`. The summary's `parent_event_id` remains `final_result.seq`. The summary is best-effort: any sink failure during its emission MUST NOT change the pipeline result. `final_result` is still emitted exactly once per run regardless of oracle or summary success.

## Event-type catalog (15 types)

| Event | Source component | Pipeline scope | When emitted |
|---|---|---|---|
| `task_started` | pipeline | pipeline | once per run, before any other event |
| `routing_decision` | pipeline | pipeline | after CLASSIFY stage |
| `topology_selected` | pipeline | pipeline | after SELECT TOPOLOGY stage |
| `model_assigned` | pipeline | pipeline | per-node after ASSIGN MODELS |
| `node_started` | topology_runner | runner | before each node executes |
| `node_completed` | topology_runner | runner | after each node succeeds |
| `controller_decision` | controller | runner | per-node controller `evaluate_and_decide` |
| `failure` | topology_runner | runner | per-node failure (provider, controller_reroute, budget_exceeded, unknown_edge_type, etc.) |
| `budget` | topology_runner | runner | when budget threshold crossed |
| `state_applied` | topology_runner | runner | per-node reduction in StateCore strict mode (R6, behind SAGE_STATECORE=1) |
| `final_result` | pipeline | pipeline | once per run, last BUSINESS event, with `status: success` / `failure` / `budget_exceeded` |
| `oracle_verdict` | pipeline | pipeline | OracleStack verdict event (R9, **default-on since cycle 7 — `SAGE_ORACLE` unset = ON**); follows `final_result` with `parent_event_id == final_result.seq`; payload is `OracleVerdict.to_dict()` with no raw output. Kill-switch via `SAGE_ORACLE=0|false|off|no|disable|disabled`. |
| `run_frame_summary` | pipeline | pipeline | trailing DIAGNOSTIC event (R7, behind SAGE_RUN_FRAME=1); follows `final_result` with `parent_event_id == final_result.seq`; best-effort emission — sink failure here MUST NOT change pipeline result |
| `prompt_injection_detected` | pipeline | pipeline | prompt-injection detector observation event; additive RuntimeEventLog event with v1 payload schema. Its presence in the catalog does not make prompt-injection detection default-on. |

## Core fields (always present on every event)

```python
schema_version: Literal["1.0"]
run_id: str            # canonical 26-char Crockford-Base32 ULID
trace_id: str          # = run_id in v0
parent_event_id: int | None  # parent's seq; None for task_started
seq: int               # 0-indexed, contiguous, strictly monotonic per run
timestamp_ns: int      # time.time_ns()
event_type: str        # one of EVENT_TYPES above (15 as of 2026-05-09)
source_component: str  # pipeline | topology_runner | controller | model_assigner | provider_pool
task_hash: str         # sha256(task_text), 64 lowercase hex
payload_hash: str      # sha256({schema_version, event_type, payload}), 64 lowercase hex
redaction_state: str   # redacted | raw | partial | none_applicable
```

## R6-prep no-op fields (placeholder, currently unused)

```python
edge_type: str | None       # always None in R5/R6 v0; R6.1 may populate
channel: str | None         # always None in R5/R6 v0; R6.1 may populate
state_version: int | None   # always None in R5/R6 v0; R6.1 may populate
```

These fields are reserved for future expansion. They are dropped from `to_dict()` output when None to keep JSONL slim.

## Schema versioning policy

- `SCHEMA_VERSION = "1.0"`. Bump major version (`"2.0"`) on breaking changes:
  - Removing a required field.
  - Changing a field's type (e.g., int → str).
  - Renaming an event type.
- Bump minor version (`"1.1"`) on additive backward-compatible changes:
  - Adding a new event type.
  - Adding a new optional field that absent readers can ignore.
- Adding a field unconditionally to OFF-mode events is a **breaking change** — readers expect byte-identical R5 schema.

## Golden fixtures

`sage-python/tests/golden/runtime_events/`:
- `oracle_verdict.json` - oracle_verdict event contract (default-on since cycle 7; suppressed only when `SAGE_ORACLE=0|false|off|no|disable|disabled`)
- `controller_decision.json` - controller_decision event contract; allowlist-only forced payload under default-on (cgpro 2026-04-30 cycle-7 VERIFY round-1)
- `statecore_off_node_started.json` — NodeStarted contract in legacy mode
- `statecore_on_node_started.json` — NodeStarted contract in strict mode
- `state_applied.json` — state_applied event contract (only emitted in strict mode)
- `run_frame_summary.json` — run_frame_summary trailing diagnostic event (only emitted under SAGE_RUN_FRAME=1)

The `_required_always` / `_required_in_payload` / `_forbidden_in_payload_when_off` keys in each fixture are read by `test_runtime_event_contracts.py` to validate live JSONL output against the locked schema.

## Adding a new event type or field

1. Update `EVENT_TYPES` in `sage/runtime/event_log/schema.py`.
2. Add the private `_<event>` dataclass in `sage/runtime/event_log/events.py`.
3. Add the `emit_<event>` method in `sage/runtime/event_log/writer.py`.
4. Add a golden fixture in `tests/golden/runtime_events/<event>.json`.
5. Update this doc's catalog + the mode-aware contract matrix.
6. If the field is mode-conditional, add the row to the matrix above + extend `test_runtime_event_contracts.py` to enforce ON/OFF asymmetry.
7. Bump `SCHEMA_VERSION` per the policy above.

## References

- R9 cycle 5 spec - `.tmp/cgpro_r9_design_locked_spec.md`
- R5 cycle 2 spec — `.tmp/cgpro_r5_design_locked_spec.md`
- R6 cycle 3 spec — `.tmp/cgpro_r6_design_locked_spec.md`
- R7 cycle 4 spec — `.tmp/cgpro_r7_design_locked_spec.md`
- cgpro 2026-04-29 cycle 3 reassess — recommended this contract hardening as R6.0.1 follow-up
- cgpro 2026-04-29 R7 verify — required this doc update before R7 SHIP (12 event types, final-event semantics)
- cgpro 2026-04-30 cycle-7 VERIFY round-1 PUSH BACK — flipped SAGE_ORACLE row to default-on; tightened `controller_decision.payload` to allowlist-only (no free-form `reason`); operator-friendly kill-switch values (`disable`, `disabled`)
