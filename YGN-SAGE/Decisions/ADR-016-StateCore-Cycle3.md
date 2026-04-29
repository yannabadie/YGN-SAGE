---
title: ADR-016 StateCore v0 (Cycle 3, R6)
type: adr
status: shipped
date: 2026-04-29
commits: ["bc481588", "4f38a51c", "af26d75c"]
tags: [runtime, state, edge-channels, statecore]
---

# ADR-016 — StateCore v0 (Cycle 3, R6 + R6.0.1)

## Context

Pre-R6 `TopologyRunner._gather_predecessor_context(node_idx)` aggregated ALL predecessor outputs as text, blind to edge type. Rust `topology_graph.rs:22-28` exposed `EdgeType::{Control, Message, State}` + `edges_of_type(EdgeType)` method but Python never called these. The Rust IR was right ; the Python runtime ignored it.

CGPRO.md R6 sketched StateCore as the next strategic move after RuntimeContracts (cycle 1) and event log (cycle 2). cgpro 2026-04-28 cycle-2-reassess locked the framing: "R6 should now make the runner's data-flow semantics truthful" — replace text aggregation with edge-type-aware Control / Message / State channels.

## Decision

Behind `SAGE_STATECORE=1` (all-or-nothing flag, default OFF byte-identical to legacy). NEW module `sage/runtime/state/` (4 files, 317 LOC):

- **frame.py**: public frozen dataclasses `StateFrame`, `StateDelta`, `StateApplyResult`, `EvidenceRef` (`slots=True`).
- **reducer.py**: `apply_delta(frame, delta, *, source_node_id, raise_on_conflict)` and `apply_deltas(frame, [(src_id, delta), ...])` (atomic batch, sort by source_node_id, all-or-nothing). Plus `normalize_assumption_id` helper.
- **errors.py**: `StateConflict(RuntimeError)`.
- **__init__.py**: public exports.

### Channel partition algorithm

```python
def _partition_incoming_edges(graph, node_idx, *, legacy_mode: bool) -> tuple[control_preds, message_preds, state_preds]:
    """Uses graph.get_edges() (R5.1 contract — Python-visible API).
    
    Legacy mode (SAGE_STATECORE unset/0):
        control → control + message (BOTH — preserves text aggregation)
        message → message
        state   → state
        unknown → message + warning log only (NO new failure event — preserves byte-identical OFF)
    
    Strict mode (SAGE_STATECORE=1):
        control → control only
        message → message only
        state   → state only
        unknown → emit Failure event + raise ValueError
    """
```

### 9-row reducer conflict table (cgpro locked)

| Case | v0 behavior |
|---|---|
| Same constraint added twice | idempotent accept |
| Remove missing constraint | accept, no field change |
| Add+remove same constraint in same batch | conflict |
| Add+invalidate same assumption in same batch | conflict |
| Invalidate assumption | remove from `assumptions`, add to `invalidated_assumptions` |
| Same entity field updated to same value by multiple sources | idempotent accept |
| Same entity field updated to different values by sibling sources | conflict |
| Decision without evidence AND without source_node_id | conflict |
| Tool fact without tool_call_id AND without node_id/source_node_id | conflict |

Conflict semantics: `applied=False`, `conflicts` non-empty, frame UNCHANGED. NEVER write-wins, NEVER silently mutate, NEVER raise (caller decides via `raise_on_conflict=True`).

### NEW event type: `state_applied` (11th)

`_StateApplied(_EventCore)` with `target_node_id`, `source_node_ids`, `before_version`, `after_version`, `delta_count`, `conflict_count`, `applied: bool` (cgpro v0 schema correction — disambiguates accepted-no-op from blocked-conflict), `invalidated_assumption_ids`. Emitted only when `SAGE_STATECORE=1`.

### NodeStarted partition (cgpro VERIFY round-trip catch)

When `SAGE_STATECORE=1`, NodeStarted payload carries `predecessors_by_channel: {"control": [...], "message": [...], "state": [...]}` partition. When OFF, the field is absent entirely (byte-identical R5 schema preserved).

### Two cgpro traps fixed

1. **Planner injection channel-aware** (`_maybe_planner_injection`): pre-fix used `get_predecessors()` which discards edge types, so planner output could leak through control/state edges into downstream system prompt. R6 fixed: filter predecessors by message channel via `_partition_incoming_edges` in strict mode.
2. **Sibling state delta atomic merge**: parallel sibling nodes don't see each other's state. Downstream join node merges siblings via `apply_deltas` with all-or-nothing semantics ; if any sibling delta conflicts, the whole batch is rejected.

## R6.0.1 — Contract snapshot tests (`4f38a51c..af26d75c`, +523 LOC)

Pre-cycle-4 hardening cgpro recommended after R6 verify round-trip caught the NodeStarted ON/OFF asymmetry. Type stubs alone don't catch the class of bug where a field type-checks fine but is forbidden in legacy mode.

NEW:
- **docs/contracts/runtime-event-log.md** — canonical event contract doc (mode-aware matrix, 13-event catalog by R7, schema versioning policy, golden-fixture protocol).
- **tests/golden/runtime_events/statecore_off_node_started.json** — declares `_forbidden_in_payload_when_off: ["predecessors_by_channel"]`.
- **tests/golden/runtime_events/statecore_on_node_started.json** — declares `_required_in_payload + _required_predecessors_by_channel_keys`.
- **tests/golden/runtime_events/state_applied.json** — declares `applied: bool` invariant + value-type contracts.
- **tests/test_runtime_event_contracts.py** — 5 contract tests enforcing ON/OFF asymmetry directly + `EVENT_TYPES` catalog drift detector.

This is the upstream guard pattern that prevents future cycles from accidentally adding fields that violate byte-identical OFF.

## Consequences

- 13 R6 acceptance tests + 5 R6.0.1 contract tests = 18 total.
- StateDelta production from nodes is **deferred to R6.1** (this v0 ships with empty `StateDelta()` from all nodes — channel partition + reducer correctness is the v0 contract; delta production is the next cycle).
- `_gather_predecessor_context` STAYS for OFF mode (byte-identical legacy preserved). Strict mode replaces it with `_assemble_node_inputs` returning `(message_text, state_frame, control_ready)`.
- Spec oracle (R9) later consumes `state_frames` for contradiction detection.
- 1340 LOC added in R6 + 523 in R6.0.1.

## Related

- [[ADR-014-RuntimeContracts-Cycle1]] — R2 `_run_core` is where channel partitioning is wired
- [[ADR-015-RuntimeEventLog-Cycle2]] — state_applied event extends the R5 taxonomy
- [[ADR-017-RunFrame-Cycle4]] — R7 `RunFrame.state_frames` exposes per-node frames
- [[ADR-018-OracleStack-Cycle5]] — `_spec_oracle` uses `view.state_frames` + R6.1a-produced `formal_verifier/assumption_invalidated` deltas
- `docs/contracts/runtime-event-log.md` — mode-aware contract matrix (SAGE_STATECORE row)
- `roadmap.md` — R6.1a sequencing note
