# Learning Side-Effect Ledger v0

The Learning Side-Effect Ledger is an audit-only JSONL sidecar emitted next to
`RuntimeEventLog` traces. It records the learning side-effect decisions already
made by the runtime gates, so reviewers can verify that bandit, MAP-Elites,
online evolution, and training-memory paths were not updated from unverified
outputs.

It does **not** authorize learning. The authoritative gates remain the
OracleStack, `record_outcome_checked()`, and the existing runtime-integrity
invariants in `docs/contracts/runtime-integrity-ledger.md`.

## Scope

- File name: `learning_side_effects.jsonl`
- Python package: `sage.runtime.credit_assignment`
- Validator: `python -m sage.runtime.credit_assignment.validate <trace_dir> [--run-id <id>]`
- Schema version: `learning_side_effect.v0`
- Runtime behavior: fail-open for sidecar writes; validation failures fail
  evidence gates, not the live task execution path
- Protocol behavior: not emitted on `sage run --jsonl` stdout, not counted in
  the 15 `RuntimeEventLog` event types, and not part of CLI protocol v0

## Record Contract

Each line is one canonical JSON object with:

- `seq`, `prev_record_hash`, `record_hash`: monotonic hash chain for the sidecar.
- `run_id`, `trace_id`, `task_hash`: copied from the active `RuntimeEventLog`.
- `parent_event_refs`: references to prior runtime events by
  `{event_type, seq, payload_hash}`.
- `oracle_verdict_ref`: oracle verdict summary and payload hash when an oracle
  verdict exists.
- `policy_ref`: bandit decision id, routing event ref, policy snapshot hash,
  candidate-set hash, and the explicit note that Thompson propensity is not
  currently logged.
- `subject`: finalized model/template/topology subject when available.
- `side_effect`: one of:
  `bandit_record_outcome`, `bandit_cancel_pending`,
  `map_elites_record_outcome`, `online_evolution_should_evolve`,
  `online_evolution_evolve`, `training_memory_consolidate`.
- `decision`: `allowed`, `blocked`, `skipped`, or `failed`.
- `reason_code`: allowlisted reason code.
- `gate`: `oracle_enabled`, `oracle_trainable`, `allow_training_updates`, and
  `quality_source`.
- `metrics`: finite `quality`, `cost_usd`, and `latency_ms` values or null.
- `result_summary`: redacted bounded object; raw prompt/output/result keys are
  forbidden.

## Validator Guarantees

`validate_trace_dir()` checks:

- sidecar exists and is non-empty;
- each record matches `learning_side_effect.v0`;
- `seq` and hash chain are monotonic;
- every `parent_event_refs[]` item exists in the sibling `RuntimeEventLog` and
  its `payload_hash` matches;
- `oracle_verdict_ref` points to a real `oracle_verdict` event with matching
  hash;
- an `allowed` oracle-on learning update requires an oracle verdict ref with
  `trainable=True`;
- `bandit_cancel_pending` is treated as an allowed safety side-effect, not a
  learning update.

## Integrity Position

This sidecar reinforces invariant 2 (Oracle evidence) and invariant 6 (Bandit
attribution). It deliberately does not add an 11th invariant because it is not a
new authorizing label. If future code consumes ledger records to authorize
updates, that would be a new runtime-integrity invariant and must be registered
in `docs/contracts/runtime-integrity-ledger.md` before shipping.
