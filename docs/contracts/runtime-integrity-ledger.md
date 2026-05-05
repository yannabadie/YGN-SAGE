# Runtime Integrity Ledger

**Status**: documentary contract (no code refactor in cycle 8 / 9). Per cgpro 2026-04-30 architect review Q-A: *"créer un registre documentaire/testable, pas un gros move"*.

After cycle-7 + cycle-8 R6.1c + cycle-8 A14 + cycle-9 A14b, YGN-SAGE has accreted 6 invariant-binding mechanisms across `sage-python/src/sage/runtime/`, `sage-python/src/sage/`, and `sage-core/src/topology/`. They are **conceptually a Runtime Integrity subsystem** but **physically distributed** to keep coupling local. This ledger is the cross-reference contract.

## The 9 invariants

Pattern that emerged from 6 cycles of "declared ≠ verified" traps (cycle-7 contract drift, cycle-8 R6.1c raw-leak vs audit policy drift, cycle-8 A14 epoch ≠ provenance, cycle-9 A3 timeout enforcement under host suspend, cycle-9 α telemetry blank-field self-deception, cycle-12 prelude pi-mono pivot CLI surface): **any label that authorizes a side-effect or learning decision must be bound to verified content, schema, provenance, or executable proof.**

| Invariant | Declared label | Verified content | Side-effect blocked if invalid |
|---|---|---|---|
| **Event payload schema** | `event_type` + `payload_schema_version` (envelope) | allowlist `field_specs` + canonical fixture + max_utf8_bytes | event emission (writer raises `EventLogSchemaError`); validator audit acceptance |
| **Oracle evidence** | `OracleVerdict.trainable` (True/False) | structured `EvidenceRef.evidence_hash` SHA-256 + producer schema (`payload_schema_version` per producer) | bandit / MAP-Elites / online-evolution / training-memory updates |
| **Posterior epoch** | `~/.sage/posterior_epoch.json.epoch` (integer) | `topology_state_manifest.json.state_files[].sha256` over A14 state file bytes | `TopologyEngine::load_state` / `save_state` (Rust + Python preflight) |
| **Contaminated backup** | `_CONTAMINATED.json.contaminated=true` (operator-readable poison-pill) | `audit_dump_sha256` cross-reference to immutable audit MANIFEST.json | normal load (any contaminated marker present in active state dir = fail-closed) |
| **RunFrame summary** | `run_frame_summary.payload.parent_event_id` | `final_result.seq` consistency (parent_event_id == final_result.seq) | diagnostic trust (downstream `path_e_validate` event-order check) |
| **Bandit attribution** | `bandit_decision_id` from Stage-0 `SystemRouter.route_integrated()` | `SystemRouter.record_outcome_checked()` verifies pending `(model_id, template)` against executed `(model_id, template)` | bandit posterior update (mismatch emits `bandit_attribution_mismatch` and skips recording) |
| **Timeout enforcement** | per-task `timeout_s` declared at bench config + bound to `asyncio.wait_for(timeout=...)` | `elapsed_wall_ms <= timeout_s × grace_factor` (wall-clock measured via `time.time()`, which advances during OS suspend; default `grace_factor=2.0`) | pass-rate aggregation (tasks with `host_suspend_or_event_loop_stall=true` emit `TASK_ABORT reason=host_suspend_detected` and are excluded from gate-quality stats); the run is marked non-gate-quality |
| **Control-surface completeness** | bench `TASK_END.control_surface` claims topology mechanism (`executed_template`, `node_count`, `selected_template`) | when `node_count > 0`, `executed_template` MUST be non-empty AND `dag_features.{omega,delta,gamma}` MUST be present | "topology X → topology Y" mechanism claims (a blank `executed_template` invalidates any robust-vs-sequential narrative); downstream replay analysis must reject ledger entries failing this contract |
| **CLI protocol versioning** | every JSONL frame on `sage run --jsonl` stdout carries `protocol_version` (currently `"v0"`) and the first frame is `cli_started` | the `protocol_version` MUST equal `sage.cli.run.CLI_PROTOCOL_VERSION` AT EMIT TIME — drift between the constant and emitted frames means the backend and the spec disagree; frontends fail-close on receipt of a mismatch | external consumer interpretation of YGN-SAGE runtime decisions: a frontend that sees a different `protocol_version` than the one it speaks MUST refuse to render the stream (do NOT attempt graceful degradation — bytes after a version mismatch could mean anything). The runtime contract surface MUST NOT become a side-channel that bypasses cycle-7+ guarantees. |

## Module cross-reference

| Invariant | Primary module (Python) | Primary module (Rust) | Tests |
|---|---|---|---|
| Event payload schema | `sage/runtime/event_log/payload_schemas.py` | n/a (Python-emitted, Rust-consumed via PyO3 if any) | `tests/test_payload_schemas.py` (18) + `tests/test_runtime_event_contracts.py` (forced payload contract) |
| Oracle evidence | `sage/runtime/oracle/_oracles.py`, `sage/runtime/evidence/producers/*.py` | n/a | `tests/test_oracle_*.py`, evidence producer round-trip JSON pairs |
| Posterior epoch | `sage/posterior_epoch.py` | `sage-core/src/topology/posterior_epoch.rs` | `tests/test_posterior_epoch.py` (Python), Rust unit tests in `posterior_epoch.rs` |
| Contaminated backup | `sage/ops/a14_reset.py` | n/a (Python ops surface) | `tests/test_a14_reset.py` |
| RunFrame summary | `sage/runtime/run_frame/__init__.py` | n/a | `tests/test_run_frame.py` |
| Bandit attribution | `sage/pipeline.py`, `sage/runtime/event_log/payload_schemas.py` | `sage-core/src/routing/system_router.rs` | `tests/test_pipeline_bandit_causality.py`, `system_router::tests::test_record_outcome_checked_*` |
| Timeout enforcement | `sage/bench/watchdog.py`, `sage/bench/event_ledger.py`, `sage/bench/bigcodebench_bench.py:run` | n/a (suspend is host-OS, not Rust; `sage/bench/keep_awake.py` is the Windows-side mitigation) | `tests/test_bench_watchdog.py` (7), `tests/test_event_ledger.py::test_task_abort_event_marks_excluded`, `tests/test_bench_host_suspend_integration.py` (end-to-end, γ.2) |
| Control-surface completeness | `sage/bench/bigcodebench_bench.py:_capture_control_surface` (consumer), `sage/pipeline.py` `BenchContext.executed_template` / `bandit_template` / `topology_id` / `dag_features` (producer) | n/a (bench-layer contract; pipeline ctx fields are populated by Stage 2/3) | `tests/test_bench_host_suspend_integration.py::test_normal_task_emits_task_end` (asserts `topology_id` / `selected_template` / `executed_template` / `dag_omega/delta/gamma` are present in `control_surface`) |
| CLI protocol versioning | `sage/cli/run.py` (`CLI_PROTOCOL_VERSION` constant + `_emit_cli_event` envelope) | n/a (CLI surface is Python-emitted; pi-mono adapter is TypeScript and consumes the protocol read-only) | `tests/test_runtime_event_contracts.py::test_cli_protocol_version_is_locked` (asserts `CLI_PROTOCOL_VERSION == "v0"` AND every emitted frame carries that string verbatim — cycle-12 prelude). Specification: `docs/contracts/SAGE_CLI_PROTOCOL.md`. |

## Boundary against accidental coupling

These 9 invariants are **conceptually a family** but **physically deliberately not consolidated** under a single `sage/runtime/integrity/` umbrella. Reason (cgpro Q-A verdict 2026-04-30):

> "payload_schemas.py est naturellement couplé à runtime/event_log, tandis que posterior_epoch est naturellement couplé à topology et aux fichiers bandit_state.db, archive_state.db, engine_extras.json. Un refactor physique maintenant créerait surtout churn/import risk sans benchmark gain."

Phase 2 / v0.2 may add re-export aliases (`sage.runtime.integrity.epoch`, `sage.runtime.integrity.schemas`) without moving the actual files. **Do not relocate modules in cycle 9.**

## Adversarial threats this ledger defends against

The 5 traps surfaced by cgpro across cycle-7 / cycle-8 / cycle-9 VERIFY rounds plus the cycle-12 prelude pre-emption, all in the "declared ≠ verified" class:

1. **Cycle-7**: `SAGE_ORACLE` declared default-on in code, but `runtime-event-log.md` contract docs still said "ONLY emitted when SAGE_ORACLE=1" (closed at `f3a89631` via stale-phrase lint test).
2. **Cycle-8 R6.1c round-1**: `controller_decision.payload.reason` declared "safe" (forced under default-on), but redaction layer was credential-only — no allowlist, no PII ban. Audit mode accepts legacy `reason` while raw-leak scanner hard-rejects it (closed at `9944674e + 49648263` via allowlist + Option A doc disclosure).
3. **Cycle-8 A14 round-1**: `posterior_epoch.json.epoch=1` declared "fresh epoch", but no binding to the actual DB bytes. Operator copy-restoring `bandit_state.db` from contaminated backup left the epoch label valid while the content was poisoned (closed at `f9521616` via `topology_state_manifest.json` SHA-256 binding).
4. **Cycle-9 A3 N=50 abort 2026-05-04**: per-task `task_timeout=120s` declared and bound to `asyncio.wait_for(timeout=120)`, but Windows Modern Standby S0 DRIPS suspended the asyncio loop along with the process. On wake, BCB/273 reported `elapsed_wall_ms=20278211` (5h 38min) without firing the timeout — the loop's internal timer counts loop ticks, not wall-clock. Closed at commits `b44156e7` (wall-clock watchdog using `time.time()` which advances during suspend) + `0036217b` (per-task `TASK_ABORT reason=host_suspend_detected` event in the bench ledger; tasks above `timeout × grace_factor` are excluded from gate-quality pass-rate). cgpro recovery analysis 2026-05-04 (conv `cgpro_a3_recovery_20260504`).
5. **Cycle-9 α telemetry self-deception 2026-05-04**: bench `control_surface.executed_template` was sourced from `trace.topology_id` (which is a ULID, e.g. `01KQQM93`) instead of the actual template name from `BenchContext.executed_template` / `bandit_template`. The α paired diagnostic ledger reported `node_count: 5 → 3` for BCB/82 between configs (true), then the post-hoc analysis claimed `robust → sequential` (NOT supported by the ledger because `executed_template` was always empty). cgpro round-2 review 2026-05-04 caught this: "executed_template is empty in both records. The data supports '5-node topology → 3-node topology', not specifically 'robust → sequential' unless you have a second source outside this NDJSON." Closed at commit (this PR) by routing `bigcodebench_bench.py` `trace` capture to `ctx.executed_template` and `ctx.bandit_template` directly + adding the `Control-surface completeness` invariant above. The α post-mortem analysis at `.tmp/paired_diagnostic_n8_analysis.md` was relabeled non-gate.
6. **Cycle-12 prelude pi-mono pivot CLI surface 2026-05-05 (pre-emptive)**: the new `sage run --jsonl` machine-readable backend (commit `d09bed4d`) opens a NEW external surface — TypeScript front-ends (pi-mono adapter shipping in cycle-13) consume YGN-SAGE runtime decisions through subprocess + JSONL. cgpro consultation `cgpro_pi_mono_pivot_20260505` (Option 1 verdict) flagged the risk class explicitly: *"Le CLI ne doit pas devenir une side-channel qui bypasse les garanties cycle-7+."* If the protocol drifts silently — backend bumps `CLI_PROTOCOL_VERSION` from `"v0"` to `"v1"` without the spec changing, OR the spec bumps but the constant doesn't — the frontend may render mis-attributed events (e.g. show "topology=sequential" when the backend meant a different shape). Closed pre-emptively by **invariant 9 above + regression test** `tests/test_runtime_event_contracts.py::test_cli_protocol_version_is_locked` (asserts the constant matches the spec literal AND that emitted frames carry it verbatim). Source: `docs/contracts/SAGE_CLI_PROTOCOL.md` (the spec) ↔ `sage/cli/run.py:CLI_PROTOCOL_VERSION` (the constant). The cycle-13 npm adapter (`clients/pi-ygn-sage/`) MUST pin its expected protocol version explicitly and fail-close on mismatch.

**Cycle-9+ design principle**: any new "label authorizes side-effect" code path MUST register here with all 4 columns filled BEFORE the side-effect ships. This is the architectural pattern (cgpro 2026-04-30):

> "tout label qui autorise un side-effect ou une décision d'apprentissage doit être lié à un contenu vérifié, un schéma, une provenance, ou une preuve exécutable."

## Maintenance discipline

- When adding a new invariant: append a row to **both** tables (the invariant table AND the module cross-reference). Wire a regression test that proves the side-effect is blocked when the verification fails.
- When changing an existing invariant's verified-content schema: bump the schema version (per `payload_schemas.py` discipline, e.g. `v1 → v2_X`) and provide migration/inference rules. Old traces remain readable in audit mode, new emissions strict-current.
- This ledger is referenced from `CLAUDE.md` directive #8 (A14 guard) and should be referenced from any future directive adding a new invariant.

## References

- cgpro 2026-04-30 architect review (saved at `.tmp/cgpro_architect_review_finaltext.md`, conv `cgpro_architect_review`)
- ADR-018 / ADR-019 (Runtime cycle 5/6 design)
- `docs/operations/2026-04-29-a14-reset.md` (A14 reset operational runbook, post-guard)
- `docs/contracts/runtime-event-log.md` (event-log contract matrix, default-on flipped 2026-04-30 cycle-7 round-2)
