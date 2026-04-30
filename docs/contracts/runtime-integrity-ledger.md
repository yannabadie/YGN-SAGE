# Runtime Integrity Ledger

**Status**: documentary contract (no code refactor in cycle 8 / 9). Per cgpro 2026-04-30 architect review Q-A: *"créer un registre documentaire/testable, pas un gros move"*.

After cycle-7 + cycle-8 R6.1c + cycle-8 A14, YGN-SAGE has accreted 5 invariant-binding mechanisms across `sage-python/src/sage/runtime/`, `sage-python/src/sage/`, and `sage-core/src/topology/`. They are **conceptually a Runtime Integrity subsystem** but **physically distributed** to keep coupling local. This ledger is the cross-reference contract.

## The 5 invariants

Pattern that emerged from 3 cycles of "declared ≠ verified" traps (cycle-7 contract drift, cycle-8 R6.1c raw-leak vs audit policy drift, cycle-8 A14 epoch ≠ provenance): **any label that authorizes a side-effect or learning decision must be bound to verified content, schema, provenance, or executable proof.**

| Invariant | Declared label | Verified content | Side-effect blocked if invalid |
|---|---|---|---|
| **Event payload schema** | `event_type` + `payload_schema_version` (envelope) | allowlist `field_specs` + canonical fixture + max_utf8_bytes | event emission (writer raises `EventLogSchemaError`); validator audit acceptance |
| **Oracle evidence** | `OracleVerdict.trainable` (True/False) | structured `EvidenceRef.evidence_hash` SHA-256 + producer schema (`payload_schema_version` per producer) | bandit / MAP-Elites / online-evolution / training-memory updates |
| **Posterior epoch** | `~/.sage/posterior_epoch.json.epoch` (integer) | `topology_state_manifest.json.state_files[].sha256` over A14 state file bytes | `TopologyEngine::load_state` / `save_state` (Rust + Python preflight) |
| **Contaminated backup** | `_CONTAMINATED.json.contaminated=true` (operator-readable poison-pill) | `audit_dump_sha256` cross-reference to immutable audit MANIFEST.json | normal load (any contaminated marker present in active state dir = fail-closed) |
| **RunFrame summary** | `run_frame_summary.payload.parent_event_id` | `final_result.seq` consistency (parent_event_id == final_result.seq) | diagnostic trust (downstream `path_e_validate` event-order check) |

## Module cross-reference

| Invariant | Primary module (Python) | Primary module (Rust) | Tests |
|---|---|---|---|
| Event payload schema | `sage/runtime/event_log/payload_schemas.py` | n/a (Python-emitted, Rust-consumed via PyO3 if any) | `tests/test_payload_schemas.py` (18) + `tests/test_runtime_event_contracts.py` (forced payload contract) |
| Oracle evidence | `sage/runtime/oracle/_oracles.py`, `sage/runtime/evidence/producers/*.py` | n/a | `tests/test_oracle_*.py`, evidence producer round-trip JSON pairs |
| Posterior epoch | `sage/posterior_epoch.py` | `sage-core/src/topology/posterior_epoch.rs` | `tests/test_posterior_epoch.py` (Python), Rust unit tests in `posterior_epoch.rs` |
| Contaminated backup | `sage/ops/a14_reset.py` | n/a (Python ops surface) | `tests/test_a14_reset.py` |
| RunFrame summary | `sage/runtime/run_frame/__init__.py` | n/a | `tests/test_run_frame.py` |

## Boundary against accidental coupling

These 5 invariants are **conceptually a family** but **physically deliberately not consolidated** under a single `sage/runtime/integrity/` umbrella. Reason (cgpro Q-A verdict 2026-04-30):

> "payload_schemas.py est naturellement couplé à runtime/event_log, tandis que posterior_epoch est naturellement couplé à topology et aux fichiers bandit_state.db, archive_state.db, engine_extras.json. Un refactor physique maintenant créerait surtout churn/import risk sans benchmark gain."

Phase 2 / v0.2 may add re-export aliases (`sage.runtime.integrity.epoch`, `sage.runtime.integrity.schemas`) without moving the actual files. **Do not relocate modules in cycle 9.**

## Adversarial threats this ledger defends against

The 3 traps cgpro found at cycle-7/8 round-1 VERIFY rounds, all in the "declared ≠ verified" class:

1. **Cycle-7**: `SAGE_ORACLE` declared default-on in code, but `runtime-event-log.md` contract docs still said "ONLY emitted when SAGE_ORACLE=1" (closed at `f3a89631` via stale-phrase lint test).
2. **Cycle-8 R6.1c round-1**: `controller_decision.payload.reason` declared "safe" (forced under default-on), but redaction layer was credential-only — no allowlist, no PII ban. Audit mode accepts legacy `reason` while raw-leak scanner hard-rejects it (closed at `9944674e + 49648263` via allowlist + Option A doc disclosure).
3. **Cycle-8 A14 round-1**: `posterior_epoch.json.epoch=1` declared "fresh epoch", but no binding to the actual DB bytes. Operator copy-restoring `bandit_state.db` from contaminated backup left the epoch label valid while the content was poisoned (closed at `f9521616` via `topology_state_manifest.json` SHA-256 binding).

**Cycle-9+ design principle**: any new "label authorizes side-effect" code path MUST register here with all 4 columns filled BEFORE the side-effect ships. This is the architectural pattern (cgpro 2026-04-30):

> "tout label qui autorise un side-effect ou une décision d'apprentissage doit être lié à un contenu vérifié, un schéma, une provenance, ou une preuve exécutable."

## Maintenance discipline

- When adding a new invariant: append a row to **both** tables (the 5-invariant table AND the module cross-reference). Wire a regression test that proves the side-effect is blocked when the verification fails.
- When changing an existing invariant's verified-content schema: bump the schema version (per `payload_schemas.py` discipline, e.g. `v1 → v2_X`) and provide migration/inference rules. Old traces remain readable in audit mode, new emissions strict-current.
- This ledger is referenced from `CLAUDE.md` directive #8 (A14 guard) and should be referenced from any future directive adding a new invariant.

## References

- cgpro 2026-04-30 architect review (saved at `.tmp/cgpro_architect_review_finaltext.md`, conv `cgpro_architect_review`)
- ADR-018 / ADR-019 (Runtime cycle 5/6 design)
- `docs/operations/2026-04-29-a14-reset.md` (A14 reset operational runbook, post-guard)
- `docs/contracts/runtime-event-log.md` (event-log contract matrix, default-on flipped 2026-04-30 cycle-7 round-2)
