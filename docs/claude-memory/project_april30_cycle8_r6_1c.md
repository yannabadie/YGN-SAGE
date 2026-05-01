---
name: April 30 cycle-8 R6.1c payload schema versioning + 2-round cgpro VERIFY APPROVED
description: Cycle-8 R6.1c shipped 2026-04-30 at commit 49648263. 13 event types schemás avec versioning (v1 + v1_pre_allowlist_reason legacy), manifest drift tripwire byte-exact, validator audit/strict-current modes. cgpro 2-round VERIFY (PUSH BACK narrow round-1 → APPROVED round-2 Option A doc disclosure). Empirical Q6 N=50 audit revalidation found 56 raw-phrase hits in pre-f6711385 oracle_verdict.reason_codes (cycle-7 historical artifact, post-fix prevented).
type: project
originSessionId: dc83c9bb-b729-40fa-aa8c-ca8f426eebc5
---
# Cycle 8 R6.1c — payload schema versioning + all-event allowlists

**Final state**: APPROVED by cgpro round-2 at commit `49648263`. 3 ship commits between cycle-7 closure (`4b8af448`) and cycle-8 R6.1c closure (`49648263`).

## The 3-commit stack

```
49648263 docs(cycle7): cycle-8 R6.1c re-validation disclosure on N=50 evidence
9944674e fix(runtime/event_log,bench): R6.1c VERIFY round-1 fixes #1+#2
78565578 feat(runtime/event_log,bench): cycle-8 R6.1c — payload schema versioning
```

## Methodology trail (DESIGN → IMPLEMENT → VERIFY → SHIP)

1. **cgpro DESIGN locked spec** — 22 KB locked spec at `.tmp/cgpro_cycle8_r6_1c_design_locked_spec.md`. Conv `cgpro_cycle8_r6_1c_design` (UUID `69f317e2-...`) — fresh per-ticket conv in YGN-SAGE project per CLAUDE.md guidance. cgpro answered all 8 design questions (Q1–Q8) with one-line locks + wrote the full schema spec (13 event types exhaustively, versioning format, max-length policy, public/private API, 18 tests, 9 PUSH BACK triggers).
2. **codex IMPLEMENT** — gpt-5.5 xhigh full-auto direct exec. ~3285 LOC across 8 files. 14 manifests committed as drift tripwire.
3. **claude verify-local** — 115 contract+oracle+payload_schema tests pass + 156 broader suite. ruff/mypy clean.
4. **cgpro VERIFY round-1** — PUSH BACK narrow (4 fix items, all docs/spec). Resume `cgpro_cycle8_r6_1c_design`.
5. **Round-1 fixes**: manifest canonicality byte-exact + 4 strict-current rejection tests (committed at `9944674e`).
6. **Empirical Q6 N=50 audit revalidation** — ran `_payload_schema_distribution_for_events` against committed cycle-7 N=50 dir. 0 schema errors, 157 controller_decision events resolved as v1_pre_allowlist_reason legacy_accepted. Distribution table pasted in round-2 prompt.
7. **cgpro VERIFY round-2** — APPROVED with Option A (validation.md doc disclosure only). Refused options B/C/D (would blur schema/raw-leak boundary).
8. **Cycle-8 R6.1c closure** — disclosure committed at `49648263`.

## What's verrouillé

### Schema layer (sage-python)

Single source of truth at `sage-python/src/sage/runtime/event_log/payload_schemas.py` (1355 LOC):

- **PAYLOAD_SCHEMAS registry**: 13 event types × {1 current schema} + 1 legacy `controller_decision.v1_pre_allowlist_reason`
- **CURRENT_PAYLOAD_SCHEMA_VERSIONS** map: explicit per-event-type current version
- **Public exports** (9): `PayloadSchemaVersion`, `PayloadFieldSpec`, `EventPayloadSchema`, `PAYLOAD_SCHEMAS`, `CURRENT_PAYLOAD_SCHEMA_VERSIONS`, `DEFAULT_PAYLOAD_STRING_MAX_BYTES`, `get_schema_for_event`, `get_current_payload_schema_version`, `EventLogSchemaError`
- **Private enforcement** (5): `_validate_payload_against_schema`, `_resolve_payload_schema_version`, `_assert_current_payload_schema_for_emit`, `_payload_schema_distribution_for_events`, `_canonical_json_size_bytes`
- **Schema versioning**: NEW envelope field `payload_schema_version: str` (optional-on-read, REQUIRED on emit), regex `^v[1-9][0-9]*(?:_[a-z0-9][a-z0-9_:-]*)?$`
- **Inference rule**: missing `payload_schema_version` + payload shape match → resolve to first registered schema; for `controller_decision`, `payload.reason` present → `v1_pre_allowlist_reason`

### Manifests (drift tripwire)

14 JSON files at `sage-python/src/sage/runtime/event_log/payload_schema_manifests/`:
- 13 current event-type manifests (`{event_type}.v1.json` for 12 + `controller_decision.v2_allowlist_only.json`)
- 1 legacy `controller_decision.v1_pre_allowlist_reason.json`
- Format: `json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` + NO trailing newline (cgpro round-1 fix)
- Test `test_payload_schema_manifests_match_python_sot` compares `path.read_bytes() == canonical_text.encode("utf-8")` exactly + json.loads(==) belt-and-suspenders

### Writer enforcement (sage-python)

`sage-python/src/sage/runtime/event_log/writer.py`:
- Per-event-type schema validation in `_emit` path
- `payload_schema_version=schema.version` populated on emit
- Schema violations: `EventLogSchemaError` (fail-closed via `SAGE_TRACE_FAIL_CLOSED=1`) OR structured `event_log_error` log + writer disabled (fail-open default)
- **No silent truncation, no extra-field drop, no schema coercion** (except documented existing coercions: reason_code slug, quality_score clamp)

### Validator (sage-python/bench)

`sage-python/src/sage/bench/path_e_validate.py`:
- Imports schema SoT from `sage.runtime.event_log.payload_schemas` (no local duplicate allowlist)
- CLI `--payload-schema-mode {audit,strict-current}` (default `audit`)
- **Audit mode**: accepts any registered version, infers missing, reports distribution, treats non-current as soft warning
- **Strict-current mode**: rejects missing version, rejects explicit non-current, rejects inferred legacy, rejects malformed (4 distinct cases all unit-tested)
- Markdown report has `## Payload schema version distribution` table
- Manifest has `payload_schema_policy` + `payload_schema_version_distribution` blocks

## Empirical Q6 N=50 audit revalidation result (round-2 closure)

```
Total events: 948
Total trace files: 50
Schema errors: 0
Schema warnings: 157

Distribution:
  controller_decision.v1_pre_allowlist_reason: 157 inferred (legacy_accepted)
  failure.v1: 1 inferred (current)
  final_result.v1: 49 inferred (current)
  model_assigned.v1: 166 inferred (current)
  node_completed.v1: 162 inferred (current)
  node_started.v1: 164 inferred (current)
  oracle_verdict.v1: 49 inferred (current)
  routing_decision.v1: 50 inferred (current)
  run_frame_summary.v1: 49 inferred (current)
  task_started.v1: 50 inferred (current)
  topology_selected.v1: 51 inferred (current)
  
  budget and state_applied: NOT observed in this artifact set
```

## The cycle-7 historical leak finding (NEW, round-2 trap)

cycle-8 R6.1c validator's phrase scanner finds **56 raw-phrase hits** in pre-`f6711385` `oracle_verdict.reason_codes[1]` (and mirrored `run_frame_summary.payload.oracle_verdict.reason_codes[1]`). Sample content:
```python
'======================================================================\nERROR: test_case_1 (__main__.TestCases.test_case_1)\nTest '
```

Timeline:
- `162e82ea` (BEFORE fix) = N=50 evidence generated WITH leaks in `oracle_verdict.reason_codes[1]`
- `f6711385` (FIX) = `_exact_oracle()` no longer appends raw `bench_result.reason`; structured tags only; SHA-256 hash → `EvidenceRef.evidence_hash`. Plus added `_RAW_LEAK_PHRASES` scan to validator.
- `f9305d74` (POST-FIX) = N=5 smoke validates 0 leaks across 106 events

The committed cycle-7 N=50 evidence is **frozen pre-fix**. cgpro explicitly refused to:
- ❌ Add `oracle_verdict.v0_pre_f6711385` legacy schema (would blur schema/leak boundary)
- ❌ Downgrade raw-leak findings via audit mode (same)
- ❌ Add schema-blind exemption (same)

cgpro chose **Option A** (document & accept). The `validation.md` got a new "## Cycle-8 R6.1c re-validation disclosure" section verbatim per cgpro's template, including the distribution table + raw-leak re-audit disclosure.

> "This finding belongs in the audit trail, not in the schema system. The boundary between 'schema shape is historically readable' and 'raw output leak is acceptable' is what R6.1c is supposed to make harder to confuse." — cgpro round-2

## Methodology insights from cycle-8 R6.1c

1. **DESIGN-first locked spec is non-negotiable for runtime contracts**: cycle-7 was Claude-led from a runbook (no DESIGN-locked spec) — found 2 round-1 + 4 round-2 sub-blockers. Cycle-8 R6.1c had a 22 KB DESIGN locked spec from cgpro BEFORE codex started — only 4 narrow round-1 fixes (2 closed mechanically, 2 surfaced as round-2 questions). Better outcome with same compute.
2. **Empirical Q6-style verification before closure**: cgpro forced "run the audit-mode validator against committed evidence" as round-1 closure work. Without it, we would have shipped without knowing the 56 oracle_verdict.reason_codes findings existed. Pattern: when a fix changes how validation runs, RE-RUN validation against the historical artifacts the fix is supposed to interpret.
3. **Don't preventively add audit exemption surfaces unless the data demands it**: I almost implemented the controller_decision.reason audit exemption preventively per cgpro round-1 advice. Empirical Q6 showed N=50 has no `controller_decision.reason` >64 chars (max 21). cgpro then said "defer". Lesson: empirical-data-driven decisions over preventive code surface.
4. **Schema versioning > one-off compat flag** (locked methodology): cgpro deferred `--allow-legacy-controller-reason` from cycle-7 to cycle-8 R6.1c on the principle that one-off compat flags solve today's convenience but don't generalize. Schema versioning IS the architectural answer. R6.1c v1_pre_allowlist_reason / v2_allowlist_only is the proof.
5. **Boundary preservation is a design value**: cgpro round-2 refusal to add `oracle_verdict.v0_pre_f6711385` was specifically about preserving the boundary "schema shape historically readable ≠ raw output leak acceptable". When in doubt about whether to extend the schema system to cover a historical-data-quality issue, the answer is usually NO — disclose in audit trail instead.
6. **Manifest canonicality must be byte-exact, not just semantic**: cgpro round-1 caught that `json.loads(==)` test would miss CRLF/LF drift, indentation drift, key reorder. Fix: `path.read_bytes() == canonical_text.encode("utf-8")`. Lesson: when committing serialization artifacts as tripwire, lock byte-level canonical form, not just structural equivalence.

## Cycle-9 ordering (cgpro recommended, locked at round-3 cycle-7)

1. **A14 epoch fail-closed guard** — `~/.sage/posterior_epoch.json.epoch != 1` AND state files exist → raise. **Note**: `boot_topology.py` is sage-python, but bandit posteriors are stored in **Rust sage-core** SQLite. Fail-closed should be defense-in-depth at BOTH Rust state load AND Python boot. User reminder 2026-04-30: "N'oublies pas que YGN-SAGE ce n'est pas que sage-python" — applies here.
2. A22 follow-ups (verifier reason codes bucket-analysis, off-mode regression, deletion-side `/dev/null`)
3. T2 phase 2/3 (memory write paths beyond per-node AgentLoop wiring)
4. Planner producer live integration (LAST per cgpro — adds evidence flow, must wait for tight schema gates)

## Open follow-ups from R6.1c (deferred to cycle-9+)

- **N=50 evidence file generation regenerator**: when path E re-runs happen, the `path_e_validate.py` already takes `--cycle-tag` / `--oracle-mode` (cycle-7 round-2 fix). Future N=X regen with R6.1c writer will produce `payload_schema_version=v2_allowlist_only` traces — explicit versions in the distribution column.
- **Non-blocking polish from cgpro round-3 cycle-7**: validation.md criteria table row 3 still says "on every run (49/50)" — cgpro suggested "on every verdict-emitting run" for cycle-9 cosmetic.
- **Real legacy long content scenario**: if/when a future trace shows long `controller_decision.payload.reason` content, apply scoped audit-only exception per cgpro's round-1 preferred fix.
