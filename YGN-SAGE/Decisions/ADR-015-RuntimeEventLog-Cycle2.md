---
title: ADR-015 RuntimeEventLog v0 (Cycle 2, R5)
type: adr
status: shipped
date: 2026-04-28
commits: ["86fded60", "4742295e"]
tags: [runtime, observability, jsonl, event-log]
---

# ADR-015 — RuntimeEventLog v0 (Cycle 2, R5 + R5.1)

## Context

After R2 unified `_run_core` (Cycle 1), the runner's typed event stream became the natural injection point for an internal execution-contract trace. Two existing observability surfaces existed but didn't fit the need:

- **OTel GenAI spans (B1, shipped 2026-04-25)** — distributed external observability via OTLP/console/logfire exporters. Spans use `_safe_str()` redaction with 4096-byte truncation gated on `SAGE_OTEL_RAW_PAYLOADS`. Wrong semantics for an internal-contract ledger (truncation breaks hash inputs ; env-gated redaction is OTel-specific).
- **AgentEvent / EventBus** — UI-bound dashboard events. Different audience, different lifecycle.

cgpro 2026-04-28 cycle-1-reassess locked: framing as **RuntimeEventLog v0** — internal execution contract ledger, NOT OTel extension. Behind `SAGE_TRACE_JSONL_DIR=<path>` opt-in.

## Decision

NEW module `sage/runtime/event_log/` with 6 files (708 LOC):

- **schema.py**: `SCHEMA_VERSION = "1.0"`, `EVENT_TYPES` tuple (10 types initially: TaskStarted / RoutingDecision / TopologySelected / ModelAssigned / NodeStarted / NodeCompleted / ControllerDecision / Failure / Budget / FinalResult ; later extended to 13 with StateApplied [R6], OracleVerdict [R9], RunFrameSummary [R7]), `REDACTION_STATES`, `SOURCE_COMPONENTS`.
- **events.py**: 10+ private frozen dataclasses with `_EventCore` base + R6-prep no-op fields (`edge_type`, `channel`, `state_version`).
- **redaction.py**: forced `RedactionFilter(enabled=True)` from `sage.security.redaction` (NOT `_safe_str` — different semantics). Full 64-char SHA-256 hashes (not truncated). Canonical JSON envelope `{schema_version, event_type, payload}` for hashing.
- **writer.py**: per-run JSONL files at `<trace_dir>/<run_id>.jsonl`. Exclusive create (FileExistsError → fail). `fsync` only on FinalResult by default. **Disable on first failure** preserves contiguous JSONL prefix (cgpro DESIGN insight: retry/buffering with gaps would violate the monotonic seq invariant). `SAGE_TRACE_FAIL_CLOSED=1` escalates to `EventLogUnavailable` exception.
- **errors.py**: `EventLogUnavailable(RuntimeError)`.
- **__init__.py**: public API surface.

Pipeline integration: `run_id` generated as canonical 26-char Crockford-Base32 ULID at task entry (uses `ulid` lib if installed, else in-repo monotonic generator with same canonical contract). Pipeline emits TaskStarted / RoutingDecision / TopologySelected / ModelAssigned / FinalResult inside try/finally ensuring FinalResult fires exactly once. Runner emits NodeStarted / NodeCompleted / ControllerDecision / Failure / Budget per node, in **executor ready order**.

## Key invariants

- **Default OFF**: writer disabled when `SAGE_TRACE_JSONL_DIR` unset. Zero overhead in legacy mode.
- **Per-run isolation**: each run gets its own JSONL file, no cross-run interleaving. Concurrent `pipeline.run()` calls in `asyncio.gather` work safely.
- **`seq` strictly monotonic**: 0-indexed, contiguous in healthy files. Used as the canonical pointer for `parent_event_id` (every event except TaskStarted has a parent).
- **`task_hash` = sha256(task_text)[:64]** ; `payload_hash` = sha256(canonical_envelope) — NEVER truncated to 16 chars. Storage is cheap ; audit value matters.
- **Default redaction**: raw task text, raw node output, raw tool args/results NEVER on disk. Only hashes + safe metadata. `SAGE_TRACE_RAW=1` opt-in stores payloads but still strips credentials via `RedactionFilter`.
- **Sink failure NEVER changes pipeline result** unless `SAGE_TRACE_FAIL_CLOSED=1`.

## R5.1 — Edge-binding contract follow-up (`4742295e`)

Pre-R6 prep test pinning `graph.get_edges()` as Python-visible canonical edge-typing API. cgpro CGPRO.md R6 design suggested calling `graph.edges_of_type(EdgeType.State)` for channel partitioning, but verification confirmed `edges_of_type()` is Rust-only (not exposed via PyO3). The Python-visible API is `get_edges()` returning `[(src, dst, edge_type_str), ...]` with edge types as strings. 4 contract tests prevent future drift toward the Rust-only helper.

## Consequences

- 16 acceptance tests pinning event taxonomy + 1 cgpro VERIFY round-trip regression (`test_run_id_is_canonical_ulid_even_when_ulid_dependency_missing` — schema contract held even when soft import fails).
- Sets the typed-event substrate for R6 StateCore (state_applied event), R7 RunFrame (run_frame_summary trailing diagnostic), R9 OracleStack (oracle_verdict gate).
- Bumps event count from 10 (R5) → 11 (R6 +state_applied) → 12 (R7 +run_frame_summary) → 13 (R9 +oracle_verdict).
- mypy 0/191 (was 185, +6 for new module).
- 1771 LOC added across 9 files in R5 + 99 LOC in R5.1.

## Related

- [[ADR-014-RuntimeContracts-Cycle1]] — R2's `_run_core` is the natural event injection point
- [[ADR-016-StateCore-Cycle3]] — R6 adds state_applied event; uses R5 sink
- [[ADR-017-RunFrame-Cycle4]] — R7 adds run_frame_summary trailing event
- [[ADR-018-OracleStack-Cycle5]] — R9 adds oracle_verdict event
- `docs/contracts/runtime-event-log.md` — canonical event contract doc (mode-aware matrix, 13-event catalog, schema versioning policy, golden fixtures)
- `tests/golden/runtime_events/` — golden JSON fixtures
