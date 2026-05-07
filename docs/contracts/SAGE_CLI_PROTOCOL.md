# SAGE CLI Protocol — `sage run --jsonl` v0

**Status**: Implemented (Python backend gaps closed). Proposed 2026-05-05 in
cycle-12 prelude (`cgpro_pi_mono_pivot_20260505` Option 1 verdict). Implementation
shipped 2026-05-05 (`d09bed4d` `sage run --jsonl` backend) and the four NYI
v0 protocol gaps closed in cycle-13 K post-Phase-2.2 cli_gaps stage chain
2026-05-07 (Stages A-D under `cgpro_cli_protocol_gaps_20260507`). The
TypeScript `clients/pi-ygn-sage/` adapter is tracked separately and has not
yet shipped. Architecture: pi-mono front-end + YGN-SAGE backend
communicating via subprocess + JSONL/RPC.

---

## Context

YGN-SAGE has no end-user CLI surface today. The pivot agreed in cycle-11 (plan at
`C:\Users\yann.abadie\.claude\plans\abstract-finding-pixel.md`) makes `sage run
--jsonl` the **canonical machine-readable backend** that any TypeScript /
TUI / IDE front-end can drive — with `pi-mono` (`badlogic/pi-mono`) as the
reference shell. The protocol is **versioned, fail-closed on mismatch, and
binds each event to the runtime-integrity invariants** so the CLI cannot
become a side-channel that bypasses cycle-7+ guarantees.

Goals:
1. **Reuse** the 14 event types already shipped in `RuntimeEventLog` v0
   (`sage-python/src/sage/runtime/event_log/schema.py:6-21`) — the CLI is a
   *consumer* of the runtime contract, not a parallel one.
2. **Add only 4 CLI-shell envelope events** (`cli_started`, `cli_progress`,
   `cli_tool_request`, `cli_complete`) — these are protocol-layer, NOT
   runtime-layer; they MUST NOT appear in `RuntimeEventLog` v0's
   `EVENT_TYPES` taxonomy.
3. **Sovereignty stays in the backend**. The frontend MUST NOT decide model,
   topology, or learning gate. It MAY veto (cancel, deny tool call) and MAY
   tighten budget. It MAY NOT loosen budget mid-run, change the model
   selection, or override the oracle gate.
4. **Strict JSONL discipline**: LF-only delimiters (NOT Node `readline`-
   compatible — pi-mono's RPC spec already documents this). UTF-8 only.
   `\r` is permitted inside JSON strings but never as a record separator.

Non-goals:
- Web sockets / SSE (subprocess + stdio is enough; web search 2026: CLI is
  4–32× cheaper in tokens than MCP and 100% vs 72% reliability).
- Bidirectional RPC with futures / IDs *as a hard requirement* — the
  protocol is event-stream + occasional commands. Tool-approval is the only
  request/response sub-flow and uses `correlation_id`.

---

## Frame envelope

Every JSONL frame on stdout (backend → frontend) has the same envelope:

```json
{
  "protocol_version": "v0",
  "event_type": "<one of 18>",
  "seq": <monotonic int per run, starts at 0>,
  "run_id": "<26-char ULID, stable for the entire run>",
  "ts_ms": <unix-ms, wall-clock>,
  "payload_schema_version": "v1" | "v1_X",
  "payload": { ... event-specific fields ... }
}
```

`protocol_version` is the **CLI envelope version** (this document) — distinct
from `payload_schema_version` which is the per-event payload schema (existing
`payload_schemas.py` v0 envelope from cycle-7 R6.1c, commit `78565578`). A
mismatch in `protocol_version` is a hard fail-close on the frontend (do NOT
attempt graceful degradation; the bytes after a version mismatch could mean
anything).

Inbound frames on stdin (frontend → backend) use a smaller envelope:

```json
{ "command": "<one of 5>", "id": "<optional client-supplied correlation>", "args": { ... } }
```

`id` is required only for `approve_tool_call` / `deny_tool_call` (must echo
the `correlation_id` from the matching `cli_tool_request` event).

---

## Outbound events (18 types)

### Inherited from `RuntimeEventLog` v0 (14 types)

These pass through unchanged from the existing taxonomy at
`sage-python/src/sage/runtime/event_log/schema.py:6-21`. Each event's payload
schema is already canonical in
`sage-python/src/sage/runtime/event_log/payload_schemas.py`. The CLI emits
them via the same writer, multiplexed onto stdout.

| `event_type` | Source component | When emitted | Invariant |
|---|---|---|---|
| `task_started` | pipeline | First frame after the run-frame builder is set up | 1 (event payload schema) |
| `routing_decision` | pipeline | Stage 0 classify completion | 1 |
| `bandit_attribution_mismatch` | pipeline | Stage 5 when (model_id, template) executed != pending | 6 (bandit attribution) |
| `topology_selected` | pipeline | Stage 2 select_topology completion | 8 (control-surface) |
| `model_assigned` | pipeline / model_assigner | Stage 3 per-node assignment | 1 |
| `node_started` | topology_runner | Each node enters execution | 1 |
| `node_completed` | topology_runner | Each node exits (success / fail) | 1 |
| `controller_decision` | controller | Each runtime adaptation choice | 1 |
| `state_applied` | topology_runner | Reroute / prune / upgrade applied | 1 |
| `failure` | pipeline / runner | Any unrecoverable error | 1 |
| `budget` | cost_tracker | Per-node cost record | 1 |
| `final_result` | pipeline | Stage 5 entry, before learn | 1, 5 (RunFrame summary) |
| `oracle_verdict` | pipeline | Oracle gate firing (cycle-7+) | 2 (oracle evidence) |
| `run_frame_summary` | pipeline | After learn, terminal frame for the run | 5 |

Inherited runtime events preserve their payload schema and forensic
archive bytes (the `RuntimeEventLog` file kept under `trace_dir`). When
mirrored to stdout, the CLI driver rewrites only the envelope `seq`
field into the per-run unified CLI stdout sequence domain so every
stdout frame — inherited runtime AND CLI-shell envelope — sits on
ONE monotonic counter. The `RuntimeEventLog` archive keeps its own
internal sequence numbers unchanged for forensic reproducibility.
Frontends consume the stdout stream and reconcile via `final_seq`
(invariant 5); forensic consumers read `trace_dir` directly.

### CLI-shell envelope events (4 NEW types)

These are emitted by the CLI driver, NOT through `RuntimeEventLog`. They are
**protocol-layer**, not runtime-layer. They do NOT update bandit / MAP-Elites
/ training memory and do NOT appear in `EVENT_TYPES`. Their payload schemas
are versioned independently of `payload_schemas.py`.

| `event_type` | When emitted | Payload (v1) | Invariant |
|---|---|---|---|
| `cli_started` | Frame 0 (always first) | `{ "protocol_version", "sage_version", "sage_commit_sha", "task", "budget_usd", "system_hint", "tier" }` | 9 (CLI protocol versioning, NEW) |
| `cli_progress` | Idle heartbeat: emitted every 5s when no other stdout frame has fired for ≥10s. The CLI driver runs a timer-based heartbeat task (NOT event piggyback) so the frontend has liveness during long idle waits (e.g. Stage 4 model thinking). | `{ "stage", "elapsed_ms", "human_readable" }` where `stage` is one of `boot` / `classify` / `decompose` / `select_topology` / `assign_models` / `execute` / `learn`, `elapsed_ms` is monotonic ms since `cli_started`, `human_readable` is a free-form display hint. | None (UX-only) |
| `cli_tool_request` | `TopologyRunner.approval_callback` fires | `{ "correlation_id", "tool_name", "tool_args_redacted", "node_id", "model_id" }` | 9 |
| `cli_complete` | Last frame, terminal | `{ "exit_code", "total_cost_usd", "total_latency_ms", "outcome": "success" \| "failure" \| "cancelled", "final_seq" }` | 9, 5 |

The 4 CLI events live behind `protocol_version="v0"`. Bumping any of them
requires a new `protocol_version` (v1, etc) and frontends MUST validate the
version on `cli_started`.

---

## Inbound commands (5 verbs)

| Command | Args | Effect | Notes |
|---|---|---|---|
| `prompt` | `{ "task": "<text>", "budget_usd"?: <float>, "system_hint"?: 1\|2\|3 }` | Begins the run | MUST be the first command. Subsequent `prompt` is REJECTED — one run per process. (Future: multi-turn via a separate `chat` subcommand.) |
| `approve_tool_call` | `{ "id": "<correlation_id from cli_tool_request>" }` | Releases the awaited approval | If `id` doesn't match a pending request, ignored (idempotent). |
| `deny_tool_call` | `{ "id": "<correlation_id>", "reason"?: "<text>" }` | Releases the wait, signals deny | The runner treats the tool as unavailable for this call (does NOT abort the run). |
| `cancel` | `{ "reason"?: "<text>" }` | Cooperative abort | Requests cooperative cancellation of the Python pipeline task via `asyncio.Task.cancel()`. Emits exactly one `failure(kind="cli_cancel", error_type="cancelled", message=<reason or "cancel requested">)` followed by terminal `cli_complete(outcome="cancelled", exit_code=130)`. Idempotent at the stream level: a second `cancel` does NOT emit a second failure frame. See "Known v0 limitation — cooperative cancellation" below for the cooperative-at-await semantics. |
| `set_budget` | `{ "budget_usd": <float> }` | Mid-run budget change | **TIGHTEN ONLY** — `budget_usd` is the new POSITIVE FINITE REMAINING budget cap (NOT absolute total). Accepted updates rebase `cost_tracker.budget_usd` to `total_spent + budget_usd` and emit a `budget(kind="budget_tightened", ...)` event. Rejection emits a non-terminal `failure(kind="cli_command", error_type=...)` event with one of the reason codes below; the run does NOT abort. Reason codes: `budget_before_prompt` (no active run yet), `budget_invalid_value` (zero / negative / NaN / inf / non-numeric `budget_usd` — zero is rejected because `budget_usd <= 0` is `CostTracker`'s unlimited sentinel), `budget_loosen_rejected` (new remaining > current `cost_tracker.remaining`). |

The frontend MUST NOT send any other command. Unknown commands emit a
`failure(reason="unknown_command", details=<command>)` event but do NOT abort
the run.

---

## Invariant binding (the 9 rules the CLI MUST preserve)

This is the load-bearing table. The CLI is a NEW exit point for the runtime
contract — it MUST NOT become a side-channel. Each invariant from the
ledger is reproduced here with the CLI-specific binding.

| # | Invariant | CLI binding |
|---|---|---|
| 1 | **Event payload schema** | The 14 inherited events use the same `payload_schemas.py` envelope. Frame writes go through the same `_assert_current_payload_schema_for_emit` validator. The 4 CLI-shell events use an INDEPENDENT v1 schema (cli_payload_schemas.py) — ledger invariant 1 still holds because schema mismatches fail-close at write time. |
| 2 | **Oracle evidence** | The CLI MUST NOT update bandit / MAP-Elites / online-evolution / training-memory based on its own state. All those updates remain gated by `OracleVerdict.trainable` inside `sage.pipeline_v2.learn.learn` (the canonical Stage 5 module function — Phase 2.2 retired the `_stage_learn` private method). The CLI is a CONSUMER of `oracle_verdict` events (it can show "✓ Trainable" / "✗ Abstain" in the TUI), never a producer. |
| 3 | **Posterior epoch** | CLI is read-only on bandit_state.db / archive_state.db / posterior_epoch.json. It does NOT call `engine.save_state` directly. The pipeline's existing periodic flush (post `213183c1` epoch preflight) and the atexit handler are the only writers. |
| 4 | **Contaminated backup** | CLI MUST refuse to start when `~/.sage/_CONTAMINATED.json` is present. It surfaces the operator-readable poison-pill in `cli_started`'s `tier` field with `tier="contaminated_refuse"`, then exits 78 (EX_CONFIG). |
| 5 | **RunFrame summary** | `cli_complete` is always the terminal frame on stdout. `cli_complete.payload.final_seq` MUST equal the stdout `seq` of the frame IMMEDIATELY PRECEDING `cli_complete`. On the normal success path that frame is the stdout-mirrored `run_frame_summary`; on cancel / failure / mid-tool-call paths it may be any other stdout frame including a CLI-shell event such as `cli_tool_request`. Frontends use this as a stream reconciliation gate — any frame with `seq > final_seq` after `cli_complete` is a stream-level violation. |
| 6 | **Bandit attribution** | The CLI MUST NOT short-circuit Stage 0 routing or Stage 5 learn. The Rust `SystemRouter.route_integrated()` is the only authoritative source of `bandit_decision_id`. Mismatches surface as `bandit_attribution_mismatch` events that the frontend SHOULD display. |
| 7 | **Timeout enforcement** | Per-task `budget_usd` is wall-clock-bounded via the existing `CostTracker.is_over_budget`. The CLI's `set_budget` command can TIGHTEN the budget (security: attacker who got `set_budget` access can NOT exfiltrate by extending). |
| 8 | **Control-surface completeness** | When `node_count > 0`, frames `topology_selected` + `model_assigned` MUST appear before `node_started`. Frontend SHOULD validate this ordering and refuse to render a topology card without both. |
| 9 | **CLI protocol versioning (NEW)** | `cli_started` MUST be the first frame. `protocol_version` MUST be `"v0"` (bumped per protocol change). Frontends fail-close on mismatch. Backend version must come from `sage.__version__`, not be a hardcoded literal. |

Invariant 9 is recorded in `docs/contracts/runtime-integrity-ledger.md` as
the 9th invariant (backported in cycle-12 commit `f647c5ae`, per the ledger
maintenance discipline: "When adding a new invariant: append a row to both
tables AND wire a regression test that proves the side-effect is blocked
when the verification fails."). The regression test
`tests/test_runtime_event_contracts.py::test_cli_protocol_version_is_locked`
asserts `CLI_PROTOCOL_VERSION == "v0"` AND every emitted frame carries that
string verbatim.

---

## Boundary against pi-mono (cgpro trap mitigations)

The CLI protocol is **adapter-versioned**, not pi-mono-versioned. The npm
package `clients/pi-ygn-sage/` (cycle-13) pins `pi-mono = exact-X.Y.Z` AND
declares which `protocol_version` of THIS doc it speaks (`"v0"` initially).
A new pi-mono major release does NOT require a backend release unless the
adapter chooses to bump.

What pi-mono does NOT see / decide:
- **Which model to use** — that's `bandit.select_with_context_for_template`
  via Stage 0. The frontend reads `model_assigned` and may DISPLAY a
  "model: claude-sonnet-4-6" pill, but cannot override it.
- **Topology shape** — that's `sage.pipeline_v2.select_topology.select_topology`
  via DAG features (Phase 2.2 retired the `_stage_select_topology` private
  method; module-function patching is the permanent test seam). The frontend
  reads `topology_selected` and may render a graph, but cannot reroute.
- **Tool list** — the runner's `tool_registry` is fixed at boot. pi-mono's
  4 default tools (read/write/edit/bash) are NOT injected; YGN-SAGE's typed
  repo tools (`read_file` / `search_repo` / `list_files` / `run_tests` /
  `apply_patch` / `git_diff`) own that surface.
- **Learning gate** — `oracle.trainable` decides whether the run feeds
  back into bandit / MAP-Elites. The frontend reads `oracle_verdict` and
  MAY show a "✓ Trainable" badge, but cannot force a `trainable=True` if
  the backend abstains.

What pi-mono DOES own:
- Terminal rendering (TUI, markdown, autocomplete).
- Provider credentials configuration (env vars / config file). The
  backend READS the env, the frontend WRITES it before subprocess spawn.
- Session management (multi-run history, replay, export). The runtime
  decision trace is in the JSONL stream the frontend captured; YGN's
  own `~/.sage/event_log/` is the canonical archive but the frontend
  may keep its own user-facing transcript.
- Tool-approval UX (the YES/NO prompt rendering).

---

## Inbound rate / liveness

- The backend MUST emit a `cli_progress` heartbeat every **5 seconds** when no
  other stdout frame has fired in the last **10 seconds**. The heartbeat is
  timer-based (NOT event-piggyback) so liveness is preserved during long
  idle waits (e.g. 30s of Stage 4 model thinking). A fast S1 bypass that
  completes in 200ms emits no `cli_progress`.
- The heartbeat starts immediately after `cli_started` and reports
  `stage="boot"` until the orchestrator updates the label before each
  high-level pipeline stage. The heartbeat task is cancelled in the
  CLI driver's `finally` block before `cli_complete`, so no
  `cli_progress` frame ever appears after the terminal frame.
- `cli_progress` itself does NOT reset the idle clock. The driver tracks
  two timestamps (`last_non_progress_frame_at` and
  `last_progress_frame_at`) and emits when both
  `now - last_non_progress_frame_at >= 10s` AND
  `now - last_progress_frame_at >= 5s`. Resetting on the heartbeat
  itself would degrade the cadence to 10s during long idle periods.
- Frontends SHOULD treat absence of any frame for **60 seconds** as a probable
  hang and offer the user a `cancel` action.
- The backend MUST flush `stdout` after every frame. (The `_SinkHandle.flush`
  already does this on file writes; the stdout mirror must add it.)

---

## Streaming guarantees and ordering

- Frames arrive monotonically increasing in `seq`. There are no gaps.
- For each `correlation_id` issued in `cli_tool_request`, exactly one
  `approve_tool_call` OR `deny_tool_call` MUST be received before the next
  `node_started` for that node.
- `cli_started` is the first frame; `cli_complete` is the last frame. No
  `RuntimeEventLog` events appear before `cli_started` or after `cli_complete`.
- `final_result` and `oracle_verdict` are emitted within Stage 5; their
  ordering relative to each other is `final_result` → `oracle_verdict` →
  `run_frame_summary` (per cycle-7 R6.1c locked event order).
- Runtime `failure` frames use the cycle-7 R6.1c redacted FLAT shape on
  stdout: `kind` / `error_type` / `node_id` are TOP-LEVEL event fields
  (NOT nested under `payload`). The `message` is hashed into
  `payload_hash` for forensic redaction; the full payload (including
  `message`) is preserved in the forensic file under `trace_dir`. CLI
  envelope events (`cli_started` / `cli_progress` / `cli_tool_request` /
  `cli_complete`) keep the nested `payload` shape and are written
  directly by the CLI driver, not through `RuntimeEventLog`.
- A `failure` frame whose top-level `kind == "cli_command"` is ALWAYS
  non-terminal: the backend uses these to surface command rejections
  (e.g. `set_budget` reason codes `budget_before_prompt` /
  `budget_invalid_value` / `budget_loosen_rejected`) without aborting
  the run. The CLI MUST NOT add a `recoverable` field to the failure
  schema; the runtime carries only `kind`, `error_type`, and `message`
  (cycle-13 K Stage B lock 2026-05-07).
- For `failure` frames with other top-level `kind` values (e.g.
  node-level failures, multi-agent error fallback, FrugalGPT cascade),
  the frontend SHOULD treat them as terminal-leaning: the run continues
  only if a subsequent non-terminal frame appears before `cli_complete`.
  Terminality is inferred from the surrounding event stream and the
  `cli_complete.payload.outcome` field (`success` / `failure` /
  `cancelled`).
- A `failure` frame whose top-level `kind == "cli_cancel"` and
  `error_type == "cancelled"` is the ALWAYS-EMITTED penultimate
  frame on the cancel path. Exactly one such frame appears immediately
  before `cli_complete(outcome="cancelled", exit_code=130)`, regardless
  of how many `cancel` commands the frontend sent (idempotent stream
  contract). The frame is mirrored through the unified stdout sequence
  domain, so `cli_complete.payload.final_seq` equals the cancel
  failure's stdout `seq`.

---

## Known v0 limitation — cooperative cancellation

Cancellation is cooperative at Python `await` boundaries. When the
frontend sends `cancel`, the backend requests cancellation of the
pipeline task with `asyncio.Task.cancel()`. Python raises
`asyncio.CancelledError` into the task at the next opportunity.
In-flight provider HTTP calls, blocking tool calls, and Rust
executor work do NOT support fine-grained interruption in v0.
A frontend SHOULD display "cancellation requested" until it
receives `cli_complete.payload.outcome == "cancelled"`. Deeper
runtime cancellation (signal-based interrupt of provider calls,
Rust `TopologyExecutor` work) is out of scope for the cycle-13 K
post-Phase-2.2 cli_gaps stage chain and is tracked as a
follow-up cycle initiative.

---

## Verification

Snapshot tests in `sage-python/tests/test_sage_cli_jsonl.py` (5 tests, golden
JSONL files at `sage-python/tests/golden/cli_jsonl/`):

1. **S1 bypass golden** — single-agent task, expected sequence:
   `cli_started, task_started, routing_decision, final_result, oracle_verdict,
   run_frame_summary, cli_complete`.
2. **S2 multi-agent golden** — adds `topology_selected, model_assigned,
   node_started ×3, node_completed ×3, ...`.
3. **Tool approval round-trip** — `cli_tool_request` →
   `approve_tool_call` (then `deny_tool_call`) → check both branches.
4. **Cancel mid-run** — `cancel` command produces a runtime `failure`
   frame whose top-level `kind == "cli_cancel"` and `error_type == "cancelled"`
   (cycle-7 R6.1c FLAT redacted shape on stdout, NOT nested under `payload`),
   then terminal `cli_complete(outcome="cancelled", exit_code=130)`.
   Idempotent at stream level: a second `cancel` does NOT emit a second
   failure frame.
5. **Two-run determinism** — same input → same JSONL bytes excluding
   `run_id`, `ts_ms`, and ULID fields. Mirrors P9 phase 1 test #1
   (`test_pipeline_v2_run_byte_identical.py`).

---

## References

- `docs/contracts/runtime-integrity-ledger.md` — the 8 invariants this protocol extends.
- `docs/contracts/rust-python-boundary.md` — the ownership matrix.
- `sage-python/src/sage/runtime/event_log/` — the runtime event taxonomy.
- `sage-python/src/sage/runtime/event_log/payload_schemas.py` — per-event payload schemas (cycle-7 R6.1c versioning, commit `78565578`).
- `sage-python/src/sage/cli.py` — root CLI dispatcher.
- `sage-python/src/sage/topology/runner.py:249, 1709-1718` — `approval_callback` hook reused for `cli_tool_request`.
- `sage-python/src/sage/contracts/cost_tracker.py:13-76` — `CostTracker` referenced by `set_budget`.
- pi-mono RPC spec: github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/rpc.md
- pi-mono extensions docs: github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/extensions.md
- cgpro consultation `cgpro_pi_mono_pivot_20260505` (Option 1 verdict, 8 traps).
- Cycle-12 prelude plan: `C:\Users\yann.abadie\.claude\plans\abstract-finding-pixel.md`.

---

## Status changes

- 2026-05-05: Proposed (cycle-12 prelude, this document).
- 2026-05-05 (`d09bed4d`, cycle-12 prelude): Accepted with `sage run --jsonl`
  implementation. cli_started + cli_complete envelope + RuntimeEventLog file
  ↔ stdout TEE + prompt + cancel + approval bridge wired. 4 NYI gaps
  documented in `cli/run.py:21-29` for follow-up (final_seq, set_budget,
  cli_progress, cancel-failure-frame).
- 2026-05-07 (cycle-13 K cli_gaps stage chain): All 4 NYI gaps closed,
  Python backend contract complete:
    - Stage A `2d557b15`: unified stdout seq + populated `cli_complete.payload.final_seq`.
    - Stage B `7bd48c17`: tightening-only `set_budget` command via `CostTracker.tighten_remaining_budget` root guard.
    - Stage C `2ce3c877`: `cli_progress` idle heartbeat (timer-based, 5s cadence with 10s idle guard, 7 canonical stage labels).
    - Stage D `d0bfea2b`: cooperative Python cancellation hardening + terminal `failure(kind="cli_cancel")` frame + v0 limitation documented.
- TBD (cycle-13+): TypeScript adapter `clients/pi-ygn-sage/` shipped on npm with `protocol_version="v0"` pinned. Not yet started.
- TBD: First `protocol_version="v1"` bump (any breaking change to the 18 events / 5 commands / 9 invariants).
