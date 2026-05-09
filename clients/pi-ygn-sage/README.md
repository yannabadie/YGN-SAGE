# `@ygn-sage/pi-adapter` - SAGE CLI v0 JSONL bridge

**Status**: local v0 subprocess JSONL bridge implemented and contract-tested.
It spawns `sage run --jsonl`, writes validated inbound commands as LF-only
JSONL, parses stdout with strict v0 fail-closed validation, and exposes typed
YGN-SAGE pass-through events.

This is not an npm-published package yet, and it is not a pi-mono UI extension.
The bridge does not override backend-owned model selection, topology, tool
policy, budget truth, or learning gates.

## Protocol Contract

The source of truth is `docs/contracts/SAGE_CLI_PROTOCOL.md` in the parent
repo.

Current v0 surface:

- 15 inherited RuntimeEventLog events.
- 4 CLI-shell envelope events:
  `cli_started`, `cli_progress`, `cli_tool_request`, `cli_complete`.
- 19 outbound event types total.
- 5 inbound command types:
  `prompt`, `approve_tool_call`, `deny_tool_call`, `cancel`, `set_budget`.
- 10 runtime-integrity invariants in
  `docs/contracts/runtime-integrity-ledger.md`.

The package pins `SAGE_CLI_PROTOCOL_VERSION = "v0"` and exports value-backed
event/command catalogs from `src/index.ts`.

## What Is Shipped Locally

- `createSageBridge()` returns a bridge object with:
  - `events: AsyncIterable<SageOutboundEvent>`;
  - `send(command)` for the 5 validated inbound commands;
  - `cancel(reason?)`, idempotent at the adapter write layer;
  - `completed`, resolving only after a valid terminal `cli_complete`;
  - `close()` for subprocess cleanup.
- The subprocess is spawned with argv arrays and `shell: false`.
- stdout is parsed as bytes, split only on `0x0A`.
- The parser rejects BOM, raw CR/CRLF, invalid UTF-8, malformed JSON,
  non-object frames, protocol mismatch, unknown v0 events, sequence gaps,
  `run_id` drift, frames after `cli_complete`, missing `cli_complete`, bad
  `final_seq`, and invalid cancel terminal ordering.
- Inbound commands are written as `JSON.stringify(command) + "\n"` only.
- `set_budget` rejects invalid values and adapter-known loosening before
  writing to stdin; the backend remains authoritative for exact remaining
  budget.
- `toSageDisplayEvent()` maps all 19 v0 event types to display metadata
  without converting them into pi-mono model/tool/topology side effects.

## What Is Not Shipped Yet

- No frontend override exists for model selection, topology, cost gates, tool
  policy, or learning gates.
- No npm release readiness is claimed.
- No pi-mono API binding is imported yet.
- No pi-mono UI extension is shipped.
- No benchmark arm C/D result is claimed from this package.
- No real-backend smoke is claimed unless a run artifact explicitly says so.

## Pinning Rules

Per cgpro pivot DESIGN trap #1, pi-mono dependencies stay exact:

```json
"@mariozechner/pi-coding-agent": "0.73.0",
"@mariozechner/pi-ai": "0.73.0"
```

Never let `sage-python` or `sage-core` depend on this package. The dependency
direction is one-way: `pi-ygn-sage` consumes the YGN-SAGE backend via
subprocess.

`external/pi-mono` is optional local reference material, not a required source
for the subprocess bridge. If needed later, setup is via
`scripts/setup_pi_mono.sh`.

## Env Hygiene

Benchmark runs must avoid hidden network side effects before invoking a future
pi-mono bridge:

```bash
export PI_OFFLINE=1
export PI_TELEMETRY=0
export PI_SKIP_VERSION_CHECK=1
```

## JSONL Framing Rules

Both SAGE_CLI_PROTOCOL.md v0 and pi-mono RPC use:

- LF-only delimiter.
- Strict UTF-8, no BOM.
- One JSON object per line.
- Split on byte `0x0A`; do not use a reader that hides delimiter violations.

## Local Checks

```bash
npm run typecheck
npm test
```

The tests use Node's built-in `node:test` runner and do not require Vitest.
They include a fake backend fixture process so chunking, stdin writes, terminal
frames, process exit, and fail-closed parser behavior are exercised without
live API calls.

## References

- SAGE CLI v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`
- Runtime integrity ledger: `docs/contracts/runtime-integrity-ledger.md`
- Cycle-13 arm wiring: `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md`
- pi-mono setup helper: `scripts/setup_pi_mono.sh`
