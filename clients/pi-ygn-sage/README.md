# `@ygn-sage/pi-adapter` - pi-mono / YGN-SAGE bridge scaffold

**Status**: contract-correct scaffolding only. The subprocess bridge and
pi-mono integration are still implementation NYI and must not be described as
shipped.

**Owner**: this directory will ship as an npm package wrapping
`sage run --jsonl` as a subprocess + JSONL bridge for pi-mono coding-agent.
For now it exports the SAGE CLI v0 TypeScript catalog/types and tests that the
stub remains honest.

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

## What Is Not Shipped Yet

- `createSageBridge()` is intentionally unimplemented and throws
  `implementation NYI`.
- No subprocess is spawned.
- No stdin `prompt` behavior is implemented or normalized by this package.
- No frontend override exists for model selection, topology, cost gates, tool
  policy, or learning gates.
- No npm release readiness is claimed.
- No pi-mono API binding is imported yet.

## Pinning Rules

Per cgpro pivot DESIGN trap #1, pi-mono dependencies stay exact:

```json
"@mariozechner/pi-coding-agent": "0.73.0",
"@mariozechner/pi-ai": "0.73.0"
```

Never let `sage-python` or `sage-core` depend on this package. The dependency
direction is one-way: future `pi-ygn-sage` consumes the YGN-SAGE backend via
subprocess.

`external/pi-mono` is optional local reference material, not a required source
for this sync-only scaffold. If needed later, setup is via
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

## References

- SAGE CLI v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`
- Runtime integrity ledger: `docs/contracts/runtime-integrity-ledger.md`
- Cycle-13 arm wiring: `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md`
- pi-mono setup helper: `scripts/setup_pi_mono.sh`
