# `@ygn-sage/pi-adapter` — pi-mono ↔ YGN-SAGE bridge (cycle-13 scaffolding)

**Status**: scaffolding only (cycle-12 closeout, 2026-05-05). Cycle-13 implements.
**Owner**: this directory ships as an npm package wrapping `sage run --jsonl` as a subprocess + JSONL bridge for pi-mono coding-agent (arm C in the cycle-13 SWE-bench Pro 4-arm ablation).

## Why this exists

The pivot strategy (cgpro `cgpro_pi_mono_pivot_20260505` 2026-05-05 verdict APPROVE_WITH_FOLLOWUPS):

> *pi-mono = front-end UX/transport. YGN-SAGE = orchestration backend. Communicating via subprocess + JSONL/RPC (NOT MCP).*

This package is the bridge. It:
- Spawns `python -m sage.cli run --jsonl` as a subprocess.
- Pipes stdin (pi-mono inbound commands) → sage stdin.
- Pipes sage stdout (RuntimeEventLog v0 + cli envelope) → pi-mono frontend events.
- Translates protocol semantics where SAGE_CLI_PROTOCOL.md and pi-mono's RPC spec disagree.

The protocol contract is `docs/contracts/SAGE_CLI_PROTOCOL.md` in the parent repo (v0, 9 invariants including invariant 9 "CLI protocol versioning").

## Pinning rules (cycle-13 trap mitigation)

Per cgpro pivot DESIGN trap #1 (pi-mono v0.73 is young, churn risk), this package MUST:

- Pin pi-mono dependencies to **EXACT** versions (no `^` or `~` ranges):
  ```json
  "@mariozechner/pi-coding-agent": "0.73.0",
  "@mariozechner/pi-ai": "0.73.0"
  ```
- Never let `sage-python` or `sage-core` depend on this package (one-way dependency: `pi-ygn-sage` depends on YGN-SAGE backend via subprocess, NOT vice-versa).
- Track upstream pi-mono via `external/pi-mono` git clone pinned to commit `dbcb473d6fdb96f60570b9ebe73e7aa6316fa8fb` (v0.73.0). Setup via `scripts/setup_pi_mono.sh`.

## Env hygiene (cgpro DESIGN E trap Q3)

Benchmark runs MUST NOT have hidden network side-effects. Set these env vars before invoking pi-mono:

```bash
export PI_OFFLINE=1            # no network update checks during bench
export PI_TELEMETRY=0          # no install telemetry beacons
export PI_SKIP_VERSION_CHECK=1 # no startup version check
```

## JSONL framing rules

Both protocols (SAGE_CLI_PROTOCOL.md v0 + pi-mono RPC) use:
- **LF-only delimiter** (NOT CRLF, NOT Unicode line separators).
- **Strict UTF-8 encoding**, no BOM.
- One JSON object per line, each line independent.
- Reader MUST split on `\n` (single byte 0x0A) — `readline.createInterface()` is FORBIDDEN.

Tests in `test_sage_cli_jsonl.py:test_jsonl_only_lf_delimited` (cycle-12 prelude) lock this for the YGN side. The adapter MUST mirror.

## Package status

- **TODO (cycle-13)**: implement `src/index.ts` with the subprocess bridge + protocol translation layer.
- **TODO (cycle-13)**: tests via `vitest` or similar.
- **TODO (cycle-13)**: wheels CI integration so this package can release alongside YGN-SAGE.

This README + package.json + tsconfig.json + src/index.ts stub ship in cycle-12 closeout to lock the directory shape and pin policy. No actual TypeScript implementation lives here yet.

## References

- Pivot strategy: `C:/Users/yann.abadie/.claude/plans/abstract-finding-pixel.md` (cycle-12 prelude).
- SAGE_CLI_PROTOCOL v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`.
- Cycle-13 4-arm wiring: `docs/benchmarks/2026-05-05-cycle13-arm-wiring.md`.
- pi-mono coding-agent docs: `external/pi-mono/packages/coding-agent/`.
- cgpro pivot review: conv `cgpro_pi_mono_pivot_20260505` 2026-05-05.
