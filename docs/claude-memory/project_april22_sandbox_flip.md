---
name: April 22 — P0.4 B complete + §5 sandbox flip
description: Shipped real embedded RustPython wasm sandbox, ran 40-attack red-team, flipped sandbox to default, removed SAGE_UNSAFE_UNSANDBOXED, published ADR-013. 4 commits pushed to origin.
type: project
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
# 2026-04-22 — P0.4 B + §5 flip

Single session. Started from "continue P0.4 B" (the real embedded Wasm
component), ended with the sandbox-is-default §5 flip and ADR-013.

## Commits (in order, all pushed to origin/main)

1. **`511ac87`** — docs+test(security): A+C+D of P0.4 finish
   * Architectural reframe in safe-sandbox spec
   * `test_double_opt_in_structural_invariants` (4-state matrix test)
   * Red-team plan spec (40 attacks, harness design, §5 decision gate)

2. **`fe142e2`** — feat(security): P0.4 B — embedded RustPython wasm sandbox
   * `sage-core/build.rs` — copies `external/rustpython-wasm-target/.../rustpython.wasm`
     (37.45 MB, RustPython 0.5.0 wasm32-wasip1 + freeze-stdlib) into OUT_DIR
   * `sage-core/src/sandbox/wasm_python.rs` — `WasmPythonExecutor` via wasmtime 43
     + cranelift JIT. Deny-by-default WASI-p1: no fs, no network, no subprocess,
     no env inheritance, no stdio inheritance. 64 KiB stdout/stderr pipes.
     Epoch-interrupt timeout via watchdog thread.
   * Wired into `execute_raw` as preferred path (keeps subprocess fallback
     when bytes unavailable).
   * `ENV_MUTEX` in tool_executor tests (new wasm tests hold process for
     30-60s each, which races with env-var-mutating structural tests).
   * Rustc upgraded 1.94.0 → 1.95.0 (RustPython 0.5.0 requires it).

3. **`cf12ea4`** — test+feat(security): P0.4 B red-team — 40 attacks, all blocked
   * `sage-python/tests/test_wasm_sandbox_redteam.py` — 40 attacks across
     9 categories (FS read/write, network, subprocess, env secrets, clock,
     memory DoS, introspection, engine-level). 138s wallclock.
   * Two real Rust-side bugs uncovered + fixed:
     - **Monotonic epoch deadlines** (AtomicU64). `set_epoch_deadline(1)` was
       ABSOLUTE — after first watchdog bumped shared engine's epoch to 1,
       all subsequent calls started already past deadline and trapped with
       no output. Rust tests missed it because each test creates a fresh
       executor; Python uses scope="module" (correct usage pattern) and
       exposed it. Fix: `fetch_add` a fresh deadline per call.
     - **256 MiB StoreLimits memory cap**. wasm32 default is 4 GiB, MEM-1's
       `[0] * (10 ** 9)` would have eaten host RAM. Added via
       `StoreLimitsBuilder::new().memory_size(256 * 1024 * 1024)`.
   * New `sage_core.embedded_wasm_available()` PyO3 function — the plan's
     draft harness skip-gate `ToolExecutor.has_wasm()` only answers for
     Component-Model path, NOT my execute_raw→embedded-RustPython path.
     Without this, harness would have silently skipped all 40 tests.

4. **`c2113d8`** — feat(security): P0.4 §5 flip — wasm sandbox is now default
   * `validate_and_execute` runs through embedded RustPython by default,
     no opt-in. Order: Component-Model → embedded RustPython → HARD FAIL
     (subprocess fallback REMOVED).
   * `SAGE_UNSAFE_UNSANDBOXED` env var removed. `execute_raw` still gated
     by `SAGE_UNSAFE_RAW_EXEC` (bypasses AST + sandbox = different capability).
   * `sandbox` + `cranelift` + `tool-executor` moved into Cargo default
     features. `cargo build` with no flags bundles the wasm runtime.
   * `create_python_tool` in `sage.tools.meta` switched from `execute_raw`
     to `validate_and_execute` — fixes the pre-existing P0.3 regression of
     `test_created_tool_executes_in_sandbox`.
   * Wrapper contract: `json`, `sys` exposed as top-level imports in user
     scope (matches old subprocess contract); `codecs` stays underscore-
     prefixed (internal only).
   * Tests rewritten: `test_validate_and_execute_uses_embedded_wasm_by_default`
     and `test_execute_raw_gate_independent_of_validate_and_execute` (2-state
     matrix collapsed from 4 because only one gate remains).
   * ADR-013 at `docs/adr/ADR-013-wasm-sandbox-default.md`.

## Key architectural decisions

* **No subprocess fallback in `validate_and_execute` anymore.** If the
  Wasm sandbox is absent (no bundled rustpython.wasm AND no loaded
  Component-Model), `validate_and_execute` hard-fails with explanatory
  stderr — "rebuild with --features sandbox,cranelift or load a
  component". The subprocess code is still in the crate, but only
  reachable via `execute_raw` (which requires `SAGE_UNSAFE_RAW_EXEC`).

* **`execute_raw` stays gated.** It prefers Wasm when bundled, falls
  through to subprocess otherwise. The gate is because it bypasses AST
  validation — that's a real capability difference vs `validate_and_execute`.

* **Orthogonal to `dangerous_tools`.** ADR-013 is about the SANDBOX
  default. The bash-tool-default-on (`AgentConfig.dangerous_tools=True`)
  is a separate item tracked for a follow-up commit gated on paired
  SWE-bench smoke (typed-only vs bash, ±2pp parity) — ~4h, $20-50,
  deferred at user's direction.

## What surprised me (advisor saves)

* **First advisor save (during B commit)**: I was about to wire wasm_python
  into `validate_and_execute` as part of the initial B commit. Advisor
  pointed out my own red-team plan §5 says "flip defaults AFTER red-team
  passes" — doing it in B would invalidate the structural tests I'd just
  shipped in A+C+D. Reverted, kept B scope clean.
* **Second advisor save (before red-team run)**: The plan's draft skip-gate
  was `executor.has_wasm()` — which only returns True for Component-Model
  path. My embedded RustPython was NOT covered. Would have silently
  skipped all 40 tests. Fix: expose `sage_core.embedded_wasm_available()`
  through PyO3.

## Baseline post-session

* Rust: 496 passed, 0 failed.
* Python: 1999 passed (+40 red-team, +1 fixed regression vs 2026-04-20
  baseline of 1958).
* Default `cargo build` bundles wasm, no flags needed.
* Red-team: 40/40 attacks blocked, zero SENTINEL leaks, 138s wallclock.

## Deferred follow-ups

* Paired SWE-bench smoke (typed-only vs bash, ±2pp parity) — prerequisite
  for `AgentConfig.dangerous_tools=False` default flip.
* Remove `execute_bash` registration branch from boot.py after parity
  smoke validates.
* `Module::serialize` → on-disk cache for JIT output (currently ~30s
  cold-start on first execute_raw / validate_and_execute call).
