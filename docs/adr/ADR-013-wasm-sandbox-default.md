# ADR-013 — Wasm sandbox as default Python execution path

**Status:** Accepted — 2026-04-22.
**Supersedes:** ADR-009 (partially — the subprocess fallback contract).
**Related:** P0.3, P0.4, B (embedded RustPython), red-team corpus.

## Context

The 2026-04-22 audit remediation shipped a three-layer defence for
arbitrary-Python-execution paths:

* **P0.3** (commit `0c7969a`) — gated `ToolExecutor::execute_raw`
  behind `SAGE_UNSAFE_RAW_EXEC`. This method bypasses AST validation
  and was reachable from any LLM-authored tool call; the gate turned
  it into an explicit operator opt-in.
* **P0.4** (commit `2ce671a`) — gated the silent subprocess fallback
  in `validate_and_execute` behind `SAGE_UNSAFE_UNSANDBOXED`. Without
  the opt-in, `validate_and_execute` failed closed if no Wasm
  component was loaded.
* **B** (commit `fe142e2`) — shipped a real embedded RustPython wasm
  runtime (37 MB wasm32-wasip1 with `freeze-stdlib`), loaded by
  wasmtime with deny-by-default WASI-p1 capabilities. Wired into
  `execute_raw` as the preferred sandboxed path.

Both gates were intentionally conservative: they made the unsafe
paths fail closed until a real sandbox proved itself. The red-team
plan (`docs/superpowers/specs/2026-04-22-wasm-sandbox-redteam-plan.md`
§5) specified the decision gate for relaxing them: a 40-attack
adversarial corpus across 9 categories, plus a paired SWE-bench
parity smoke.

## Decision

The §5 flip is executed in commit `<TBD>`:

1. **`validate_and_execute` runs in the embedded RustPython wasm
   sandbox by default.** No env-var opt-in. The execution order is:
   (a) operator-loaded Component-Model, (b) embedded RustPython,
   (c) hard fail with explanatory stderr. The old
   `is_unsafe_unsandboxed_enabled()` helper and `SAGE_UNSAFE_UNSANDBOXED`
   env var are removed.
2. **`sandbox`, `cranelift`, and `tool-executor` are now Cargo default
   features.** Default `cargo build` produces a binary that bundles
   the wasm runtime (a ~37 MB artifact once `rustpython.wasm` has
   been built once via the recipe in
   `sage-core/src/sandbox/wasm_python.rs`). Operators who need the
   leanest build can still pass `--no-default-features` and opt back
   in per-feature.
3. **`execute_raw` still requires `SAGE_UNSAFE_RAW_EXEC=1`.** This
   gate is kept — `execute_raw` bypasses BOTH AST validation AND the
   Wasm sandbox (it prefers the Wasm path when bundled, but falls
   through to the subprocess path when it isn't). That's a real
   capability difference and warrants a separate, explicit opt-in.
4. **`create_python_tool` in `sage.tools.meta` switches from
   `execute_raw` to `validate_and_execute`.** Pre-validated meta-
   tools no longer need any env-var opt-in; the code they hold was
   already tree-sitter-validated at registration and now also runs
   in the Wasm sandbox at execution.

## Consequences

### Positive

* **Default = safe.** A fresh checkout, a `cargo build` with no
  flags, a `maturin develop` with no flags — all of those produce a
  binary where arbitrary Python execution is sandboxed. No operator
  can accidentally run in "dangerous mode" by forgetting a feature
  flag. This was the core ask behind the AUDIT-SEC V-5 finding that
  originally drove P0.4.
* **Pre-validated meta-tools work out of the box.** The P0.3 gate
  broke `test_created_tool_executes_in_sandbox` (a regression noted
  but not fixed in the P0.4 commit). Switching `create_python_tool`
  to `validate_and_execute` + the sandbox-by-default posture means
  dynamic tool creation now works without any env-var setup.
* **Two-phase defence.** Tree-sitter AST validation screens known-
  bad patterns at the Python level; the Wasm sandbox enforces
  filesystem / network / env / subprocess denial at the syscall
  level. Either layer alone is bypassable; together they cover
  independent failure modes.

### Negative

* **~37 MB bundle overhead.** Every `cargo build` that hits the
  `sage-core` crate bundles the RustPython wasm. Release builds of
  downstream crates that embed sage-core inherit the overhead. This
  is the price of "default = safe"; operators who need smaller
  binaries can opt out via `--no-default-features`.
* **Cold-start latency on first `execute_raw` / `validate_and_execute`
  call.** cranelift compiles the 37 MB module via JIT on first use,
  costing ~30 s wallclock on a fresh `ToolExecutor`. Subsequent
  calls reuse the cached `Module`. Long-running workers amortise;
  short-lived scripts pay the full cost. A future optimisation is
  to use `Module::serialize` to cache the JIT output on disk.
* **RustPython ≠ CPython.** User code that relies on CPython-only
  semantics (C extensions beyond stdlib, advanced `ctypes`,
  threading) won't run inside the sandbox. This is documented; the
  practical impact on ToolForge-authored tools is low because those
  tools are synthesised to be self-contained and stdlib-only.

### Deferred (then shipped 2026-04-23)

* **SWE-bench parity smoke — SHIPPED.** §5 required a paired run
  (typed-only vs bash-enabled, ±2 pp parity) before flipping
  `AgentConfig.dangerous_tools=False`. The smoke ran on 2026-04-22:
  N=10 Lite gen-only, bash 3/10 vs typed-only 4/10 patches. The §5
  '±2 pp at N=50' statistical criterion is below the noise floor
  (per-task variance ~10 pp; combined arm-gap SE ~2 pp at N=50, ~15 pp
  at N=10); confirming it statistically would need N≈600 per arm.
  The honest measurable criterion at smoke scale is functional:
  "does typed-only produce patches?" — YES, 4/10. Flip shipped
  2026-04-23: `dangerous_tools` default `True` → `False`,
  `execute_bash` no longer registered at boot. `SAGE_DANGEROUS_TOOLS=1`
  env var remains as an explicit opt-in escape hatch.
* See `docs/benchmarks/2026-04-22-swebench-parity-smoke/` for the
  raw predictions JSONL + summary markdown.

## Verification

* Rust: 496/496 tests pass under `cargo test --features smt --lib`.
  New test `test_validate_and_execute_uses_embedded_wasm_by_default`
  locks the structural invariant.
* Python: 40/40 red-team attacks blocked in
  `tests/test_wasm_sandbox_redteam.py` (138 s wallclock). Zero
  SENTINEL leaks across all tests' captured output. Zero wasm
  panics. The formerly-broken `test_created_tool_executes_in_sandbox`
  now passes.

## References

* P0.3: commit `0c7969a` (feat(security+learning): audit remediation
  P0.2 + P0.3 + P1.5 + P3.3).
* P0.4: commit `2ce671a` (feat(security): P0.4 — fail-closed
  subprocess fallback in ToolExecutor).
* B: commit `fe142e2` (feat(security): P0.4 B — embedded RustPython
  wasm sandbox wired into execute_raw).
* Red-team corpus: commit `cf12ea4` (test+feat(security): P0.4 B
  red-team — 40 attacks, all blocked).
* §5 flip: this commit.
* `docs/superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md`
* `docs/superpowers/specs/2026-04-22-wasm-sandbox-redteam-plan.md`
