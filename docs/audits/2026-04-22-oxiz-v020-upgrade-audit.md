# OxiZ v0.2.0 upgrade audit (2026-04-22)

## TL;DR

**The v0.1 → v0.2.0 bump is a one-line Cargo.toml change with zero code edits to `sage-core`.** The 2026-04-20 parking note (`project_oxiz_v020_deferred.md`) was correctly defensive — "300-file restructure" looked breaking — but the restructure was **internal-only**: the public API surface sage-core depends on is preserved either exactly or via source-compatible widening (Vec → IntoIterator, i64 → Into<BigInt>).

Recommendation: **unpark. Bump to `oxiz = { version = "0.2", optional = true }`, run the existing SMT test suite, ship if green.** Rollback cost is reverting one line.

## Current usage in sage-core

Single consumer: `sage-core/src/verification/smt.rs` (the QualityLabeler's SMT backend, behind `smt` feature flag).

API surface actually imported + called:

```rust
use oxiz::{Solver, SolverResult, TermId, TermManager};
```

Methods called (collected via `grep -E 'tm\.mk_|solver\.|tm\.sorts' smt.rs`):

| sage-core call | v0.1 assumption | v0.2.0 actual signature | Compat |
|---|---|---|---|
| `Solver::new()` | `fn new() -> Self` | Same | ✅ |
| `solver.set_logic("QF_LIA")` | `fn set_logic(&self, &str)` | Same | ✅ |
| `solver.assert(term, &mut tm)` | `fn assert(&mut self, TermId, &mut TermManager)` | Same | ✅ |
| `solver.check(&mut tm) -> SolverResult` | Same | Same | ✅ |
| `SolverResult::Unsat`, `::Sat` | enum variants | Same | ✅ |
| `TermManager::new()` | `fn new() -> Self` | Same | ✅ |
| `tm.sorts.int_sort` | field access | field access on `SortManager` (renamed from `Sorts`) | ✅ (field name + access pattern unchanged; type name change is invisible to callers) |
| `tm.sorts.bool_sort` | field access | same | ✅ |
| `tm.mk_var(name, sort)` | `fn mk_var(&mut self, &str, SortId) -> TermId` | Same | ✅ |
| `tm.mk_int(i64_literal)` | `fn mk_int(&mut self, i64)` | `fn mk_int(&mut self, impl Into<BigInt>)` | ✅ source-compat (i64 → BigInt via num-bigint) |
| `tm.mk_false()` | `fn mk_false(&self) -> TermId` | Same | ✅ |
| `tm.mk_gt/lt/ge/le/eq(l, r)` | `fn mk_X(&mut self, TermId, TermId) -> TermId` | Same | ✅ |
| `tm.mk_add(vec![l, r])` | `fn mk_add(&mut self, Vec<TermId>)` | `fn mk_add(&mut self, impl IntoIterator<Item = TermId>)` | ✅ `Vec<T>: IntoIterator<Item = T>` |
| `tm.mk_sub(l, r)` | `fn mk_sub(&mut self, TermId, TermId)` | Same | ✅ |
| `tm.mk_mul(vec![l, r])` | `Vec` | IntoIterator | ✅ |
| `tm.mk_and(vec![l, r])` | `Vec` | IntoIterator | ✅ |
| `tm.mk_or(vec![l, r])` | `Vec` | IntoIterator | ✅ |
| `tm.mk_not(t)` | `fn mk_not(&mut self, TermId)` | Same | ✅ |
| `tm.mk_implies(...)` | mentioned in sage module docstring | Present: `fn mk_implies(&mut self, TermId, TermId) -> TermId` | ✅ |

**Zero call sites require code changes.** Every `vec![a, b]` literal sage-core passes to a variadic continues to work — `Vec<TermId>` satisfies the new `impl IntoIterator<Item = TermId>` bound. Every `tm.mk_int(expected - tolerance)` with `i64` operands continues to compile — `i64` has `impl From<i64> for BigInt` in `num-bigint`.

## What actually moved in v0.2.0

The restructure is **workspace internal**:

```
v0.1: monolithic crate with all code under oxiz/src/
v0.2: umbrella oxiz/ + 15 sub-crates
        oxiz-core, oxiz-math, oxiz-sat, oxiz-theories,
        oxiz-solver, oxiz-opt, oxiz-spacer, oxiz-proof,
        oxiz-nlsat, oxiz-py, oxiz-wasm, oxiz-smtcomp,
        oxiz-cli, oxiz-ml, oxiz-vscode
```

The umbrella `oxiz` crate's `src/lib.rs` re-exports `Solver`, `SolverResult`, `TermId`, `TermManager`, `Sort`, `SortId`, `Term` from their new homes (`oxiz-solver`, `oxiz-core`). External callers who `use oxiz::{Solver, TermManager}` see zero path change.

One rename that does NOT affect sage-core: the sort-storage struct went from `Sorts` → `SortManager`. sage-core uses it only via field access (`tm.sorts.int_sort`), never as a type name, so the rename is invisible.

## New v0.2.0 features

The umbrella crate now exposes four opt-in features via Cargo:

```toml
[features]
default = ["std"]
std = [ "oxiz-core/std", "oxiz-math/std", "oxiz-sat/std",
        "oxiz-theories/std", "oxiz-solver/std" ]
nlsat        = ["dep:oxiz-nlsat", "std"]        # nonlinear real arith
optimization = ["dep:oxiz-opt", "std"]          # MaxSMT / OptSMT
spacer       = ["dep:oxiz-spacer", "std"]       # Horn-clause / IC3-style
proof        = ["dep:oxiz-proof", "std"]        # proof-producing mode
standard     = ["nlsat", "optimization", "proof"]
full         = ["standard", "spacer"]
```

None are required for QualityLabeler (QF_LIA-only). Keep the existing line `oxiz = { version = "0.2", optional = true }` — no features added. Upgrade to `standard` or individual flags is a separate future decision if we ever need nonlinear / optimization queries for a new verification task.

## Migration steps

1. Edit `sage-core/Cargo.toml:45` — change `oxiz = { version = "0.1", optional = true }` → `oxiz = { version = "0.2", optional = true }`.
2. `cargo update -p oxiz` to pick up the new major line.
3. Build: `cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor`. If this succeeds, Rust-level compat confirmed.
4. Test: `cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib`. The QualityLabeler equivalence tests (480 Rust tests currently) must stay green.
5. If steps 3 or 4 fail, the breaking change is localized — diagnose from the compile error or the specific failing test. Most likely spot of surprise: a type-inference ambiguity introduced by `impl Into<BigInt>` widening on `mk_int` when sage-core passes an expression that coerces to multiple integer types. Fix = one `.into()` annotation.

## Rollback

Single-line revert in `sage-core/Cargo.toml`. No code state to roll back in `smt.rs` or elsewhere.

## What this audit does NOT cover

* **Performance regression.** v0.2.0 may have different solve times vs v0.1. QualityLabeler is not on a hot path (Z3 queries are batched per pipeline tick), so small regressions are unlikely to move the needle, but a benchmark sweep is still the right honest check before declaring "done".
* **Workspace-consuming callers.** If any YGN-SAGE code starts calling into `oxiz-nlsat` / `oxiz-opt` / etc. directly (not via the umbrella), this audit is void — those sub-crates have different API surfaces. Currently we do not import any sub-crate, only `oxiz::*`.
* **SMT-LIB textual-input parser.** Not used by sage-core; we build terms programmatically.
* **Python bindings.** Not used; sage-core calls oxiz directly from Rust.

## Decision

**Unpark.** The 2026-04-20 deferral was a reasonable "don't upgrade blindly" precaution; the audit shows the blindspot is narrow. Bump to 0.2, run `cargo test --features smt`, and ship if green. If you want even more conservatism, bump to `version = "0.2.0"` (pinned exact) to avoid picking up a hypothetical 0.2.1 with its own surprises — but sage-core's pin is already `version = "0.1"` (major-line only), so keeping `"0.2"` matches that policy.

---

## 2026-04-22 empirical validation (post-audit)

After this audit was drafted, user pushed back: "did you actually retrieve the new code? do you know what it does? what are potential consequences?" — a well-placed challenge given the "one-line zero risk" framing. Follow-up verification done before committing the bump:

### Dependency-tree check — new transitive sub-crates

`cargo tree --no-default-features --features smt,tool-executor -p sage-core`:

```
sage-core v0.1.0
└── oxiz v0.2.0
    ├── oxiz-core v0.2.0
    ├── oxiz-math v0.2.0
    ├── oxiz-sat v0.2.0
    ├── oxiz-solver v0.2.0
    │   ├── oxiz-proof v0.2.0   ← NEW transitive in v0.2
    │   └── oxiz-theories v0.2.0
    │       └── oxiz-nlsat v0.2.0   ← NEW transitive in v0.2
    └── oxiz-theories v0.2.0 (dup)
```

Two new sub-crates are linked via transitive deps, **not opt-in via features**:

* `oxiz-proof` — proof-generation machinery. Pulled by `oxiz-solver` unconditionally. Binary size grows but CPU cost is zero unless our code calls into proof-producing mode (we don't).
* `oxiz-nlsat` — nonlinear-arithmetic solver. Pulled by `oxiz-theories` unconditionally. Same deal: present but inert as long as we stick to QF_LIA.

**Consequence:** larger compiled binary (~70 MB of extra .rlib in debug; release will strip most of it), no runtime overhead on our existing workload.

### Paired-run semantic + performance check

Flipped `Cargo.toml` back to `oxiz = "0.1"`, ran the test suite, flipped forward to `"0.2"`, ran again. Both configurations built and tested the **same sage-core source tree**.

| Configuration | Tests passed | Total test time | 97 verification tests |
|---------------|-------------:|----------------:|----------------------:|
| v0.1.3 (`oxiz = "0.1"`) | 485 / 485 | 2.13 s | 0.02 s |
| v0.2.0 (`oxiz = "0.2"`) | 485 / 485 | 2.15 s | 0.02 s |

**Semantic equivalence on the test workload:** every SMT formula in the 97 verification tests (bounds checks, loop bounds, arithmetic violations, invariant implications, provider-assignment SAT) produces the same `Sat` / `Unsat` decision in both versions. If v0.2.0 had changed any decision procedure in a way that affects our queries, those specific tests would have failed.

**Performance:** 1 % delta on test execution, deep inside noise. The 97 verification tests stay at 0.02 s total, meaning per-query SMT time is sub-millisecond on average in both versions.

### What this still does NOT prove

* **Non-test production inputs.** A pathological formula our tests don't cover could behave differently. The test suite is a representative sample, not an exhaustive probe.
* **Release-build size delta.** Only measured debug artifacts. Release-stripped `.pyd` size was not benchmarked — low priority since the wheel is a maturin build-artifact, not on a deployment hot path.
* **Long-tail numerical stability.** Integer arithmetic in QF_LIA is exact (no float semantics to drift), so this class of risk is small.

### Net conclusion

Evidence supports the bump. The v0.1 / v0.2 paired run produces identical test outcomes and identical execution time; the two new transitive sub-crates add weight but not work. If a production formula behaves differently, the smoke logs (benches already run multiple times per day) will surface it within one run — and the one-line revert stays available.
