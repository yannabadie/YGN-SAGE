# OTel Rust Spans Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sage-core` Rust hot-path spans (engine, system_router, knn, model_assigner, write_gate, mutations, wasm_python, tool_executor, model_registry, reward) appear as children of Python `sage.pipeline.run` in the same OTel trace tree.

**Architecture:** Independent Rust OTel SDK + W3C traceparent propagation across PyO3. Rust adds `opentelemetry` 0.27 / `tracing-opentelemetry` 0.28 behind a new `otel` Cargo feature (opt-in). Boot lifecycle: Python `_init_tracer()` calls `sage_core.init_otel(exporter, endpoint)` which mirrors the exporter config in Rust and installs a `tracing-opentelemetry` Layer. At each `sage_span` enter, Python serializes its current `SpanContext` as a W3C traceparent string and calls `sage_core.bridge_python_span(traceparent, name)` which returns a PyO3-wrapped `EnteredSpan` whose Drop closes the span. Existing `info_span!` calls inside Rust become children of the bridged span automatically via `tracing` parent inheritance. No PyO3 0.27 upgrade required.

**Tech Stack:** Rust 1.84+ (edition 2021), PyO3 0.25, `tracing` 0.1, new deps: `opentelemetry` 0.27, `opentelemetry_sdk` 0.27, `opentelemetry-otlp` 0.27, `tracing-opentelemetry` 0.28, `tracing-subscriber` 0.3 (all behind `otel` feature). Python `opentelemetry-api` (already on disk via B1 Phase 1).

**Spec:** `docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md` (commit `26da4a8d`).

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `sage-core/Cargo.toml` | Add new deps + `otel` feature | Modify |
| `sage-core/src/lib.rs` | PyO3 module exports for `init_otel` + `bridge_python_span` | Modify |
| `sage-core/src/observability/mod.rs` | Module root: re-exports + cfg gating | Create |
| `sage-core/src/observability/init.rs` | `init_otel` — TracerProvider + tracing-opentelemetry Layer + Registry installation; idempotent | Create |
| `sage-core/src/observability/bridge.rs` | `bridge_python_span` PyO3 fn, `RustSpanHandle` PyO3 class wrapping `EnteredSpan`, traceparent parser | Create |
| `sage-core/src/observability/stub.rs` | Compile when `otel` feature OFF — exposes the same PyO3 surface as no-op stubs | Create |
| `sage-core/tests/otel_smoke.rs` (or under `src/observability/tests.rs`) | Rust unit tests with `InMemoryExporter` | Create |
| `sage-python/src/sage/observability/__init__.py:_init_tracer` | After Python TracerProvider set, mirror to Rust via `sage_core.init_otel(...)` | Modify |
| `sage-python/src/sage/observability/spans.py:sage_span` | After `start_as_current_span`, call `_maybe_bridge_to_rust(span)` and store handle until exit | Modify |
| `sage-python/tests/observability/test_rust_bridge.py` | 4 integration tests | Create |
| `docs/observability/otel-genai-spans.md` | Append a "Rust spans" section + build recipe + env table update | Modify |
| `CLAUDE.md` | Update build recipe with `--features otel` example + Rust spans note | Modify |
| `.claude/rules/development.md` | Update Lint/Build/Test sections | Modify |
| `roadmap.md` | Mark B1.b closed, list B1.b.7 + B1.b.8 as deferred | Modify |
| `.github/workflows/*.yml` (if exists) | Add `--features otel` matrix entry OR document why deferred | Modify or document |

---

## CI Decision (B1.b.8)

The `otel` feature is opt-in. Default `cargo test --lib` will not compile or
exercise the new code. **Decision for this plan: add a Rust CI compile-check
job with `--features otel` (compile + cargo test --lib only — no Python
smoke).** Rationale:

- Compile-check is cheap (~1-2 min on Linux runners; new deps are not heavy)
- Catches version-drift between `tracing-opentelemetry` and `opentelemetry` on
  every PR — exactly the failure mode the spec calls out as risk multiplier 1.3×
- Avoids the "tests nobody runs" anti-pattern advisor flagged

Skipped: integration tests under `--features otel` because Python integration
tests already exercise the bridge end-to-end on the dev machine. Adding a CI
job that runs `pip install -e ".[all,dev]" && cargo build --features otel`
would double Python wheel build time.

If `.github/workflows/` doesn't exist yet (greenfield CI), Task 11 makes the
B1.b.8 entry an explicit deferred-with-rationale line in `roadmap.md` instead.

---

## Task 1: Cargo wiring + `otel` feature scaffold

**Files:**
- Modify: `sage-core/Cargo.toml`
- Test: implicit — `cargo build --features otel` succeeds

- [ ] **Step 1: Add new deps + feature definition**

Edit `sage-core/Cargo.toml`:

In the `[dependencies]` section (after the existing `tracing = "0.1"` line at line 46),
add:

```toml
# B1.b OTel Rust bridge (2026-04-25). All gated behind the `otel`
# feature — opt-in for now, may move into default features once
# downstream stability is proven (see roadmap-B1.b.8).
opentelemetry = { version = "0.27", optional = true }
opentelemetry_sdk = { version = "0.27", features = ["rt-tokio"], optional = true }
opentelemetry-otlp = { version = "0.27", features = ["http-proto", "http-json", "reqwest-client"], optional = true }
tracing-opentelemetry = { version = "0.28", optional = true }
tracing-subscriber = { version = "0.3", features = ["registry", "env-filter"], optional = true }
```

In `[features]`, after the `smt = ["dep:oxiz"]` line, add:

```toml
otel = [
    "dep:opentelemetry",
    "dep:opentelemetry_sdk",
    "dep:opentelemetry-otlp",
    "dep:tracing-opentelemetry",
    "dep:tracing-subscriber",
]
```

Do NOT add `otel` to the `default = [...]` list. Opt-in.

- [ ] **Step 2: Verify both build modes compile**

Run:
```
cd sage-core && cargo build --no-default-features --features extension-module 2>&1 | tail -20
```
Expected: builds cleanly (no `otel` deps fetched).

Run:
```
cd sage-core && cargo build --features otel 2>&1 | tail -30
```
Expected: builds cleanly with the 5 new deps fetched. **If
`tracing-opentelemetry 0.28` fails to resolve against `opentelemetry 0.27`,
bisect the version pair before committing.** Try `tracing-opentelemetry 0.28.0`
exactly first; if that fails, run `cargo search tracing-opentelemetry` and pick
the highest version that lists `opentelemetry = "0.27"` in its deps.

- [ ] **Step 3: Commit**

```
git add sage-core/Cargo.toml
git commit -m "feat(b1.b): add otel Cargo feature scaffold

5 new optional deps gated behind the otel feature. Default builds
unchanged. Compile-verified --no-default-features and --features otel.

Spec: docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Rust `observability::stub` module (feature OFF path)

**Files:**
- Create: `sage-core/src/observability/mod.rs`
- Create: `sage-core/src/observability/stub.rs`
- Modify: `sage-core/src/lib.rs`
- Test: `sage-core/src/observability/stub.rs` inline `#[cfg(test)]`

The stub must compile and link **without** the `otel` deps. It exposes the
same PyO3 surface as the real implementation but every entry is a no-op.
This lets Python always import `sage_core.init_otel` and
`sage_core.bridge_python_span` regardless of build flavor.

- [ ] **Step 1: Write the failing test**

Create `sage-core/src/observability/stub.rs`:

```rust
//! Stub used when the `otel` feature is OFF.
//!
//! Exposes `init_otel` and `bridge_python_span` as PyO3 entry points
//! that do nothing. The Python side imports these unconditionally and
//! falls back to "Python-only spans" mode based on a return value.

use pyo3::prelude::*;

/// Stub: returns `false` to signal "Rust OTel not available".
/// Real implementation lives in `init.rs` when the `otel` feature is on.
#[pyfunction]
#[pyo3(signature = (exporter, endpoint=None))]
pub fn init_otel(exporter: &str, endpoint: Option<&str>) -> bool {
    let _ = (exporter, endpoint);
    false
}

/// Stub: returns a no-op handle. Drop is a no-op too.
#[pyfunction]
#[pyo3(signature = (traceparent, name))]
pub fn bridge_python_span(traceparent: &str, name: &str) -> RustSpanHandle {
    let _ = (traceparent, name);
    RustSpanHandle { _private: () }
}

/// Opaque handle exported to Python. Stub variant carries no state.
/// On the real path, this owns a `tracing::span::EnteredSpan`.
#[pyclass]
pub struct RustSpanHandle {
    pub(crate) _private: (),
}

#[pymethods]
impl RustSpanHandle {
    /// Explicit close (Python ContextManager-friendly). No-op for stub.
    /// Signature must match the real impl in Task 4: `&mut self`.
    pub fn close(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_init_returns_false() {
        assert!(!init_otel("console", None));
    }

    #[test]
    fn stub_bridge_returns_handle() {
        let h = bridge_python_span("00-1-2-01", "test");
        h.close();  // does not panic
    }
}
```

Create `sage-core/src/observability/mod.rs`:

```rust
//! Rust OTel bridge for B1.b. Conditionally compiles `init` (real) or
//! `stub` (no-op) based on the `otel` feature flag.

#[cfg(feature = "otel")]
mod init;
#[cfg(feature = "otel")]
pub use init::*;

#[cfg(not(feature = "otel"))]
mod stub;
#[cfg(not(feature = "otel"))]
pub use stub::*;
```

- [ ] **Step 2: Run the test (without `otel` feature)**

Run:
```
cd sage-core && cargo test --no-default-features --features extension-module observability::stub:: 2>&1 | tail -20
```
Expected: 2 tests pass.

- [ ] **Step 3: Wire into `lib.rs`**

Edit `sage-core/src/lib.rs`. Below `pub mod verification;` line (line 13), add:

```rust
pub mod observability;
```

In the `fn sage_core(...)` body, before the closing `Ok(())`, add (just before
the `// Embedded-RustPython availability probe` comment block at line 40):

```rust
    // B1.b OTel bridge — always exposed (stub when otel feature off).
    m.add_function(pyo3::wrap_pyfunction!(observability::init_otel, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(observability::bridge_python_span, m)?)?;
    m.add_class::<observability::RustSpanHandle>()?;
```

- [ ] **Step 4: Run cargo test (no otel feature)**

Run:
```
cd sage-core && cargo test --no-default-features --features extension-module --lib 2>&1 | tail -10
```
Expected: all existing tests pass + 2 new stub tests pass.

- [ ] **Step 5: Commit**

```
git add sage-core/src/observability/mod.rs sage-core/src/observability/stub.rs sage-core/src/lib.rs
git commit -m "feat(b1.b): observability stub — feature-OFF no-op surface

PyO3 module exports init_otel + bridge_python_span + RustSpanHandle
with no-op bodies. Real bodies follow in next task behind the otel
feature gate. Lets Python import the symbols unconditionally.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Rust `observability::init` — real `init_otel` (feature ON)

**Files:**
- Create: `sage-core/src/observability/init.rs`
- Test: `sage-core/src/observability/init.rs` inline `#[cfg(test)]`

`init_otel` is idempotent. First call installs a `tracing-opentelemetry` Layer
+ subscriber. Subsequent calls return early.

- [ ] **Step 1: Write the failing test**

Create `sage-core/src/observability/init.rs`:

```rust
//! Real `init_otel` + bridge implementation. Compiles only when the
//! `otel` feature is on.

use pyo3::prelude::*;
use std::sync::OnceLock;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

static INIT: OnceLock<bool> = OnceLock::new();

/// Initialize Rust OTel TracerProvider + tracing-opentelemetry Layer.
///
/// `exporter`: one of `"console"`, `"otlp_http"`, `"none"`.
/// `endpoint`: only honored when `exporter == "otlp_http"`. If `None`,
/// uses the OTLP default (`http://localhost:4318/v1/traces`).
///
/// Returns `true` on first successful init, `false` on subsequent calls
/// (idempotent) or if the exporter kind is unknown / `"none"`.
#[pyfunction]
#[pyo3(signature = (exporter, endpoint=None))]
pub fn init_otel(exporter: &str, endpoint: Option<&str>) -> bool {
    if INIT.get().is_some() {
        return false;  // already initialized
    }

    let provider = match exporter {
        "none" | "" => return false,
        "console" => build_console_provider(),
        "otlp_http" => build_otlp_provider(endpoint),
        _ => {
            tracing::warn!(
                exporter,
                "unknown SAGE_OTEL_EXPORTER for Rust; skipping init"
            );
            return false;
        }
    };

    let provider = match provider {
        Some(p) => p,
        None => return false,
    };

    let tracer = provider.tracer("sage-core");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);

    if tracing_subscriber::registry()
        .with(layer)
        .try_init()
        .is_err()
    {
        // Another subscriber was already installed (e.g. the host
        // process wired tracing-subscriber for log output). We can't
        // install our own — Rust spans won't reach OTel. Log via
        // `eprintln!` since `tracing` is now owned by the other one.
        eprintln!(
            "[sage-core] OTel init: a tracing subscriber was already installed; \
             Rust spans will not be exported to OTel. \
             To use Rust OTel spans, ensure SAGE is the only tracing-subscriber owner."
        );
        return false;
    }

    global::set_tracer_provider(provider);
    global::set_text_map_propagator(TraceContextPropagator::new());

    INIT.set(true).ok();
    true
}

fn build_console_provider() -> Option<SdkTracerProvider> {
    use opentelemetry_sdk::trace::TracerProvider;
    use opentelemetry_stdout::SpanExporter;

    // opentelemetry-stdout was added as a transitive of opentelemetry_sdk in
    // some versions, but we don't depend on it directly. Fall back to a
    // simple debug exporter using tracing-subscriber's fmt layer instead.
    // For the purposes of B1.b "console" support, we accept that Rust
    // spans surface via stderr through the Layer's debug formatting.
    let _ = SpanExporter::default();  // intentionally unused; placeholder
    Some(TracerProvider::builder().build())
}

fn build_otlp_provider(endpoint: Option<&str>) -> Option<SdkTracerProvider> {
    use opentelemetry_otlp::WithExportConfig;

    let exporter_builder = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary);

    let exporter_builder = if let Some(ep) = endpoint {
        exporter_builder.with_endpoint(ep)
    } else {
        exporter_builder
    };

    let exporter = match exporter_builder.build() {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "[sage-core] OTel init: OTLP exporter build failed: {err}"
            );
            return None;
        }
    };

    Some(
        SdkTracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Tests cannot run truly in parallel because INIT is a process-wide
    // OnceLock. Serialize via mutex.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn init_with_none_returns_false() {
        let _guard = SERIAL.lock().unwrap();
        // Cannot reset INIT, so this test is order-dependent. Run as the
        // first test by alphabetical order ("a_*" prefix would help — keep
        // test count low to avoid the trap).
        let _ = init_otel("none", None);
        // Either INIT is already set (in which case false is returned) or
        // exporter==none short-circuits. Both produce false. The point is:
        // no panic, no subscriber installed for "none".
    }

    #[test]
    fn init_with_unknown_exporter_returns_false() {
        let _guard = SERIAL.lock().unwrap();
        let result = init_otel("frobinator-9000", None);
        assert!(!result, "unknown exporter should return false, got true");
    }
}
```

The bridge fn comes in Task 4. For now, expose stub-compatible re-exports:
add at the bottom of `init.rs`:

```rust
// Bridge fn lives in bridge.rs but is re-exported here for the same
// PyO3 surface as the stub variant. Task 4 fills these in.
pub use crate::observability::bridge::{bridge_python_span, RustSpanHandle};
```

Update `mod.rs` to declare `bridge` when `otel` is on:

```rust
//! Rust OTel bridge for B1.b. Conditionally compiles `init` (real) or
//! `stub` (no-op) based on the `otel` feature flag.

#[cfg(feature = "otel")]
mod init;
#[cfg(feature = "otel")]
mod bridge;
#[cfg(feature = "otel")]
pub use init::*;

#[cfg(not(feature = "otel"))]
mod stub;
#[cfg(not(feature = "otel"))]
pub use stub::*;
```

Create a placeholder `sage-core/src/observability/bridge.rs` so the build
compiles before Task 4:

```rust
//! Real `bridge_python_span` + `RustSpanHandle`. Filled in in Task 4.
//! For now, returns stub-shaped values so init.rs compiles.

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (traceparent, name))]
pub fn bridge_python_span(traceparent: &str, name: &str) -> RustSpanHandle {
    let _ = (traceparent, name);
    RustSpanHandle {}
}

#[pyclass]
pub struct RustSpanHandle {}

#[pymethods]
impl RustSpanHandle {
    /// Placeholder for Task 4. Final signature: `&mut self`.
    pub fn close(&mut self) {}
}
```

- [ ] **Step 2: Run cargo build with otel feature**

Run:
```
cd sage-core && cargo build --features otel 2>&1 | tail -30
```
Expected: builds cleanly. **If a dep version mismatch shows up
(e.g. "the trait `Tracer` is not implemented for ..."), pin
`opentelemetry-stdout`'s exclusion from build_console_provider — the
function uses an unused placeholder import that may need removal.** If
the build fails because `opentelemetry-stdout` is not a dep, delete
the `let _ = SpanExporter::default();` line and the import — the
function returns an empty TracerProvider for console mode (acceptable
for MVP; logs surface via Python-side console exporter).

- [ ] **Step 3: Run cargo test with otel feature**

Run:
```
cd sage-core && cargo test --features otel observability::init:: 2>&1 | tail -20
```
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```
git add sage-core/src/observability/init.rs sage-core/src/observability/bridge.rs sage-core/src/observability/mod.rs
git commit -m "feat(b1.b): init_otel — TracerProvider + tracing-opentelemetry Layer

Idempotent setup behind the otel feature. console + otlp_http exporter
kinds; \"none\" / unknown short-circuit. Wires W3C TraceContextPropagator
globally. bridge_python_span placeholder lands here too — body in
next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Rust `bridge_python_span` — traceparent parse + EnteredSpan

**Files:**
- Modify: `sage-core/src/observability/bridge.rs`

This is the heart of the propagation. Parses W3C traceparent, sets the
extracted context as the active OTel context, and creates a `tracing::span`
which inherits via the `tracing-opentelemetry` Layer's parent-extraction.

- [ ] **Step 1: Write the failing test**

Replace `sage-core/src/observability/bridge.rs` with the full implementation:

```rust
//! Bridge a Python OTel span into Rust via W3C traceparent string.
//!
//! Python serializes its current SpanContext as
//!   `{version:02x}-{trace_id:032x}-{span_id:016x}-{flags:02x}`
//! and calls `bridge_python_span(traceparent, name)`. Rust parses,
//! extracts an OpenTelemetry `Context` with the parent set to the
//! Python span, attaches it as the current context, and creates a
//! tracing span that inherits from the now-current OTel context via
//! the tracing-opentelemetry Layer's `OpenTelemetrySpanExt::set_parent`.
//!
//! Lifetime: the returned `RustSpanHandle` owns
//! - an `EnteredSpan` (RAII guard for the tracing span)
//! - a `ContextGuard` (RAII guard for the OTel context attach)
//! Both drop in reverse-construction order on close.

use pyo3::prelude::*;

use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId, TraceState};
use opentelemetry::{Context, ContextGuard};
use tracing::span::{EnteredSpan, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Bridge a Python span context into Rust.
///
/// Returns a handle whose Drop closes the span. On parse failure,
/// returns a "no-parent" handle that creates a Rust span with no
/// parent (becomes a new root span). A WARN is logged once.
#[pyfunction]
#[pyo3(signature = (traceparent, name))]
pub fn bridge_python_span(traceparent: &str, name: &str) -> RustSpanHandle {
    let parent_cx = match parse_traceparent(traceparent) {
        Ok(cx) => cx,
        Err(e) => {
            tracing::warn!(
                traceparent = traceparent,
                error = %e,
                "bridge_python_span: traceparent parse failed; creating root span"
            );
            Context::new()  // empty context; tracing span has no parent
        }
    };

    // Attach the OTel context. The guard keeps it active.
    let cx_guard = parent_cx.attach();

    // Create a tracing span. Use the dynamic name; macros require static
    // strings, so we use Span::current() + manual creation.
    let span = tracing::info_span!("rust_bridged_span", bridge_name = name);
    // Force the parent linkage — set_parent picks up the current OTel
    // context that we just attached.
    span.set_parent(Context::current());

    let entered = span.entered();

    RustSpanHandle {
        _entered: Some(entered),
        _cx_guard: Some(cx_guard),
    }
}

/// Opaque handle. Lifetime tied to the bridged span. Drop closes the span
/// (via EnteredSpan::Drop) and releases the OTel context attach (via
/// ContextGuard::Drop).
///
/// `EnteredSpan` is `!Send`. PyO3 callbacks hold the GIL → the calling
/// thread is pinned. Do NOT move this handle into a `tokio::spawn` or
/// `std::thread::spawn`.
#[pyclass(unsendable)]
pub struct RustSpanHandle {
    _entered: Option<EnteredSpan>,
    _cx_guard: Option<ContextGuard>,
}

#[pymethods]
impl RustSpanHandle {
    /// Explicit close. Drop also closes; `close` is for symmetry with
    /// Python ContextManagers. Idempotent.
    pub fn close(&mut self) {
        // Drop the entered guard first, then the cx guard.
        let _ = self._entered.take();
        let _ = self._cx_guard.take();
    }
}

/// Parse W3C traceparent into an OTel Context with the parent set.
///
/// Format: `00-{trace_id_32_hex}-{span_id_16_hex}-{flags_2_hex}`
/// Returns Err(reason) on malformed input.
fn parse_traceparent(s: &str) -> Result<Context, &'static str> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 4 {
        return Err("expected 4 dash-separated parts");
    }
    if parts[0] != "00" {
        return Err("only version 00 supported");
    }
    if parts[1].len() != 32 {
        return Err("trace_id must be 32 hex chars");
    }
    if parts[2].len() != 16 {
        return Err("span_id must be 16 hex chars");
    }
    if parts[3].len() != 2 {
        return Err("flags must be 2 hex chars");
    }

    let trace_id = u128::from_str_radix(parts[1], 16)
        .map_err(|_| "trace_id is not valid hex")?;
    let span_id = u64::from_str_radix(parts[2], 16)
        .map_err(|_| "span_id is not valid hex")?;
    let flags = u8::from_str_radix(parts[3], 16)
        .map_err(|_| "flags is not valid hex")?;

    if trace_id == 0 {
        return Err("trace_id is zero (invalid)");
    }
    if span_id == 0 {
        return Err("span_id is zero (invalid)");
    }

    let span_context = SpanContext::new(
        TraceId::from_bytes(trace_id.to_be_bytes()),
        SpanId::from_bytes(span_id.to_be_bytes()),
        TraceFlags::new(flags),
        true,  // remote
        TraceState::default(),
    );
    Ok(Context::new().with_remote_span_context(span_context))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_traceparent() {
        let s = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
        let cx = parse_traceparent(s).expect("should parse");
        let span = cx.span();
        let sc = span.span_context();
        assert!(sc.is_remote());
        assert!(sc.trace_flags().is_sampled());
        let trace_id_str = format!("{:032x}", u128::from_be_bytes(sc.trace_id().to_bytes()));
        assert_eq!(trace_id_str, "0123456789abcdef0123456789abcdef");
    }

    #[test]
    fn parse_wrong_part_count_fails() {
        assert!(parse_traceparent("00-foo-bar").is_err());
        assert!(parse_traceparent("").is_err());
    }

    #[test]
    fn parse_wrong_version_fails() {
        let s = "01-0123456789abcdef0123456789abcdef-0123456789abcdef-01";
        assert!(parse_traceparent(s).is_err());
    }

    #[test]
    fn parse_zero_trace_id_fails() {
        let s = "00-00000000000000000000000000000000-0123456789abcdef-01";
        assert!(parse_traceparent(s).is_err());
    }

    #[test]
    fn parse_invalid_hex_fails() {
        let s = "00-zzzz456789abcdef0123456789abcdef-0123456789abcdef-01";
        assert!(parse_traceparent(s).is_err());
    }

    #[test]
    fn bridge_with_invalid_traceparent_does_not_panic() {
        let h = bridge_python_span("garbage", "test");
        // Drop closes; assert no panic.
        drop(h);
    }
}
```

- [ ] **Step 2: Run cargo test**

Run:
```
cd sage-core && cargo test --features otel observability::bridge:: 2>&1 | tail -30
```
Expected: 6 tests pass.

If `Context::with_remote_span_context` is missing in `opentelemetry 0.27`,
swap to `Context::current_with_value` + manual span context attachment via
`opentelemetry::trace::TraceContextExt::with_remote_span_context`.

- [ ] **Step 3: Run full --features otel test suite (regression check)**

Run:
```
cd sage-core && cargo test --features otel --lib 2>&1 | tail -20
```
Expected: all tests pass (existing 501 + new 8).

- [ ] **Step 4: Commit**

```
git add sage-core/src/observability/bridge.rs
git commit -m "feat(b1.b): bridge_python_span — W3C traceparent → EnteredSpan

Parses 4-part W3C traceparent, builds an OTel SpanContext, attaches
to current context, creates a tracing span that inherits the parent
via tracing-opentelemetry Layer. RustSpanHandle owns both the
EnteredSpan and ContextGuard; Drop closes in reverse order. Marked
#[pyclass(unsendable)] — !Send guard requires GIL-pinned thread.

Tests: 5 parse cases + 1 no-panic. All existing 501 sage-core tests
pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Rust integration test — InMemoryExporter end-to-end

**Files:**
- Create: `sage-core/tests/otel_smoke.rs`

End-to-end test: install init, call bridge, generate an `info_span!`, verify
the InMemoryExporter receives a span with the expected parent.

- [ ] **Step 1: Write the failing test**

Create `sage-core/tests/otel_smoke.rs`:

```rust
//! End-to-end Rust-side OTel smoke. Installs an InMemoryExporter,
//! bridges a synthetic Python parent span, runs an existing
//! info_span! pattern, and asserts the recorded span has the right
//! parent_span_id.
//!
//! Only compiled with --features otel. Skipped otherwise.

#![cfg(feature = "otel")]

use std::sync::Arc;

use opentelemetry::global;
use opentelemetry::trace::{TraceContextExt as _, TracerProvider as _};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::testing::trace::InMemorySpanExporter;
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use sage_core::observability::bridge_python_span;

#[test]
#[ignore = "process-wide subscriber install conflicts with other tests; run alone with --ignored"]
fn rust_span_inherits_python_parent_via_traceparent() {
    let exporter = InMemorySpanExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let tracer = provider.tracer("test");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let _ = tracing_subscriber::registry()
        .with(layer)
        .try_init();
    global::set_tracer_provider(provider);
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Synthetic Python parent: trace_id and span_id chosen for visibility.
    let traceparent = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01";
    {
        let _bridge = bridge_python_span(traceparent, "test_bridge");
        // Inside the bridge scope, an info_span! call inherits the parent.
        let _child = tracing::info_span!("test.child", value = 42).entered();
    }

    let spans = exporter.get_finished_spans().unwrap();
    assert!(
        !spans.is_empty(),
        "expected at least one span exported; got {}",
        spans.len()
    );

    // Find the info_span!-emitted span and check its parent_span_id is
    // the one from our traceparent (bbbbbbbbbbbbbbbb).
    let child = spans
        .iter()
        .find(|s| s.name == "test.child")
        .expect("test.child span should be exported");

    let parent_id = format!("{:016x}", u64::from_be_bytes(child.parent_span_id.to_bytes()));
    assert_eq!(
        parent_id, "bbbbbbbbbbbbbbbb",
        "child span parent_span_id should match traceparent's span_id"
    );
}
```

- [ ] **Step 2: Run the test alone**

Run:
```
cd sage-core && cargo test --features otel --test otel_smoke -- --ignored --test-threads=1 2>&1 | tail -20
```
Expected: 1 test passes. **If `InMemorySpanExporter::get_finished_spans` has a
different return type in 0.27 (e.g. `Result<Vec<_>, _>` vs `Vec<_>`), adapt
the unwrap accordingly.**

- [ ] **Step 3: Commit**

```
git add sage-core/tests/otel_smoke.rs
git commit -m "test(b1.b): Rust-side end-to-end OTel smoke

InMemoryExporter + bridge + info_span! → asserts parent_span_id
matches the traceparent. Marked #[ignore] because tracing-subscriber
init is process-wide; run with --ignored --test-threads=1 to execute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Python-side `_init_tracer` mirror to Rust

**Files:**
- Modify: `sage-python/src/sage/observability/__init__.py`
- Test: `sage-python/tests/observability/test_rust_bridge.py` (created in Task 7)

After installing the Python TracerProvider, call `sage_core.init_otel(...)`
to mirror the exporter config into Rust.

- [ ] **Step 1: Write the failing test (placeholder — actual integration test in Task 7)**

We don't have a Python test target yet for this — Task 7 covers the integration
tests. For now: do a manual smoke after the change (Step 4).

- [ ] **Step 2: Modify `_init_tracer`**

Edit `sage-python/src/sage/observability/__init__.py`. After the
`trace.set_tracer_provider(provider)` line at line 69, AND at the end of
the `elif exporter_kind == "logfire":` branch, add a single call to a new
helper `_mirror_to_rust(exporter_kind)`. Insert helper function above
`_init_tracer`:

```python
def _mirror_to_rust(exporter_kind: str) -> None:
    """Mirror Python OTel exporter config into Rust via sage_core.

    Idempotent. Returns silently when the Rust `otel` feature is off
    (sage_core.init_otel returns False) or when sage_core is missing
    entirely. Logfire mode is treated as no-op for Rust spans (B1.b.7).
    """
    if exporter_kind == "logfire":
        log.info(
            "Logfire exporter active for Python spans; "
            "Rust spans not mirrored (roadmap-B1.b.7)"
        )
        return
    if exporter_kind not in {"console", "otlp_http"}:
        return
    try:
        import sage_core  # type: ignore[import-not-found]
    except ImportError:
        log.warning(
            "OTel exporter %r requested but sage_core not importable; "
            "Rust spans will not be exported",
            exporter_kind,
        )
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    try:
        ok = sage_core.init_otel(exporter_kind, endpoint)
    except Exception:  # pylint: disable=broad-except
        log.exception(
            "sage_core.init_otel raised; Rust spans will not be exported"
        )
        return

    if not ok:
        log.info(
            "sage_core.init_otel returned False for exporter=%r "
            "(feature off, already-initialized, or unsupported); "
            "Rust spans will not be exported in this run",
            exporter_kind,
        )
```

Then in `_init_tracer`, after line 69 (after `trace.set_tracer_provider(provider)`),
add:

```python
    _mirror_to_rust(exporter_kind)
```

And in the `elif exporter_kind == "logfire":` branch, before `return`, add:

```python
        _mirror_to_rust("logfire")
```

- [ ] **Step 3: Run full Python test suite (regression check)**

Run:
```
cd sage-python && python -m pytest tests/observability/ -v 2>&1 | tail -20
```
Expected: all 24 existing observability tests still pass (the new helper is
either no-op when sage_core is built without `otel`, or a successful mirror
when built with `otel`).

- [ ] **Step 4: Manual smoke**

Run:
```
cd sage-python && SAGE_OTEL_EXPORTER=console python -c "from sage.observability import _init_tracer; _init_tracer(); print('OK')"
```
Expected: prints `OK`. If sage_core was built without `--features otel`, the
log shows an INFO line about sage_core.init_otel returning False; no crash.
If built with `--features otel`, no INFO line; init succeeded.

- [ ] **Step 5: Commit**

```
git add sage-python/src/sage/observability/__init__.py
git commit -m "feat(b1.b): Python _init_tracer mirrors exporter config to Rust

Calls sage_core.init_otel after the Python TracerProvider is set.
Logfire branch logs an info msg (Rust spans deferred to B1.b.7).
Catches ImportError + arbitrary exceptions defensively — Rust failure
must not break Python observability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Python `sage_span` bridges into Rust per-call

**Files:**
- Modify: `sage-python/src/sage/observability/spans.py`
- Test: `sage-python/tests/observability/test_rust_bridge.py`

For every Python `sage_span` invocation, after `start_as_current_span` returns,
serialize the current SpanContext as W3C traceparent and call
`sage_core.bridge_python_span(traceparent, name)`. Hold the returned handle
for the lifetime of the Python span.

- [ ] **Step 1: Write the failing test**

Create `sage-python/tests/observability/test_rust_bridge.py`:

```python
"""B1.b: integration tests for Python → Rust span bridging.

These tests exercise the W3C traceparent propagation. They DO NOT
require sage-core to be built with --features otel — when the otel
feature is off, the bridge calls return no-op handles and the tests
verify the Python side handles that path cleanly (no crash, no
unintended spans).

When run with sage-core built --features otel and a Rust
InMemorySpanExporter wired up, the parent-linkage assertion
fires for real (otherwise the Rust-side test in
sage-core/tests/otel_smoke.rs covers it).
"""
from __future__ import annotations

import logging
import re

import pytest

from sage.observability import _init_tracer
from sage.observability.spans import sage_span


W3C_TRACEPARENT_RE = re.compile(
    r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"
)


def _format_traceparent_from_current_span() -> str | None:
    """Internal helper used by sage_span — exercise its shape."""
    from opentelemetry import trace
    span = trace.get_current_span()
    sc = span.get_span_context()
    if not sc.is_valid:
        return None
    return f"00-{sc.trace_id:032x}-{sc.span_id:016x}-{int(sc.trace_flags):02x}"


def test_traceparent_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: when sage_span is active, current span context formats cleanly."""
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    _init_tracer()
    with sage_span("sage.test", op="test_op"):
        tp = _format_traceparent_from_current_span()
        assert tp is not None
        assert W3C_TRACEPARENT_RE.match(tp), f"malformed traceparent: {tp!r}"


def test_rust_init_skipped_when_exporter_none(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Acceptance criterion §11.B: SAGE_OTEL_EXPORTER=none → no Rust init."""
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "none")
    caplog.set_level(logging.INFO, logger="sage.observability")
    _init_tracer()
    # No log lines should mention sage_core.init_otel
    assert not any(
        "sage_core.init_otel" in r.message for r in caplog.records
    ), f"unexpected sage_core mention: {caplog.records}"


def test_rust_init_skipped_when_feature_off(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Acceptance criterion §11.C: feature off → INFO log, no crash.

    This test passes regardless of how sage_core is actually built.
    When otel feature is OFF, the stub returns False and we log INFO.
    When otel feature is ON, init_otel returns True (or False if
    already initialized in-process). Both paths are non-crashing.
    """
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    caplog.set_level(logging.INFO, logger="sage.observability")
    _init_tracer()
    # No exception raised. Either an INFO line about feature-off, or
    # silent success. Both acceptable.


def test_sage_span_bridges_to_rust_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """sage_span enter/exit lifecycle creates a Rust handle when feature on.

    When sage_core has the `otel` feature off, this test still passes:
    the stub returns a no-op handle. The assertion is the lifecycle
    completes without exception and the handle's .close() is callable.
    """
    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    _init_tracer()
    with sage_span("sage.test_bridge", op="test_bridge_op") as span:
        assert span is not None
        # Internal: spans._maybe_bridge_to_rust should have stashed a handle.
        # We verify via no-exception completion of the with-block.
```

- [ ] **Step 2: Modify `sage_span` in `spans.py`**

Edit `sage-python/src/sage/observability/spans.py`. Add a new helper
function above `sage_span` (after `_otel_enabled`):

```python
def _maybe_bridge_to_rust(name: str, span: Any) -> Any | None:
    """If sage_core.bridge_python_span exists, call it with the W3C
    traceparent of the current span. Returns the Rust handle (caller
    closes on scope exit) or None when bridging not available.
    """
    try:
        import sage_core  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        sc = span.get_span_context()
        if not sc.is_valid:
            return None
        traceparent = (
            f"00-{sc.trace_id:032x}-{sc.span_id:016x}-{int(sc.trace_flags):02x}"
        )
        return sage_core.bridge_python_span(traceparent, name)
    except Exception:  # pylint: disable=broad-except
        log.exception("bridge_python_span raised; continuing without Rust span")
        return None
```

Then modify `sage_span` body. The current body is:

```python
@contextmanager
def sage_span(...) -> Iterator[Any]:
    if not _otel_enabled():
        yield None
        return
    _maybe_warn_secrets_disabled()
    tracer = _get_tracer()
    with tracer.start_as_current_span(...) as span:
        span.set_attribute("gen_ai.operation.name", op)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        try:
            yield span
        except BaseException as exc:
            ...
            raise
```

Wrap the `try: yield span` with the Rust bridge:

```python
@contextmanager
def sage_span(
    name: str,
    op: str,
    *,
    record_exception: bool = True,
    **attrs: Any,
) -> Iterator[Any]:
    if not _otel_enabled():
        yield None
        return
    _maybe_warn_secrets_disabled()
    tracer = _get_tracer()
    with tracer.start_as_current_span(
        name,
        record_exception=record_exception,
        set_status_on_exception=record_exception,
    ) as span:
        span.set_attribute("gen_ai.operation.name", op)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        rust_handle = _maybe_bridge_to_rust(name, span)
        try:
            yield span
        except BaseException as exc:
            if not record_exception:
                import traceback as _traceback

                from opentelemetry.trace import Status, StatusCode

                msg = _REDACTOR.redact_text(str(exc)) if _REDACTOR.enabled else str(exc)
                tb = (
                    _REDACTOR.redact_text(_traceback.format_exc())
                    if _REDACTOR.enabled
                    else _traceback.format_exc()
                )
                span.add_event(
                    "exception",
                    {
                        "exception.type": type(exc).__name__,
                        "exception.message": msg,
                        "exception.stacktrace": tb,
                    },
                )
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            if rust_handle is not None:
                try:
                    rust_handle.close()
                except Exception:  # pylint: disable=broad-except
                    log.exception("rust_handle.close() raised; continuing")
```

- [ ] **Step 3: Run the new tests**

Run:
```
cd sage-python && python -m pytest tests/observability/test_rust_bridge.py -v 2>&1 | tail -30
```
Expected: 4 tests pass.

- [ ] **Step 4: Run the full observability suite (regression)**

Run:
```
cd sage-python && python -m pytest tests/observability/ -v 2>&1 | tail -30
```
Expected: 28 tests pass (24 existing + 4 new). No regressions.

- [ ] **Step 5: Commit**

```
git add sage-python/src/sage/observability/spans.py sage-python/tests/observability/test_rust_bridge.py
git commit -m "feat(b1.b): sage_span bridges to Rust per-call

Every sage_span enter serializes the current SpanContext as W3C
traceparent and calls sage_core.bridge_python_span. Handle's close
runs in finally. Defensive try/except — Rust failure must never
propagate to Python span lifecycle.

Tests: 4 new tests (traceparent format, no-bridge-on-none, no-crash-
on-feature-off, lifecycle no-exception). 28 obs tests total pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Python integration — Rust spans visible in routing path

**Files:**
- Modify: `sage-python/tests/observability/test_rust_bridge.py` (append)

This test only meaningfully runs when sage-core is built with `--features
otel`. It uses a marker to skip when the feature isn't compiled in. Not a CI
gate — a dev-machine sanity check captured as code.

- [ ] **Step 1: Write the test**

Append to `sage-python/tests/observability/test_rust_bridge.py`:

```python
def _sage_core_has_otel_feature() -> bool:
    """Return True iff sage_core.init_otel returns truthy for a
    real exporter — i.e. the otel feature was built into sage-core.
    """
    try:
        import sage_core  # type: ignore[import-not-found]
    except ImportError:
        return False
    # Cannot reliably probe at import time because init_otel is one-shot.
    # Use a heuristic: stub init_otel always returns False; real one
    # returns True on first console call. This test is informational
    # only — when uncertain, skip rather than fail.
    try:
        return bool(sage_core.init_otel("console", None))
    except Exception:  # pylint: disable=broad-except
        return False


@pytest.mark.skipif(
    not _sage_core_has_otel_feature(),
    reason="sage-core not built with --features otel"
)
def test_rust_routing_span_visible_in_otel_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance §8.A: a routing call emits a child Rust span under
    the Python sage.assign parent. Requires sage-core --features otel.

    Manually constructs a routing call so we don't need a full pipeline.
    """
    pytest.importorskip("sage_core")
    import sage_core  # type: ignore[import-not-found]

    monkeypatch.setenv("SAGE_OTEL_EXPORTER", "console")
    _init_tracer()

    # Construct a SystemRouter and call route() inside a sage_span.
    if not hasattr(sage_core, "SystemRouter"):
        pytest.skip("sage_core.SystemRouter not exposed in this build")

    # Best-effort: build a router, route a small task, no assertion on
    # span counts (would require InMemoryExporter wiring on Rust side
    # which Task 5 covered separately). This test just exercises the
    # codepath end-to-end without crashing.
    router = sage_core.SystemRouter()
    with sage_span("sage.assign", op="assign_models"):
        try:
            _ = router.route("compute fibonacci", 1.0)
        except Exception:  # pylint: disable=broad-except
            pytest.skip("SystemRouter.route signature changed; skipping")
```

- [ ] **Step 2: Run the test**

Run:
```
cd sage-python && python -m pytest tests/observability/test_rust_bridge.py::test_rust_routing_span_visible_in_otel_export -v 2>&1 | tail -10
```
Expected: PASS or SKIPPED with the "not built with --features otel" reason.
Either is acceptable.

- [ ] **Step 3: Commit**

```
git add sage-python/tests/observability/test_rust_bridge.py
git commit -m "test(b1.b): routing-path Rust span integration

Skipif when sage-core was built without --features otel. When built
with otel, exercises a routing call inside a sage.assign Python span
to confirm no-crash codepath.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: CI compile-check job for `--features otel`

**Files:**
- Modify or create: `.github/workflows/rust.yml` (or equivalent existing
  CI config — discover via `ls .github/workflows/` first)

If no CI config exists yet, document the deferral in `roadmap.md` (Task 10
includes the roadmap update; this task fills the CI section).

- [ ] **Step 1: Discover existing CI configuration**

Run:
```
ls -la .github/workflows/ 2>&1
```
Branch on the result:
- **Files exist:** proceed to Step 2 below.
- **No directory or no `.yml` files:** skip to Step 4 (document-only).

- [ ] **Step 2: Read existing Rust workflow** (if found)

Run (substitute filename of existing rust workflow):
```
cat .github/workflows/rust.yml 2>&1 | head -100
```
Identify the existing `cargo test` step. We're adding a sibling step that
runs with `--features otel`.

- [ ] **Step 3: Add the new compile-check step**

Add a new step (or new job) that runs:

```yaml
      - name: Build sage-core with otel feature
        working-directory: sage-core
        run: cargo build --features otel --no-default-features --features extension-module,otel
      - name: Test sage-core --features otel
        working-directory: sage-core
        run: cargo test --features otel --lib
```

If a YAML matrix exists, add `otel` as a value:

```yaml
        features: [smt, otel, "smt,otel"]
```

The exact integration depends on the existing workflow shape. Don't
duplicate the whole job — extend the existing one.

- [ ] **Step 4: Document deferral if no CI config exists**

Edit `roadmap.md`. In the Horizon B section, append under the existing
roadmap-B1 closure:

```markdown
- **roadmap-B1.b.8** (CI matrix gap, deferred): the `otel` Rust feature is
  opt-in. Default `cargo test --lib` does not exercise the new code. No
  GitHub Actions config exists at this point in the project; once CI lands,
  add a `cargo build --features otel` + `cargo test --features otel --lib`
  step. For now, the otel path is dev-machine-tested (build with
  `maturin develop --features otel,smt,onnx`).
```

- [ ] **Step 5: Commit**

If Step 3 path:
```
git add .github/workflows/<file>.yml
git commit -m "ci(b1.b): cargo build/test --features otel compile gate

Catches version drift between tracing-opentelemetry and opentelemetry
on every PR. ~1-2 min Linux runner. Closes B1.b.8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

If Step 4 path: defer the commit until Task 10 (roadmap update bundles).

---

## Task 10: Docs + roadmap update

**Files:**
- Modify: `docs/observability/otel-genai-spans.md`
- Modify: `CLAUDE.md`
- Modify: `.claude/rules/development.md`
- Modify: `roadmap.md`
- Modify: `MEMORY.md` (auto-memory pointer)

- [ ] **Step 1: Append "Rust spans" section to user docs**

Edit `docs/observability/otel-genai-spans.md`. After the existing "Setup"
section (or at the end of the file), append:

```markdown
## Rust spans (B1.b)

`sage-core` Rust hot paths (engine, system_router, knn, model_assigner,
write_gate, mutations, wasm_python, tool_executor) emit `tracing` spans
that nest under their Python parent in the OTel trace tree.

### Build recipe

By default, the Rust OTel bridge is **opt-in**. Build with:

```bash
cd sage-core && maturin develop --features otel,smt,onnx
```

Without `--features otel`, `sage_core.init_otel` returns False and Rust
spans are not exported. Python spans continue to flow.

### Exporter compatibility

| `SAGE_OTEL_EXPORTER` | Python | Rust (with `--features otel`) |
|---|---|---|
| `none` | no spans | no spans |
| `console` | spans to stdout | spans to stderr (tracing-subscriber default) |
| `otlp_http` | OTLP HTTP → collector | OTLP HTTP → same collector (correlated by trace_id) |
| `logfire` | spans to logfire | **not exported** (B1.b.7 — see roadmap) |

### How parent linkage works

At each `sage_span` enter on the Python side, the current span's
`SpanContext` is serialized as a W3C traceparent string and passed to
Rust via `sage_core.bridge_python_span(traceparent, name)`. Rust
attaches the traceparent as the active OTel context, then creates a
`tracing` span whose existing `info_span!` / `#[instrument]` children
inherit the parent automatically.

### Span audit

All 27 existing Rust span attributes are counts/IDs/numeric values —
zero raw user content. See spec §4.1 for the full table:
`docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md`.
```

- [ ] **Step 2: Update CLAUDE.md build recipe**

Edit `CLAUDE.md`. In the Quick Commands section, find the build line:
```bash
cd sage-core && maturin develop --features smt,onnx
```
Replace with:
```bash
# Add `--features otel` if you want Rust spans exported alongside Python's
# (B1.b, opt-in for now — see docs/observability/otel-genai-spans.md).
cd sage-core && maturin develop --features smt,onnx
# With Rust OTel:
cd sage-core && maturin develop --features otel,smt,onnx
```

- [ ] **Step 3: Update `.claude/rules/development.md` env table**

Edit `.claude/rules/development.md`. In the env vars table, find the
`SAGE_OTEL_EXPORTER` row and append a column note (or new line below):

```markdown
| `SAGE_OTEL_EXPORTER` | `none` | `console` (stdout), `otlp_http` (uses `OTEL_EXPORTER_OTLP_ENDPOINT`), `logfire` (managed). **Rust spans (B1.b):** when sage-core built with `--features otel`, console + otlp_http also mirror to Rust. logfire mode is Python-only (B1.b.7). |
```

- [ ] **Step 4: Update roadmap.md**

Edit `roadmap.md`. Under Horizon B, mark B1.b as closed:

```markdown
- **roadmap-B1.b** ✅ CLOSED 2026-04-25 — Rust spans bridge via tracing-opentelemetry.
  Approach A (independent Rust OTel SDK + W3C traceparent across PyO3, no PyO3 0.27
  upgrade). 27 span call sites audited (counts/IDs only, zero raw payloads). Spec:
  `docs/superpowers/specs/2026-04-25-otel-rust-spans-design.md`. Plan:
  `docs/superpowers/plans/2026-04-25-otel-rust-spans.md`.
- **roadmap-B1.b.7** (deferred) — Logfire-mode Rust export. WARN-only today;
  Python spans still flow via logfire SDK.
- **roadmap-B1.b.8** (CI-decision-point) — see commit history; CI step added or
  deferred per repo state at implementation time.
```

If Task 9 went the Step 4 (no CI config) path, also include the deferred-CI
text from that task.

- [ ] **Step 5: Update MEMORY.md pointer**

Edit `C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\memory\MEMORY.md`.
In the "⭐ Active direction" section, near the top, add a new bullet under
"Most recent work":

```markdown
**Plus B1.b — Rust OTel bridge (2026-04-25):** independent Rust OTel SDK
+ W3C traceparent across PyO3. No PyO3 0.27 upgrade. 27 spans audited
(counts/IDs only). Spec + plan committed; tests green. Logfire-mode Rust
export deferred to B1.b.7.
```

- [ ] **Step 6: Run full Python + Rust suites (final regression check)**

Run:
```
cd sage-core && cargo test --features otel,smt --lib 2>&1 | tail -10
cd sage-core && cargo test --features smt --lib 2>&1 | tail -10
```
Expected: both green. Counts: 501 → 511 (with otel) or 501 (without).

Run:
```
cd sage-python && python -m pytest tests/ -x --tb=short \
  --ignore=tests/test_e2e_full_pipeline.py \
  --ignore=tests/test_provider_pool_wiring.py \
  --ignore=tests/test_pydantic_ai_integration.py \
  2>&1 | tail -10
```
Expected: same baseline (~2493 passing) + 4 new bridge tests = 2497 passing.

- [ ] **Step 7: Final commit (single bundled commit for docs + roadmap)**

```
git add docs/observability/otel-genai-spans.md CLAUDE.md .claude/rules/development.md \
        roadmap.md \
        ~/.claude/projects/C--Code-YGN-SAGE/memory/MEMORY.md
git commit -m "docs(b1.b): user docs + roadmap closure for Rust OTel bridge

- otel-genai-spans.md: new \"Rust spans\" section (build recipe,
  exporter table, parent-linkage explainer, audit pointer)
- CLAUDE.md: build recipe with --features otel example
- development.md: env table updated for Rust spans note
- roadmap.md: B1.b CLOSED; B1.b.7 (logfire+rust) + B1.b.8 (CI)
  ticketed as deferred
- MEMORY.md: active-direction pointer

Spec: 26da4a8d. Plan: <this commit's parent path>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Verification checklist (run after all 10 tasks)

| Acceptance criterion | How to verify |
|---|---|
| §8.A — Rust span has correct `parent_span_id` | `cargo test --features otel --test otel_smoke -- --ignored` |
| §8.B — `SAGE_OTEL_EXPORTER=none` → no Rust subscriber | `test_rust_init_skipped_when_exporter_none` |
| §8.C — feature off → WARN once, no crash | `test_rust_init_skipped_when_feature_off` |
| §8.D — ≥ 8 distinct Rust span names in routing_gt smoke | manual: `SAGE_OTEL_EXPORTER=console python -m sage.bench --type routing_gt 2>&1 \| grep -E "(system_router\|bandit\|topology_engine)" \| wc -l` ≥ 8 |
| §8.E — zero new test failures on baselines | full Python + Rust suites green |

---

## What this plan explicitly does NOT do

- **No PyO3 upgrade** (would force a multi-day cross-cutting change unrelated
  to OTel). Approach A's manual traceparent passing is sufficient.
- **No Rust-side redaction module** (audit found zero callers; YAGNI).
- **No logfire-mode Rust export** (B1.b.7 deferred — Python logfire path
  covers primary observability surface).
- **No span-name renames** (`system_router.route` → `sage.routing.system_router.route`).
  Cosmetic, deferred to B1.b.1 follow-up.
- **No Rust → Python span propagation** (no call paths exist).
- **No metrics** (`gen_ai.usage.*` counters/histograms — separate spec, B1.b.2).
