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
use tracing::span::EnteredSpan;
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
            Context::new() // empty context; tracing span has no parent
        }
    };

    // Attach the OTel context. The guard keeps it active.
    let cx_guard = parent_cx.attach();

    // Create a tracing span. Use a static span name; attach the dynamic
    // bridge name as a field. Macros require static strings, so the
    // span_name "rust_bridged_span" is intentionally stable.
    let span = tracing::info_span!("rust_bridged_span", bridge_name = name);
    // Force the parent linkage — set_parent picks up the current OTel
    // context that we just attached. set_parent borrows &self, so this
    // MUST run before span.entered() (which consumes Span by value).
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

    let trace_id =
        u128::from_str_radix(parts[1], 16).map_err(|_| "trace_id is not valid hex")?;
    let span_id = u64::from_str_radix(parts[2], 16).map_err(|_| "span_id is not valid hex")?;
    let flags = u8::from_str_radix(parts[3], 16).map_err(|_| "flags is not valid hex")?;

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
        true, // remote
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
