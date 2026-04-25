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
        let mut h = bridge_python_span("00-1-2-01", "test");
        h.close();  // does not panic
    }
}
