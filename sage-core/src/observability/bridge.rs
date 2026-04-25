//! Real `bridge_python_span` + `RustSpanHandle`. Filled in in Task 4.
//! For now, returns stub-shaped values so init.rs compiles.

use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (traceparent, name))]
pub fn bridge_python_span(traceparent: &str, name: &str) -> RustSpanHandle {
    let _ = (traceparent, name);
    RustSpanHandle {}
}

#[pyclass(unsendable)]
pub struct RustSpanHandle {}

#[pymethods]
impl RustSpanHandle {
    /// Placeholder for Task 4. Final signature: `&mut self`.
    pub fn close(&mut self) {}
}
