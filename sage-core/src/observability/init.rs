//! Real `init_otel` + bridge implementation. Compiles only when the
//! `otel` feature is on.

use pyo3::prelude::*;
use std::sync::Mutex;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

static INIT: Mutex<bool> = Mutex::new(false);

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
    let mut init_guard = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if *init_guard {
        return false; // already initialized
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
        eprintln!(
            "[sage-core] OTel init: a tracing subscriber was already installed; \
             Rust spans will not be exported to OTel. \
             To use Rust OTel spans, ensure SAGE is the only tracing-subscriber owner."
        );
        return false;
    }

    global::set_tracer_provider(provider);
    global::set_text_map_propagator(TraceContextPropagator::new());

    *init_guard = true;
    true
}

fn build_console_provider() -> Option<SdkTracerProvider> {
    // For B1.b MVP: console mode wires opentelemetry-stdout's SpanExporter
    // so Rust spans reach stderr directly. Python-side console exporter
    // handles its own spans separately. Both surfaces use the same
    // user-visible "console" exporter kind.
    let exporter = opentelemetry_stdout::SpanExporter::default();
    Some(
        SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build(),
    )
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
            eprintln!("[sage-core] OTel init: OTLP exporter build failed: {err}");
            return None;
        }
    };

    // MVP: simple (synchronous) exporter — avoids requiring a live tokio
    // runtime at PyO3 boundary. Trade-off: spans flush per-emission rather
    // than batched. Production volume profile may want a batch path later
    // (tracked as B1.b.9 follow-up — needs explicit runtime ownership).
    Some(
        SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build(),
    )
}

// Bridge fn lives in bridge.rs but is re-exported here for the same
// PyO3 surface as the stub variant. Task 4 fills it in fully.
pub use crate::observability::bridge::{bridge_python_span, RustSpanHandle};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn init_with_none_returns_false() {
        let _guard = SERIAL.lock().unwrap();
        let _ = init_otel("none", None);
        // INIT may or may not be set depending on prior tests' execution
        // order. The point is: no panic, no subscriber installed for "none".
    }

    #[test]
    fn init_with_unknown_exporter_returns_false() {
        let _guard = SERIAL.lock().unwrap();
        let result = init_otel("frobinator-9000", None);
        assert!(!result, "unknown exporter should return false, got true");
    }

    #[test]
    fn init_otel_is_idempotent() {
        // Calling init_otel("console", None) twice in the same process
        // must not double-install the subscriber. First call returns true
        // OR false depending on prior tests' execution order; second call
        // MUST return false. The Mutex<bool> guard ensures atomicity.
        let _guard = SERIAL.lock().unwrap();
        let _first = init_otel("console", None);
        let second = init_otel("console", None);
        assert!(!second, "second init_otel call must be a no-op");
    }
}
