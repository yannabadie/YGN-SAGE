//! End-to-end Rust-side OTel smoke. Installs an InMemoryExporter,
//! bridges a synthetic Python parent span, runs an `info_span!` inside
//! the bridge scope, and asserts that the bridged span's
//! `parent_span_id` matches the traceparent's span_id (proving Python
//! → Rust parent linkage works) and that the inner child shares the
//! bridged trace_id (proving downstream Rust spans inherit the trace
//! context).
//!
//! Only compiled with `--features otel`. Skipped otherwise.
//!
//! `#[ignore]` because `tracing_subscriber::registry().try_init()`
//! installs a process-wide subscriber. Other tests under
//! `--features otel` install their own subscribers (or rely on none),
//! so this test must run alone:
//!     cargo test --features otel --test otel_smoke -- --ignored --test-threads=1
//!
//! The `testing` feature on `opentelemetry_sdk` (required for
//! `InMemorySpanExporter`) is wired as a `dev-dependencies` entry in
//! sage-core/Cargo.toml — it drags `tokio/rt-multi-thread` and other
//! runtime deps that we deliberately keep out of the production
//! `otel` feature surface.

#![cfg(feature = "otel")]

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
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
    let _ = tracing_subscriber::registry().with(layer).try_init();
    global::set_tracer_provider(provider);
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Synthetic Python parent: trace_id and span_id chosen for visibility.
    // trace_id = aaaa...aaaa (32 hex), span_id = bbbb...bbbb (16 hex).
    let traceparent = "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01";
    {
        let _bridge = bridge_python_span(traceparent, "test_bridge");
        // Inside the bridge scope, an info_span! call inherits the
        // *bridged* tracing span as its parent. We emit one to prove
        // child Rust spans share the trace_id with the bridged root.
        let _child = tracing::info_span!("test.child", value = 42).entered();
    }

    let spans = exporter.get_finished_spans().unwrap();
    assert!(
        !spans.is_empty(),
        "expected at least one span exported; got {}",
        spans.len()
    );

    // The PRIMARY assertion: `rust_bridged_span` (the span emitted by
    // bridge_python_span itself, per bridge.rs:80) has parent_span_id
    // equal to the traceparent's span_id. This is the actual Python →
    // Rust parent-linkage proof.
    let bridged = spans
        .iter()
        .find(|s| s.name == "rust_bridged_span")
        .expect("rust_bridged_span should be exported");
    let bridged_parent_id = format!(
        "{:016x}",
        u64::from_be_bytes(bridged.parent_span_id.to_bytes())
    );
    assert_eq!(
        bridged_parent_id, "bbbbbbbbbbbbbbbb",
        "rust_bridged_span parent_span_id should match traceparent's span_id"
    );

    // SECONDARY assertion: the inner `test.child` span shares the
    // bridged trace_id. tracing-opentelemetry's contextual parent
    // lookup makes test.child.parent = rust_bridged_span (auto-
    // generated id, so we can't predict it), but trace_id propagates.
    let child = spans
        .iter()
        .find(|s| s.name == "test.child")
        .expect("test.child span should be exported");
    let child_trace_id = format!(
        "{:032x}",
        u128::from_be_bytes(child.span_context.trace_id().to_bytes())
    );
    assert_eq!(
        child_trace_id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "test.child should inherit the bridged trace_id"
    );
}
