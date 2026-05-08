use pyo3::prelude::*;

pub mod agent;
pub mod hardware;
pub mod memory;
pub mod observability;
pub mod pool;
pub mod routing;
#[cfg(any(feature = "sandbox", feature = "tool-executor"))]
pub mod sandbox;
pub mod sort_utils;
pub mod topology;
pub mod types;
pub mod verification;

#[pymodule]
fn sage_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Cycle-13 B Q1 (cgpro post-push 2026-05-06 NEXT_BLOCK_ID=G):
    // build-info exposure for Python-side stale-binary detection.
    // The 2026-04-27 -> 2026-04-30 manifest-write gap (closed by
    // commit `32d39bdf` regression test) wasted operator cycles
    // because the installed `.pyd` predated the source fix and
    // there was no way to know without running the test. With
    // these 4 attributes, `python -m sage.ops.sage_core_version`
    // can compare the installed binary against `git rev-parse HEAD`
    // and exit 1 if drift is detected.
    //
    // `__commit_sha__`: set by build.rs from `git rev-parse HEAD`
    //   in the sage-core manifest dir, OR overridden by
    //   `SAGE_CORE_COMMIT_SHA_OVERRIDE` env var (CI / PyPI wheel
    //   builds), OR "unknown" when git is absent.
    // `__build_timestamp__`: UNIX seconds at compile time.
    // `__build_profile__`: cargo profile (`debug` / `release`).
    // `__version__`: matches `Cargo.toml [package].version`.
    m.add("__commit_sha__", env!("SAGE_CORE_BUILD_COMMIT_SHA"))?;
    m.add("__build_timestamp__", env!("SAGE_CORE_BUILD_TIMESTAMP"))?;
    m.add("__build_profile__", env!("SAGE_CORE_BUILD_PROFILE"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<types::AgentConfig>()?;
    m.add_class::<types::ToolSpec>()?;
    m.add_class::<types::MemoryScope>()?;
    m.add_class::<types::AgentStatus>()?;
    m.add_class::<types::TopologyRole>()?;
    m.add_class::<pool::AgentPool>()?;
    m.add_class::<memory::MemoryEvent>()?;
    m.add_class::<memory::WorkingMemory>()?;
    m.add_class::<memory::smmu::PyMultiViewMMU>()?;
    m.add_class::<hardware::HardwareProfile>()?;
    #[cfg(feature = "sandbox")]
    {
        m.add_class::<sandbox::wasm::WasmSandbox>()?;
        // ebpf disabled until solana_rbpf is re-added (Phase 2)
        // m.add_class::<sandbox::ebpf::EbpfSandbox>()?;
        // m.add_class::<sandbox::ebpf::SnapBPF>()?;
    }
    #[cfg(feature = "tool-executor")]
    {
        m.add_class::<sandbox::validator::ValidationResult>()?;
        m.add_class::<sandbox::subprocess::ExecResult>()?;
        m.add_class::<sandbox::tool_executor::ToolExecutor>()?;
    }
    // B1.b OTel bridge — always exposed (stub when otel feature off).
    m.add_function(pyo3::wrap_pyfunction!(observability::init_otel, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        observability::bridge_python_span,
        m
    )?)?;
    m.add_class::<observability::RustSpanHandle>()?;
    // Embedded-RustPython availability probe. Red-team harness uses
    // this to skip tests when sage-core was built without the wasm
    // bytes bundled. `has_wasm()` on ToolExecutor only answers for
    // the Component-Model path — this covers the execute_raw path.
    m.add_function(pyo3::wrap_pyfunction!(embedded_wasm_available, m)?)?;
    m.add_class::<memory::rag_cache::RagCache>()?;
    m.add_class::<memory::relevance_gate::RustRelevanceGate>()?;
    m.add_class::<memory::write_gate::RustCompositeWriteGate>()?;
    m.add_class::<memory::write_gate::WriteGateDecision>()?;
    m.add_class::<memory::entity_graph::RustEntityGraph>()?;
    m.add_class::<routing::features::StructuralFeatures>()?;
    m.add_class::<routing::knn::RustKnnRouter>()?;
    m.add_class::<routing::quality::RustQualityEstimator>()?;
    m.add_class::<routing::model_assigner::ModelAssigner>()?;
    m.add_class::<routing::model_assigner::PyModelCandidateTrace>()?;
    m.add_class::<routing::model_card::ModelCard>()?;
    m.add_class::<routing::model_card::CognitiveSystem>()?;
    m.add_class::<routing::model_registry::ModelRegistry>()?;
    m.add_class::<routing::system_router::SystemRouter>()?;
    m.add_class::<routing::system_router::RoutingDecision>()?;
    m.add_class::<routing::system_router::RoutingConstraints>()?;
    m.add_class::<routing::bandit::BanditDecision>()?;
    m.add_class::<routing::bandit::ContextualBandit>()?;
    m.add_class::<topology::TopologyGraph>()?;
    // Standalone functions for TopologyGraph (workaround for PyO3 inventory
    // issue on Windows where new #[pymethods] don't register properly)
    m.add_function(pyo3::wrap_pyfunction!(
        topology::topology_graph::graph_get_predecessors,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        topology::topology_graph::graph_get_edges,
        m
    )?)?;
    m.add_class::<topology::TopologyNode>()?;
    m.add_class::<topology::TopologyEdge>()?;
    m.add_class::<topology::templates::PyTemplateStore>()?;
    m.add_class::<topology::verifier::PyHybridVerifier>()?;
    m.add_class::<topology::verifier::VerificationResult>()?;
    m.add_class::<topology::pyo3_wrappers::PyTopologyEngine>()?;
    m.add_class::<topology::pyo3_wrappers::PyGenerateResult>()?;
    m.add_class::<topology::pyo3_wrappers::PyTopologyExecutor>()?;
    m.add_class::<topology::pyo3_wrappers::PyMutationStats>()?;
    m.add_class::<topology::density::TopologyDensity>()?;
    m.add_class::<topology::density::DensityScore>()?;
    m.add_class::<topology::reward::TopologyReward>()?;
    m.add_class::<topology::reward::RewardScore>()?;
    // Plan 2.1 (2026-04-20): Rust TopologyController port — scaffold
    // with full state, decision paths populated in 2.2–2.6.
    m.add_class::<topology::controller::RustTopologyController>()?;
    m.add_class::<topology::controller::RustAdaptationDecision>()?;
    #[cfg(feature = "onnx")]
    {
        m.add_class::<memory::embedder::RustEmbedder>()?;
    }
    // Graph property checking (always available)
    m.add_class::<verification::ltl::LtlResult>()?;
    m.add_class::<verification::ltl::GraphPropertyChecker>()?;

    #[cfg(feature = "smt")]
    {
        m.add_class::<verification::SmtVerifier>()?;
        m.add_class::<verification::SmtVerificationResult>()?;
    }

    #[cfg(all(feature = "smt", feature = "tool-executor"))]
    {
        m.add_class::<verification::quality_labeler::QualityLabeler>()?;
        m.add_class::<verification::quality_labeler::QualityLabel>()?;
    }

    // Add sort utility functions
    m.add_function(wrap_pyfunction!(sort_utils::vectorized_partition_h96, m)?)?;
    m.add_function(wrap_pyfunction!(sort_utils::h96_quicksort, m)?)?;
    m.add_function(wrap_pyfunction!(sort_utils::h96_quicksort_zerocopy, m)?)?;
    m.add_function(wrap_pyfunction!(sort_utils::h96_argsort, m)?)?;

    Ok(())
}

/// True iff the embedded RustPython wasm runtime was bundled into
/// this build. The red-team harness uses this as its skip gate
/// because `ToolExecutor.has_wasm()` only answers for the
/// Component-Model path loaded via `load_precompiled_component()`,
/// not the `execute_raw`→embedded-RustPython path. Always exported
/// so Python callers get a stable import even on builds without the
/// sandbox feature.
#[pyfunction]
fn embedded_wasm_available() -> bool {
    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    {
        sandbox::wasm_python::WasmPythonExecutor::is_available()
    }
    #[cfg(not(all(feature = "sandbox", feature = "cranelift")))]
    {
        false
    }
}
