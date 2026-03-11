use pyo3::prelude::*;

pub mod agent;
pub mod hardware;
pub mod memory;
pub mod pool;
pub mod routing;
#[cfg(any(feature = "sandbox", feature = "tool-executor"))]
pub mod sandbox;
pub mod simd_sort;
pub mod topology;
pub mod types;
#[cfg(feature = "smt")]
pub mod verification;

#[pymodule]
fn sage_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<memory::rag_cache::RagCache>()?;
    m.add_class::<routing::features::StructuralFeatures>()?;
    m.add_class::<routing::model_card::ModelCard>()?;
    m.add_class::<routing::model_card::CognitiveSystem>()?;
    m.add_class::<routing::model_registry::ModelRegistry>()?;
    m.add_class::<routing::system_router::SystemRouter>()?;
    m.add_class::<routing::system_router::RoutingDecision>()?;
    m.add_class::<routing::system_router::RoutingConstraints>()?;
    m.add_class::<routing::bandit::BanditDecision>()?;
    m.add_class::<routing::bandit::ContextualBandit>()?;
    m.add_class::<topology::TopologyGraph>()?;
    m.add_class::<topology::TopologyNode>()?;
    m.add_class::<topology::TopologyEdge>()?;
    m.add_class::<topology::templates::PyTemplateStore>()?;
    m.add_class::<topology::verifier::PyHybridVerifier>()?;
    m.add_class::<topology::verifier::VerificationResult>()?;
    m.add_class::<topology::pyo3_wrappers::PyTopologyEngine>()?;
    m.add_class::<topology::pyo3_wrappers::PyGenerateResult>()?;
    m.add_class::<topology::pyo3_wrappers::PyTopologyExecutor>()?;
    #[cfg(feature = "onnx")]
    {
        m.add_class::<memory::embedder::RustEmbedder>()?;
        m.add_class::<routing::AdaptiveRouter>()?;
        m.add_class::<routing::RoutingResult>()?;
    }
    #[cfg(feature = "smt")]
    {
        m.add_class::<verification::SmtVerifier>()?;
        m.add_class::<verification::SmtVerificationResult>()?;
    }

    // Add SIMD functions
    m.add_function(wrap_pyfunction!(simd_sort::vectorized_partition_h96, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_quicksort, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_quicksort_zerocopy, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_argsort, m)?)?;

    Ok(())
}
