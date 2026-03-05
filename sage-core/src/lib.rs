use pyo3::prelude::*;

pub mod types;
pub mod agent;
pub mod pool;
pub mod memory;
pub mod hardware;
pub mod sandbox;
pub mod simd_sort;

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
    m.add_class::<hardware::HardwareProfile>()?;
    m.add_class::<sandbox::wasm::WasmSandbox>()?;
    m.add_class::<sandbox::ebpf::EbpfSandbox>()?;
    m.add_class::<sandbox::ebpf::SnapBPF>()?;

    // Add SIMD functions
    m.add_function(wrap_pyfunction!(simd_sort::vectorized_partition_h96, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_quicksort, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_quicksort_zerocopy, m)?)?;
    m.add_function(wrap_pyfunction!(simd_sort::h96_argsort, m)?)?;

    Ok(())
}
