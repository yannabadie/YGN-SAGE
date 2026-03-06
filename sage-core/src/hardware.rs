use pyo3::prelude::*;
use raw_cpuid::CpuId;
use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

/// Represents the hardware profile of the host machine
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    #[pyo3(get)]
    pub total_memory_mb: u64,
    #[pyo3(get)]
    pub free_memory_mb: u64,
    #[pyo3(get)]
    pub physical_cores: usize,
    #[pyo3(get)]
    pub logical_cores: usize,
    #[pyo3(get)]
    pub cpu_brand: String,
    #[pyo3(get)]
    pub has_avx2: bool,
    #[pyo3(get)]
    pub has_avx512: bool,
    #[pyo3(get)]
    pub has_neon: bool,
}

#[pymethods]
impl HardwareProfile {
    /// Detects and returns the current hardware capabilities of the host
    #[staticmethod]
    pub fn detect() -> Self {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything()),
        );
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let total_memory_mb = sys.total_memory() / 1024 / 1024;
        let free_memory_mb = sys.available_memory() / 1024 / 1024;
        let physical_cores = sys.physical_core_count().unwrap_or(1);
        let logical_cores = sys.cpus().len();

        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let mut has_avx2 = false;
        let mut has_avx512 = false;
        let has_neon;

        // Check for x86 instructions via raw_cpuid
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let cpuid = CpuId::new();
            if let Some(_feature_info) = cpuid.get_feature_info() {
                has_avx2 = cpuid
                    .get_extended_feature_info()
                    .map_or(false, |ext| ext.has_avx2());
                has_avx512 = cpuid
                    .get_extended_feature_info()
                    .map_or(false, |ext| ext.has_avx512f());
            }
            has_neon = false;
        }

        // Check for ARM NEON
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON is standard on aarch64
            has_neon = true;
        }

        // Fallback for other architectures
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            has_neon = false;
        }

        HardwareProfile {
            total_memory_mb,
            free_memory_mb,
            physical_cores,
            logical_cores,
            cpu_brand,
            has_avx2,
            has_avx512,
            has_neon,
        }
    }

    #[getter]
    pub fn is_simd_capable(&self) -> bool {
        self.has_avx2 || self.has_avx512 || self.has_neon
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}
