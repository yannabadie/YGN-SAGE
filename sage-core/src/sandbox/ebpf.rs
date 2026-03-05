use dashmap::DashMap;
use pyo3::prelude::*;
use solana_rbpf::{
    elf::Executable,
    program::{BuiltinProgram, FunctionRegistry, SBPFVersion},
    vm::{Config, EbpfVm, TestContextObject},
    memory_region::MemoryMapping,
};
use std::sync::Arc;

#[pyclass]
pub struct EbpfSandbox {
    executable: Option<Arc<Executable<TestContextObject>>>,
    loader: Arc<BuiltinProgram<TestContextObject>>,
}

#[pymethods]
impl EbpfSandbox {
    #[new]
    pub fn new() -> Self {
        let config = Config::default();
        let loader = Arc::new(BuiltinProgram::new_loader(config, FunctionRegistry::default()));
        Self {
            executable: None,
            loader,
        }
    }

    /// Complete the real ELF loading logic using the solana_rbpf 0.8.5 API
    pub fn load_elf(&mut self, elf_bytes: &[u8]) -> PyResult<()> {
        let executable = Executable::<TestContextObject>::from_elf(
            elf_bytes,
            self.loader.clone(),
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ELF load error: {}", e)))?;
        
        self.executable = Some(Arc::new(executable));
        Ok(())
    }

    /// Load raw eBPF instruction bytes directly (useful for testing without a C compiler)
    pub fn load_raw(&mut self, raw_bytes: &[u8]) -> PyResult<()> {
        let sbpf_version = SBPFVersion::V2;
        let function_registry = FunctionRegistry::default();
        let executable = Executable::<TestContextObject>::from_text_bytes(
            raw_bytes,
            self.loader.clone(),
            sbpf_version,
            function_registry,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Raw load error: {}", e)))?;
        
        self.executable = Some(Arc::new(executable));
        Ok(())
    }

    pub fn execute(&mut self, _mem: Vec<u8>) -> PyResult<(u64, u64)> {
        let executable = self.executable.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No ELF/Raw loaded"))?;
        
        let mut context_object = TestContextObject::new(100_000);
        let config = Config::default();
        let sbpf_version = SBPFVersion::V2;
        // For actual use, memory regions need proper addresses and permissions.
        let mem_mapping = MemoryMapping::new(vec![], &config, &sbpf_version)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Memory mapping error: {:?}", e)))?;
        
        let mut vm = EbpfVm::<TestContextObject>::new(
            self.loader.clone(),
            &sbpf_version,
            &mut context_object,
            mem_mapping,
            4096,
        );
        
        let (instruction_count, res) = vm.execute_program(executable, true);
        let res_std: Result<u64, _> = res.into();
        let res_val = res_std.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Execution error: {:?}", e)))?;
            
        Ok((instruction_count, res_val))
    }
}

/// Userspace CoW memory snapshotting for sub-ms mutation rollback.
#[pyclass]
pub struct SnapBPF {
    snapshots: DashMap<String, Arc<Vec<u8>>>,
}

#[pymethods]
impl SnapBPF {
    #[new]
    pub fn new() -> Self {
        Self {
            snapshots: DashMap::new(),
        }
    }

    /// Snapshot the current VM memory state.
    pub fn snapshot(&self, snapshot_id: &str, memory: Vec<u8>) {
        self.snapshots.insert(snapshot_id.to_string(), Arc::new(memory));
    }

    /// Restore a snapshot. Returns a cloned Vec (CoW isolation).
    pub fn restore(&self, snapshot_id: &str) -> Option<Vec<u8>> {
        self.snapshots.get(snapshot_id).map(|s| s.as_ref().clone())
    }

    /// Delete a snapshot. Returns true if it existed.
    pub fn delete(&self, snapshot_id: &str) -> bool {
        self.snapshots.remove(snapshot_id).is_some()
    }

    /// Number of stored snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapbpf_snapshot_and_restore() {
        let snap = SnapBPF::new();
        let memory = vec![1u8, 2, 3, 4, 5];

        snap.snapshot("test-1", memory.clone());

        let restored = snap.restore("test-1");
        assert!(restored.is_some());
        assert_eq!(restored.unwrap(), memory);
    }

    #[test]
    fn test_snapbpf_restore_nonexistent() {
        let snap = SnapBPF::new();
        assert!(snap.restore("nope").is_none());
    }

    #[test]
    fn test_snapbpf_delete() {
        let snap = SnapBPF::new();
        snap.snapshot("del-me", vec![10, 20, 30]);
        assert!(snap.delete("del-me"));
        assert!(snap.restore("del-me").is_none());
        assert!(!snap.delete("del-me")); // Already gone
    }

    #[test]
    fn test_snapbpf_cow_isolation() {
        let snap = SnapBPF::new();
        let original = vec![0u8; 1024];
        snap.snapshot("base", original.clone());

        // Restore and mutate — original snapshot should be unchanged
        let mut copy = snap.restore("base").unwrap();
        copy[0] = 0xFF;

        let original_again = snap.restore("base").unwrap();
        assert_eq!(original_again[0], 0); // CoW: original is pristine
    }
}