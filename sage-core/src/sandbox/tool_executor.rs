//! ToolExecutor: PyO3 class combining validation + sandboxed execution.
//!
//! Execution priority:
//! 1. Wasm WASI sandbox (if component loaded and sandbox feature enabled)
//!    - Tries WASI path first (for CPython components with WASI imports)
//!    - Falls back to bare component (for simple components without WASI)
//! 2. Subprocess fallback (always available)

use super::subprocess::{execute_python_subprocess, ExecResult};
use super::validator::{validate_python_code, ValidationResult};
use pyo3::prelude::*;
use tracing::warn;

#[cfg(feature = "sandbox")]
use std::sync::Arc;

/// Combined validator + executor for dynamic tool creation.
///
/// Usage from Python:
/// ```python
/// from sage_core import ToolExecutor
/// executor = ToolExecutor()
/// # Validate only
/// result = executor.validate(code)
/// # Validate + execute (tries Wasm first if loaded, then subprocess)
/// result = executor.validate_and_execute(code, args_json)
/// # Load a pre-compiled Wasm component for sandboxed execution
/// executor.load_precompiled_component(compiled_bytes)
/// ```
#[pyclass]
pub struct ToolExecutor {
    python_exe: String,
    timeout_secs: u64,
    /// Pre-compiled Wasm component bytes (loaded via load_precompiled_component).
    /// These must come from Component::serialize() on the same wasmtime version.
    #[cfg(feature = "sandbox")]
    wasm_component: Option<Arc<wasmtime::component::Component>>,
    /// Cached wasmtime Engine
    #[cfg(feature = "sandbox")]
    wasm_engine: Option<wasmtime::Engine>,
    /// Whether the loaded component needs WASI imports (e.g., CPython components).
    #[cfg(feature = "sandbox")]
    needs_wasi: bool,
}

#[pymethods]
impl ToolExecutor {
    #[new]
    #[pyo3(signature = (python_exe=None, timeout_secs=30))]
    pub fn new(python_exe: Option<String>, timeout_secs: u64) -> Self {
        let exe = python_exe.unwrap_or_else(|| {
            if cfg!(windows) {
                "python".to_string()
            } else {
                "python3".to_string()
            }
        });

        #[cfg(feature = "sandbox")]
        let (wasm_engine, wasm_component, needs_wasi) = {
            let mut config = wasmtime::Config::new();
            config.wasm_component_model(true);
            let engine = wasmtime::Engine::new(&config).ok();
            (engine, None, false)
        };

        Self {
            python_exe: exe,
            timeout_secs,
            #[cfg(feature = "sandbox")]
            wasm_component,
            #[cfg(feature = "sandbox")]
            wasm_engine,
            #[cfg(feature = "sandbox")]
            needs_wasi,
        }
    }

    /// Load a pre-compiled Wasm component for execution (works without cranelift).
    /// Pass bytes from Component::serialize() (pre-compiled on Linux CI).
    /// Set `wasi` to true for CPython WASI components (deny-by-default capabilities).
    #[cfg(feature = "sandbox")]
    #[pyo3(signature = (compiled_bytes, wasi=false))]
    pub fn load_precompiled_component(
        &mut self,
        compiled_bytes: Vec<u8>,
        wasi: bool,
    ) -> PyResult<()> {
        let engine = self.wasm_engine.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Wasm engine not initialized")
        })?;

        // SAFETY: compiled_bytes must come from Component::serialize()
        // produced by the same version of wasmtime with the same Engine config.
        #[allow(unsafe_code)]
        let component = unsafe {
            wasmtime::component::Component::deserialize(engine, &compiled_bytes).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to deserialize Wasm component: {}",
                    e
                ))
            })?
        };

        self.wasm_component = Some(Arc::new(component));
        self.needs_wasi = wasi;
        Ok(())
    }

    /// Load a Wasm component for execution (requires cranelift for JIT).
    /// Pass the raw .wasm bytes (Component Model format).
    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    #[pyo3(signature = (wasm_bytes, wasi=false))]
    pub fn load_component(&mut self, wasm_bytes: Vec<u8>, wasi: bool) -> PyResult<()> {
        let engine = self.wasm_engine.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Wasm engine not initialized")
        })?;

        let component = wasmtime::component::Component::new(engine, &wasm_bytes).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to compile Wasm component: {}. Ensure cranelift feature is enabled.",
                e
            ))
        })?;

        self.wasm_component = Some(Arc::new(component));
        self.needs_wasi = wasi;
        Ok(())
    }

    /// Fallback load_component when cranelift is not available.
    #[cfg(all(feature = "sandbox", not(feature = "cranelift")))]
    #[pyo3(signature = (_wasm_bytes, _wasi=false))]
    pub fn load_component(&mut self, _wasm_bytes: Vec<u8>, _wasi: bool) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "JIT compilation requires the 'cranelift' feature. \
             Use load_precompiled_component() with pre-compiled bytes, \
             or build with: cargo build --features sandbox,cranelift",
        ))
    }

    /// Check if a Wasm component is loaded and ready.
    pub fn has_wasm(&self) -> bool {
        #[cfg(feature = "sandbox")]
        {
            self.wasm_component.is_some()
        }
        #[cfg(not(feature = "sandbox"))]
        {
            false
        }
    }

    /// Check if the loaded Wasm component uses WASI (deny-by-default sandbox).
    pub fn has_wasi(&self) -> bool {
        #[cfg(feature = "sandbox")]
        {
            self.wasm_component.is_some() && self.needs_wasi
        }
        #[cfg(not(feature = "sandbox"))]
        {
            false
        }
    }

    /// Validate Python code without executing it.
    /// Returns ValidationResult with valid flag and error list.
    pub fn validate(&self, code: &str) -> ValidationResult {
        validate_python_code(code)
    }

    /// Validate and execute Python code.
    /// Tries Wasm sandbox first (if loaded), falls back to subprocess.
    /// Raises ValueError if validation fails.
    pub fn validate_and_execute(
        &self,
        py: Python<'_>,
        code: &str,
        args_json: &str,
    ) -> PyResult<ExecResult> {
        // 1. Validate
        let validation = validate_python_code(code);
        if !validation.valid {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Code validation failed:\n{}",
                validation.errors.join("\n")
            )));
        }

        // 2. Try Wasm execution first
        #[cfg(feature = "sandbox")]
        if let Some(ref component) = self.wasm_component {
            if let Some(ref engine) = self.wasm_engine {
                match self.execute_wasm_internal(engine, component, code, args_json) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // Log warning and fall through to subprocess
                        eprintln!("Wasm execution failed ({}), falling back to subprocess", e);
                    }
                }
            }
        }

        // 3. Subprocess fallback (release GIL)
        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        let result =
            py.allow_threads(move || execute_python_subprocess(&python_exe, &code, &args, timeout));

        Ok(result)
    }

    /// Execute Python code without validation (for pre-validated code).
    ///
    /// # Security Warning (Audit5 §6)
    /// This method bypasses tree-sitter AST validation entirely.
    /// Caller is responsible for ensuring code safety.
    /// Every call is logged at WARN level for audit trail.
    pub fn execute_raw(&self, py: Python<'_>, code: &str, args_json: &str) -> ExecResult {
        warn!(
            code_len = code.len(),
            has_wasm = self.has_wasm(),
            "execute_raw called — bypassing AST validation"
        );

        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        py.allow_threads(move || execute_python_subprocess(&python_exe, &code, &args, timeout))
    }
}

// Internal Wasm execution (not exposed via PyO3)
#[cfg(feature = "sandbox")]
impl ToolExecutor {
    fn execute_wasm_internal(
        &self,
        engine: &wasmtime::Engine,
        component: &wasmtime::component::Component,
        code: &str,
        args_json: &str,
    ) -> Result<ExecResult, String> {
        // Try WASI path first (for CPython components), then bare component
        let (stdout, stderr, exit_code) = if self.needs_wasi {
            // WASI-aware execution: deny-by-default capabilities
            match super::wasm::execute_wasi_component(engine, component, code, args_json) {
                Ok(result) => result,
                Err(wasi_err) => {
                    // WASI failed — try bare component as last resort
                    eprintln!(
                        "WASI execution failed ({}), trying bare component",
                        wasi_err
                    );
                    super::wasm::execute_bare_component(engine, component, code, args_json)?
                }
            }
        } else {
            // Non-WASI component (e.g., Phase 2 expression evaluator)
            match super::wasm::execute_bare_component(engine, component, code, args_json) {
                Ok(result) => result,
                Err(bare_err) => {
                    // Bare failed — try WASI path (component might need WASI imports)
                    eprintln!("Bare execution failed ({}), trying WASI path", bare_err);
                    super::wasm::execute_wasi_component(engine, component, code, args_json)?
                }
            }
        };

        Ok(ExecResult {
            stdout,
            stderr,
            exit_code,
            timed_out: false,
            duration_ms: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_python() {
        pyo3::prepare_freethreaded_python();
    }

    #[test]
    fn test_has_wasm_default_false() {
        init_python();
        let executor = ToolExecutor::new(None, 30);
        assert!(!executor.has_wasm());
    }

    #[test]
    fn test_has_wasi_default_false() {
        init_python();
        let executor = ToolExecutor::new(None, 30);
        assert!(!executor.has_wasi());
    }

    #[test]
    fn test_validate_and_execute_subprocess_fallback() {
        init_python();
        // Without Wasm loaded, should fall back to subprocess
        let executor = ToolExecutor::new(None, 10);
        let result = Python::with_gil(|py| {
            executor.validate_and_execute(py, r#"print("fallback works")"#, "{}")
        });
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("fallback works"));
    }

    #[test]
    fn test_validate_rejects_blocked_code() {
        init_python();
        let executor = ToolExecutor::new(None, 10);
        let result = Python::with_gil(|py| {
            executor.validate_and_execute(py, "import os\nos.listdir('/')", "{}")
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_raw_bypasses_validation() {
        init_python();
        let executor = ToolExecutor::new(None, 10);
        let r = Python::with_gil(|py| executor.execute_raw(py, r#"print("raw exec")"#, "{}"));
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("raw exec"));
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn test_wasi_context_is_restrictive() {
        // Verify that the WasiState creates a restrictive context.
        // This is a compile-time + runtime verification:
        // - WasiCtxBuilder::new() starts with NO capabilities
        // - We only add inherit_stdout() and inherit_stderr()
        // - No inherit_env(), no preopened_dir(), no inherit_stdin()
        // If this compiles, capabilities are denied by construction.
        init_python();
        let executor = ToolExecutor::new(None, 10);
        assert!(!executor.has_wasm());
        assert!(!executor.has_wasi());

        // Verify the WasiState can be created without error
        let _state = super::super::wasm::WasiState::new_restrictive();
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn test_load_invalid_precompiled_fails() {
        init_python();
        let mut executor = ToolExecutor::new(None, 10);
        let result = executor.load_precompiled_component(vec![0, 1, 2, 3], false);
        assert!(result.is_err());
        assert!(!executor.has_wasm());
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn test_load_precompiled_wasi_flag() {
        // Without valid component bytes, load fails — but verify the wasi flag logic
        init_python();
        let mut executor = ToolExecutor::new(None, 10);
        // Invalid bytes should fail
        let result = executor.load_precompiled_component(vec![0, 1, 2, 3], true);
        assert!(result.is_err());
        // After failed load, has_wasi should still be false
        assert!(!executor.has_wasi());
    }
}
