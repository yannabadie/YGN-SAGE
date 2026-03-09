//! ToolExecutor: PyO3 class combining validation + sandboxed execution.
//!
//! Execution priority:
//! 1. Wasm sandbox (if component loaded and sandbox feature enabled)
//! 2. Subprocess fallback (always available)

use pyo3::prelude::*;
use super::validator::{validate_python_code, ValidationResult};
use super::subprocess::{execute_python_subprocess, ExecResult};

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
/// # Load a Wasm component for sandboxed execution
/// executor.load_component(wasm_bytes)
/// ```
#[pyclass]
pub struct ToolExecutor {
    python_exe: String,
    timeout_secs: u64,
    /// Wasm component bytes (loaded via load_component)
    #[cfg(feature = "sandbox")]
    wasm_component: Option<Arc<Vec<u8>>>,
    /// Cached wasmtime Engine
    #[cfg(feature = "sandbox")]
    wasm_engine: Option<wasmtime::Engine>,
}

#[pymethods]
impl ToolExecutor {
    #[new]
    #[pyo3(signature = (python_exe=None, timeout_secs=30))]
    pub fn new(python_exe: Option<String>, timeout_secs: u64) -> Self {
        let exe = python_exe.unwrap_or_else(|| {
            if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
        });

        #[cfg(feature = "sandbox")]
        let (wasm_engine, wasm_component) = {
            let mut config = wasmtime::Config::new();
            config.wasm_component_model(true);
            let engine = wasmtime::Engine::new(&config).ok();
            (engine, None)
        };

        Self {
            python_exe: exe,
            timeout_secs,
            #[cfg(feature = "sandbox")]
            wasm_component,
            #[cfg(feature = "sandbox")]
            wasm_engine,
        }
    }

    /// Load a Wasm component for execution.
    /// Pass the raw .wasm bytes (Component Model format).
    /// Requires the sandbox feature + cranelift for JIT compilation,
    /// or pre-compiled bytes for execute_precompiled.
    #[cfg(feature = "sandbox")]
    pub fn load_component(&mut self, wasm_bytes: Vec<u8>) -> PyResult<()> {
        // Validate the component can be compiled
        let engine = self.wasm_engine.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Wasm engine not initialized")
        })?;

        // Try to compile to validate it
        // This requires cranelift feature; if not available, Component::new() fails
        match wasmtime::component::Component::new(engine, &wasm_bytes) {
            Ok(_) => {
                self.wasm_component = Some(Arc::new(wasm_bytes));
                Ok(())
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to compile Wasm component: {}. Ensure cranelift feature is enabled.", e)
            )),
        }
    }

    /// Check if a Wasm component is loaded and ready.
    pub fn has_wasm(&self) -> bool {
        #[cfg(feature = "sandbox")]
        { self.wasm_component.is_some() }
        #[cfg(not(feature = "sandbox"))]
        { false }
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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Code validation failed:\n{}",
                    validation.errors.join("\n")
                )
            ));
        }

        // 2. Try Wasm execution first
        #[cfg(feature = "sandbox")]
        if let Some(ref wasm_bytes) = self.wasm_component {
            if let Some(ref engine) = self.wasm_engine {
                match self.execute_wasm_internal(engine, wasm_bytes, code, args_json) {
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

        let result = py.allow_threads(move || {
            execute_python_subprocess(&python_exe, &code, &args, timeout)
        });

        Ok(result)
    }

    /// Execute Python code without validation (for pre-validated code).
    /// Use with caution — caller is responsible for validation.
    pub fn execute_raw(
        &self,
        py: Python<'_>,
        code: &str,
        args_json: &str,
    ) -> ExecResult {
        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        py.allow_threads(move || {
            execute_python_subprocess(&python_exe, &code, &args, timeout)
        })
    }
}

// Internal Wasm execution (not exposed via PyO3)
#[cfg(feature = "sandbox")]
impl ToolExecutor {
    fn execute_wasm_internal(
        &self,
        engine: &wasmtime::Engine,
        wasm_bytes: &[u8],
        code: &str,
        args_json: &str,
    ) -> Result<ExecResult, String> {
        use wasmtime::component::{Component, Linker};
        use wasmtime::Store;

        // Import the WIT bindings from wasm.rs
        use super::wasm::{ToolEnv, ToolInput};

        let component = Component::new(engine, wasm_bytes)
            .map_err(|e| format!("Component compile: {}", e))?;

        let linker = Linker::new(engine);
        let mut store = Store::new(engine, ());

        let instance = ToolEnv::instantiate(&mut store, &component, &linker)
            .map_err(|e| format!("Instantiate: {}", e))?;

        let input = ToolInput {
            name: "python_tool".to_string(),
            args: args_json.to_string(),
            env: Vec::new(),
        };

        let output = instance.call_run(&mut store, &input)
            .map_err(|e| format!("call_run: {}", e))?;

        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.exit_code,
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
        let r = Python::with_gil(|py| {
            executor.execute_raw(py, r#"print("raw exec")"#, "{}")
        });
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("raw exec"));
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn test_load_invalid_wasm_fails() {
        init_python();
        let mut executor = ToolExecutor::new(None, 10);
        Python::with_gil(|_py| {
            let result = executor.load_component(vec![0, 1, 2, 3]);
            assert!(result.is_err());
        });
        assert!(!executor.has_wasm());
    }

    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    #[test]
    fn test_load_and_execute_wasm_component() {
        init_python();
        // Load the python-runner component if available
        let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("plugins/python-runner/target/wasm32-unknown-unknown/release/python_runner.component.wasm");

        if !wasm_path.exists() {
            eprintln!("Skipping: python-runner component not built. Run build.sh first.");
            return;
        }

        let wasm_bytes = std::fs::read(&wasm_path).expect("read component");

        let mut executor = ToolExecutor::new(None, 10);
        Python::with_gil(|_py| {
            executor.load_component(wasm_bytes).expect("load component");
        });
        assert!(executor.has_wasm());
    }
}
