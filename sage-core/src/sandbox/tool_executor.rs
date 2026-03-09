//! ToolExecutor: PyO3 class combining validation + sandboxed execution.
//!
//! This is the single entry point for Python tool creation from sage-core.
//! Python `meta.py` calls `sage_core.ToolExecutor.validate_and_execute()`.

use pyo3::prelude::*;
use super::validator::{validate_python_code, ValidationResult};
use super::subprocess::{execute_python_subprocess, ExecResult};

/// Combined validator + executor for dynamic tool creation.
///
/// Usage from Python:
/// ```python
/// from sage_core import ToolExecutor
/// executor = ToolExecutor()
/// # Validate only
/// result = executor.validate(code)
/// # Validate + execute
/// result = executor.validate_and_execute(code, args_json)
/// ```
#[pyclass]
pub struct ToolExecutor {
    python_exe: String,
    timeout_secs: u64,
}

#[pymethods]
impl ToolExecutor {
    #[new]
    #[pyo3(signature = (python_exe=None, timeout_secs=30))]
    pub fn new(python_exe: Option<String>, timeout_secs: u64) -> Self {
        let exe = python_exe.unwrap_or_else(|| {
            if cfg!(windows) { "python".to_string() } else { "python3".to_string() }
        });
        Self {
            python_exe: exe,
            timeout_secs,
        }
    }

    /// Validate Python code without executing it.
    /// Returns ValidationResult with valid flag and error list.
    pub fn validate(&self, code: &str) -> ValidationResult {
        validate_python_code(code)
    }

    /// Validate and execute Python code in a sandboxed subprocess.
    /// Raises ValueError if validation fails.
    /// Returns ExecResult if validation passes and code is executed.
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

        // 2. Execute in subprocess (release GIL)
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
