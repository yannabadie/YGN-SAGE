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
    /// Priority:
    ///   1. Wasm sandbox (if a component is loaded and the `sandbox`
    ///      feature is compiled in).
    ///   2. Subprocess fallback — **fail-closed by default**. Reached
    ///      only when (a) the Wasm path is unavailable or errored
    ///      AND (b) the operator explicitly opted in via
    ///      `SAGE_UNSAFE_UNSANDBOXED=1` (or true/yes/on). Without
    ///      that opt-in, this returns an ExecResult with exit_code
    ///      != 0 and a stderr naming the opt-in var.
    /// Raises ValueError if AST validation fails.
    ///
    /// This is P0.4 of the 2026-04-22 audit remediation (see
    /// docs/superpowers/specs/2026-04-22-safe-sandbox-redesign-spec.md).
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
                        warn!(
                            wasm_error = %e,
                            "Wasm execution failed; deciding fall-back based on SAGE_UNSAFE_UNSANDBOXED opt-in"
                        );
                    }
                }
            }
        }

        // 3. Subprocess fallback — P0.4 gate. Fail-closed unless the
        //    operator explicitly opted in. Empty stdout on deny so
        //    no partial output leaks; stderr names the var.
        if !is_unsafe_unsandboxed_enabled() {
            warn!(
                has_wasm = self.has_wasm(),
                "validate_and_execute DENIED subprocess fallback — SAGE_UNSAFE_UNSANDBOXED not set"
            );
            return Ok(ExecResult {
                stdout: String::new(),
                stderr: "No sandbox available and SAGE_UNSAFE_UNSANDBOXED is not set. \
                         Load a Wasm component via load_precompiled_component() / \
                         load_component(), or set SAGE_UNSAFE_UNSANDBOXED=1 to allow \
                         unsandboxed subprocess execution (not recommended; only for \
                         trusted environments or bench paths)."
                    .to_string(),
                exit_code: -1,
                timed_out: false,
                duration_ms: 0,
            });
        }

        warn!(
            has_wasm = self.has_wasm(),
            "validate_and_execute falling back to unsandboxed subprocess (SAGE_UNSAFE_UNSANDBOXED=1)"
        );

        // Subprocess fallback (release GIL)
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
    /// # Security Warning (Audit5 §6 + 2026-04-22 audit P0.3)
    /// This method bypasses tree-sitter AST validation entirely.
    /// As of the 2026-04-22 audit remediation, it is **gated** by the
    /// `SAGE_UNSAFE_RAW_EXEC` environment variable — without that set
    /// to `1` / `true` / `yes` / `on`, every call returns a fatal
    /// `ExecResult` with an explanatory error. Legitimate callers
    /// (ToolForge self-synthesis, etc.) opt in explicitly per-process;
    /// accidental or LLM-triggered paths now fail closed.
    ///
    /// Every successful call is still logged at WARN level for audit
    /// trail. Every denied call is logged at WARN level as well so
    /// operators notice the attempt.
    pub fn execute_raw(&self, py: Python<'_>, code: &str, args_json: &str) -> ExecResult {
        if !is_unsafe_raw_exec_enabled() {
            warn!(
                code_len = code.len(),
                "execute_raw DENIED — SAGE_UNSAFE_RAW_EXEC not set to a truthy value"
            );
            return ExecResult {
                stdout: String::new(),
                stderr: "execute_raw is disabled. Set SAGE_UNSAFE_RAW_EXEC=1 to opt in (bypasses AST validation — only for trusted callers).".to_string(),
                exit_code: -1,
                timed_out: false,
                duration_ms: 0,
            };
        }
        warn!(
            code_len = code.len(),
            has_wasm = self.has_wasm(),
            "execute_raw called — bypassing AST validation (SAGE_UNSAFE_RAW_EXEC=1)"
        );

        let python_exe = self.python_exe.clone();
        let code = code.to_string();
        let args = args_json.to_string();
        let timeout = self.timeout_secs;

        py.allow_threads(move || execute_python_subprocess(&python_exe, &code, &args, timeout))
    }
}

/// True iff the operator has opted into the raw-exec backdoor by
/// setting `SAGE_UNSAFE_RAW_EXEC` to a truthy value. Any other
/// value — including unset — denies `execute_raw`.
fn is_unsafe_raw_exec_enabled() -> bool {
    read_truthy_env("SAGE_UNSAFE_RAW_EXEC")
}

/// True iff the operator has opted into the unsandboxed subprocess
/// fallback by setting `SAGE_UNSAFE_UNSANDBOXED` to a truthy value.
/// Without this opt-in, `validate_and_execute` fails closed when no
/// Wasm component is loaded or when Wasm execution errors. Part of
/// P0.4 (2026-04-22 audit remediation).
fn is_unsafe_unsandboxed_enabled() -> bool {
    read_truthy_env("SAGE_UNSAFE_UNSANDBOXED")
}

/// Common truthy-env reader used by both unsafe opt-ins.
fn read_truthy_env(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
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
    fn test_validate_and_execute_subprocess_fallback_gated_by_env_var() {
        // P0.4 (2026-04-22): validate_and_execute fails CLOSED when
        // no Wasm component is loaded and SAGE_UNSAFE_UNSANDBOXED
        // is not set. Combined into ONE test because cargo runs
        // tests in parallel and env-var mutation is process-global
        // — splitting would race.
        init_python();
        let executor = ToolExecutor::new(None, 10);

        // Part 1: denied when env var is unset.
        std::env::remove_var("SAGE_UNSAFE_UNSANDBOXED");
        let r_denied = Python::with_gil(|py| {
            executor.validate_and_execute(py, r#"print("should not run")"#, "{}")
        });
        assert!(r_denied.is_ok(), "fail-closed should return Ok(ExecResult), not Err");
        let r_denied = r_denied.unwrap();
        assert_ne!(
            r_denied.exit_code, 0,
            "fail-closed must signal non-zero exit_code"
        );
        assert!(
            r_denied.stderr.contains("SAGE_UNSAFE_UNSANDBOXED"),
            "fail-closed stderr must name the opt-in var: {}",
            r_denied.stderr
        );
        assert!(
            r_denied.stdout.is_empty(),
            "fail-closed must not leak stdout: {}",
            r_denied.stdout
        );
        assert!(
            !r_denied.stdout.contains("should not run"),
            "subprocess must not have executed"
        );

        // Part 2: subprocess runs when explicitly opted in.
        std::env::set_var("SAGE_UNSAFE_UNSANDBOXED", "1");
        let r_allowed = Python::with_gil(|py| {
            executor.validate_and_execute(py, r#"print("fallback works")"#, "{}")
        });
        std::env::remove_var("SAGE_UNSAFE_UNSANDBOXED");
        assert!(r_allowed.is_ok());
        let r_allowed = r_allowed.unwrap();
        assert_eq!(
            r_allowed.exit_code, 0,
            "opted-in fallback should succeed; stderr: {}",
            r_allowed.stderr
        );
        assert!(r_allowed.stdout.contains("fallback works"));
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
    fn test_execute_raw_gated_by_env_var() {
        // P0.3 (2026-04-22): `execute_raw` is gated by
        // SAGE_UNSAFE_RAW_EXEC. Combined into ONE test because
        // cargo test runs tests in parallel and env-var mutation
        // is process-global — splitting across two tests races.
        init_python();
        let executor = ToolExecutor::new(None, 10);

        // Part 1: denied when env var is not set.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        let r_denied = Python::with_gil(|py| {
            executor.execute_raw(py, r#"print("should not run")"#, "{}")
        });
        assert_ne!(r_denied.exit_code, 0, "execute_raw must deny by default");
        assert!(
            r_denied.stderr.contains("SAGE_UNSAFE_RAW_EXEC"),
            "deny stderr should name the opt-in var: {}",
            r_denied.stderr
        );
        assert!(r_denied.stdout.is_empty(), "no stdout leak on deny");

        // Part 2: bypass works when explicitly opted in.
        std::env::set_var("SAGE_UNSAFE_RAW_EXEC", "1");
        let r_allowed = Python::with_gil(|py| {
            executor.execute_raw(py, r#"print("raw exec")"#, "{}")
        });
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        assert_eq!(
            r_allowed.exit_code, 0,
            "execute_raw should succeed when opted in; stderr: {}",
            r_allowed.stderr
        );
        assert!(
            r_allowed.stdout.contains("raw exec"),
            "opted-in exec should show stdout: {}",
            r_allowed.stdout
        );
    }

    #[test]
    fn test_double_opt_in_structural_invariants() {
        // 2026-04-22 P0.4 follow-up — both arbitrary-Python-execution
        // bypasses are gated independently. This test locks the
        // structural invariant: flipping only ONE of the two env
        // vars must NOT let the other path through.
        //
        // Matrix:
        //   - neither set: both deny
        //   - only UNSAFE_UNSANDBOXED set: raw denies, validate-path runs
        //   - only UNSAFE_RAW_EXEC set:    raw runs, validate denies
        //   - both set: both run
        init_python();
        let executor = ToolExecutor::new(None, 10);
        let safe_code = r#"print("probe")"#;

        // State A: neither set — double deny.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        std::env::remove_var("SAGE_UNSAFE_UNSANDBOXED");
        let a_raw = Python::with_gil(|py| executor.execute_raw(py, safe_code, "{}"));
        assert_ne!(a_raw.exit_code, 0, "raw must deny without its opt-in");
        assert!(a_raw.stderr.contains("SAGE_UNSAFE_RAW_EXEC"));
        let a_val = Python::with_gil(|py| executor.validate_and_execute(py, safe_code, "{}"))
            .expect("validate should return Ok even when fail-closed");
        assert_ne!(a_val.exit_code, 0, "validate must deny without unsandboxed opt-in");
        assert!(a_val.stderr.contains("SAGE_UNSAFE_UNSANDBOXED"));

        // State B: only UNSAFE_UNSANDBOXED — raw still denies; validate runs.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        std::env::set_var("SAGE_UNSAFE_UNSANDBOXED", "1");
        let b_raw = Python::with_gil(|py| executor.execute_raw(py, safe_code, "{}"));
        assert_ne!(
            b_raw.exit_code, 0,
            "raw must STILL deny when only the unsandboxed gate is open"
        );
        let b_val = Python::with_gil(|py| executor.validate_and_execute(py, safe_code, "{}"))
            .expect("validate ok");
        assert_eq!(
            b_val.exit_code, 0,
            "validate should now run (gate open); stderr: {}",
            b_val.stderr
        );

        // State C: only UNSAFE_RAW_EXEC — raw runs; validate still denies.
        std::env::set_var("SAGE_UNSAFE_RAW_EXEC", "1");
        std::env::remove_var("SAGE_UNSAFE_UNSANDBOXED");
        let c_raw = Python::with_gil(|py| executor.execute_raw(py, safe_code, "{}"));
        assert_eq!(
            c_raw.exit_code, 0,
            "raw should now run (gate open); stderr: {}",
            c_raw.stderr
        );
        let c_val = Python::with_gil(|py| executor.validate_and_execute(py, safe_code, "{}"))
            .expect("validate ok");
        assert_ne!(
            c_val.exit_code, 0,
            "validate must STILL deny when only the raw gate is open"
        );

        // Clean up so other parallel tests aren't affected.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        std::env::remove_var("SAGE_UNSAFE_UNSANDBOXED");
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
