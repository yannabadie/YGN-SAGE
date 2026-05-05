//! ToolExecutor: PyO3 class combining Python AST validation and sandboxed execution.
//!
//! `validate_and_execute()` validates code first, then executes through the
//! safest available Wasm path: an operator-loaded Component-Model sandbox if
//! present, otherwise the embedded RustPython wasm32-wasip1 sandbox when built
//! with `sandbox + cranelift` and a bundled `rustpython.wasm`.
//!
//! The embedded sandbox is deny-by-default: no filesystem, network,
//! environment, or subprocess access, with bounded memory and timeout enforcement.
//! If no Wasm path is available, validated execution hard-fails; it does not
//! fall back to subprocess execution. The only subprocess-capable bypass is
//! `execute_raw()`, which skips AST validation and requires the explicit
//! `SAGE_UNSAFE_RAW_EXEC=1` opt-in for trusted callers.

use super::subprocess::{execute_python_subprocess, ExecResult};
use super::validator::{validate_python_code, ValidationResult};
use pyo3::prelude::*;
use tracing::warn;

#[cfg(feature = "sandbox")]
use std::sync::Arc;

#[cfg(all(feature = "sandbox", feature = "cranelift"))]
use super::wasm_python::WasmPythonExecutor;
#[cfg(all(feature = "sandbox", feature = "cranelift"))]
use std::sync::OnceLock;

/// Combined validator + executor for dynamic tool creation.
///
/// Usage from Python:
/// ```python
/// from sage_core import ToolExecutor
/// executor = ToolExecutor()
/// # Validate only
/// result = executor.validate(code)
/// # Validate + execute (Wasm-only; hard-fails if no sandbox is available)
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
    /// Lazily-initialised embedded RustPython wasm executor. Built on
    /// first use; shared across calls. `None` inside the OnceLock means
    /// "we tried and couldn't build one" — typically because build.rs
    /// didn't find a compiled `rustpython.wasm`. `execute_raw` prefers
    /// this path when populated and falls through to subprocess
    /// otherwise.
    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    wasm_python: OnceLock<Option<Arc<WasmPythonExecutor>>>,
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
            #[cfg(all(feature = "sandbox", feature = "cranelift"))]
            wasm_python: OnceLock::new(),
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
    ///
    /// Priority (2026-04-22 §5 flip — post red-team corpus):
    ///   1. Wasm Component-Model sandbox (if the operator explicitly
    ///      loaded a component via `load_component` /
    ///      `load_precompiled_component`).
    ///   2. Embedded RustPython wasm sandbox (deny-by-default WASI-p1,
    ///      no filesystem / network / env / subprocess). Available
    ///      on any build with `sandbox + cranelift` AND a bundled
    ///      `rustpython.wasm` — the default for release builds since
    ///      the §5 flip.
    ///   3. **Hard fail** — no subprocess fallback. If neither Wasm
    ///      path is available, the only route forward is to rebuild
    ///      with `sandbox + cranelift` or call the explicitly-named
    ///      `execute_raw` (gated by `SAGE_UNSAFE_RAW_EXEC=1`), which
    ///      is a separate audited bypass with its own gate.
    ///
    /// The old `SAGE_UNSAFE_UNSANDBOXED=1` silent subprocess fallback
    /// was removed in the §5 flip once the 40-attack red-team corpus
    /// showed the Wasm sandbox is safe-by-default. Callers who still
    /// need an unsandboxed exit must use `execute_raw` with the
    /// `SAGE_UNSAFE_RAW_EXEC` opt-in and accept that their code
    /// bypasses BOTH AST validation AND the Wasm sandbox.
    ///
    /// Raises ValueError if AST validation fails.
    // Cycle-11 CI debug 2026-05-05: `py` and `args_json` are used only
    // inside `#[cfg(feature = "sandbox")]` blocks below. On a build
    // without `sandbox` (tool-executor only), they're unused. Suppress
    // the lint at the function level rather than renaming with a
    // leading underscore — the public param names appear in docs and
    // IDE hints, and renaming would mask future legitimate
    // unused-variable bugs in the function body.
    #[allow(unused_variables)]
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

        // 2. Try operator-loaded Wasm Component-Model first (if any).
        #[cfg(feature = "sandbox")]
        if let Some(ref component) = self.wasm_component {
            if let Some(ref engine) = self.wasm_engine {
                match self.execute_wasm_internal(engine, component, code, args_json) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        warn!(
                            wasm_error = %e,
                            "Wasm Component-Model execution failed; trying embedded RustPython"
                        );
                    }
                }
            }
        }

        // 3. Embedded RustPython wasm sandbox — default sandboxed
        //    path since the §5 flip. Deny-by-default filesystem /
        //    network / env; 256 MiB memory cap; epoch-interrupt
        //    timeout.
        #[cfg(all(feature = "sandbox", feature = "cranelift"))]
        {
            let wasm = self
                .wasm_python
                .get_or_init(|| match WasmPythonExecutor::new() {
                    Ok(e) => Some(Arc::new(e)),
                    Err(err) => {
                        warn!(error = %err, "embedded RustPython unavailable");
                        None
                    }
                });
            if let Some(wp) = wasm {
                let wp = Arc::clone(wp);
                let code_owned = code.to_string();
                let args_owned = args_json.to_string();
                let timeout = self.timeout_secs;
                let result =
                    py.allow_threads(move || wp.execute(&code_owned, &args_owned, timeout));
                return Ok(result);
            }
        }

        // 4. Hard fail — no Wasm path available. Tell the operator
        //    exactly how to fix their build.
        warn!(
            has_wasm = self.has_wasm(),
            "validate_and_execute has no sandbox available — all Wasm paths absent"
        );
        Ok(ExecResult {
            stdout: String::new(),
            stderr: "No Wasm sandbox available. Either load a Component-Model \
                     component via load_precompiled_component() / load_component(), \
                     or rebuild sage-core with --features sandbox,cranelift and \
                     ensure build.rs bundles a compiled rustpython.wasm. \
                     If you genuinely need unsandboxed execution, use execute_raw() \
                     with SAGE_UNSAFE_RAW_EXEC=1 (bypasses AST validation AND the \
                     Wasm sandbox — audited backdoor for trusted callers only)."
                .to_string(),
            exit_code: -1,
            timed_out: false,
            duration_ms: 0,
        })
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

        // Prefer the embedded RustPython Wasm sandbox when it's
        // available — deny-by-default filesystem/network/env, bounded
        // runtime, no host access. Falls through to subprocess only
        // when the wasm bytes aren't bundled (feature off or build.rs
        // didn't find a compiled rustpython.wasm).
        #[cfg(all(feature = "sandbox", feature = "cranelift"))]
        {
            let wasm = self.wasm_python.get_or_init(|| {
                match WasmPythonExecutor::new() {
                    Ok(e) => Some(Arc::new(e)),
                    Err(err) => {
                        warn!(error = %err, "WasmPythonExecutor unavailable — falling back to subprocess");
                        None
                    }
                }
            });
            if let Some(wp) = wasm {
                warn!(
                    code_len = code.len(),
                    "execute_raw routed through embedded RustPython wasm sandbox (SAGE_UNSAFE_RAW_EXEC=1)"
                );
                let wp = Arc::clone(wp);
                let code_owned = code.to_string();
                let args_owned = args_json.to_string();
                let timeout = self.timeout_secs;
                return py.allow_threads(move || wp.execute(&code_owned, &args_owned, timeout));
            }
        }

        warn!(
            code_len = code.len(),
            has_wasm = self.has_wasm(),
            "execute_raw called — bypassing AST validation, using subprocess (SAGE_UNSAFE_RAW_EXEC=1)"
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
///
/// Note: `SAGE_UNSAFE_UNSANDBOXED` (the old silent-subprocess-
/// fallback gate for `validate_and_execute`) was removed in the
/// §5 flip after the 40-attack red-team corpus validated the Wasm
/// sandbox. `execute_raw` remains the only opt-in bypass.
fn is_unsafe_raw_exec_enabled() -> bool {
    match std::env::var("SAGE_UNSAFE_RAW_EXEC") {
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
    use std::sync::Mutex;

    /// Serialise the env-var-mutating tests. `SAGE_UNSAFE_RAW_EXEC`
    /// is process-global, so parallel tests that set/unset it race —
    /// this matters more now that the embedded wasm sandbox holds
    /// the process for 30+ s per call. Tests that touch the var
    /// MUST hold this lock for their whole duration.
    ///
    /// Post §5 flip: `SAGE_UNSAFE_UNSANDBOXED` was removed so only
    /// one env var needs serialising, but the mutex name is kept
    /// for forward-compat if a future gate is added.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

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
    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    fn test_validate_and_execute_uses_embedded_wasm_by_default() {
        // Post §5 flip (2026-04-22 after red-team corpus passed):
        // `validate_and_execute` runs code in the embedded RustPython
        // wasm sandbox by default — no opt-in required. The old
        // `SAGE_UNSAFE_UNSANDBOXED` gate is gone. Defense-in-depth
        // against filesystem / network / env escape is locked down
        // separately by the 40-attack red-team corpus in
        // sage-python/tests/test_wasm_sandbox_redteam.py. This Rust
        // test only locks the *structural* invariant: safe code runs
        // through the Wasm path end-to-end with no env-var opt-in.
        init_python();
        let executor = ToolExecutor::new(None, 10);

        let r_ok = Python::with_gil(|py| {
            executor.validate_and_execute(py, r#"print("hello from wasm default")"#, "{}")
        });
        assert!(r_ok.is_ok(), "validate_and_execute should return Ok");
        let r_ok = r_ok.unwrap();
        assert_eq!(
            r_ok.exit_code, 0,
            "default sandboxed execution should succeed; stderr: {}",
            r_ok.stderr
        );
        assert!(
            r_ok.stdout.contains("hello from wasm default"),
            "stdout captured through the sandbox: {}",
            r_ok.stdout
        );
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
        let _env_guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        init_python();
        let executor = ToolExecutor::new(None, 10);

        // Part 1: denied when env var is not set.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        let r_denied =
            Python::with_gil(|py| executor.execute_raw(py, r#"print("should not run")"#, "{}"));
        assert_ne!(r_denied.exit_code, 0, "execute_raw must deny by default");
        assert!(
            r_denied.stderr.contains("SAGE_UNSAFE_RAW_EXEC"),
            "deny stderr should name the opt-in var: {}",
            r_denied.stderr
        );
        assert!(r_denied.stdout.is_empty(), "no stdout leak on deny");

        // Part 2: bypass works when explicitly opted in.
        std::env::set_var("SAGE_UNSAFE_RAW_EXEC", "1");
        let r_allowed =
            Python::with_gil(|py| executor.execute_raw(py, r#"print("raw exec")"#, "{}"));
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
    #[cfg(all(feature = "sandbox", feature = "cranelift"))]
    fn test_execute_raw_gate_independent_of_validate_and_execute() {
        // Post §5 flip (was `test_double_opt_in_structural_invariants`
        // before the flip; the unsandboxed-subprocess gate is gone so
        // the matrix collapses from 4 states to 2).
        //
        // Contract now: `execute_raw` is the ONLY gated bypass —
        // requires SAGE_UNSAFE_RAW_EXEC because it bypasses AST
        // validation AND the Wasm sandbox. `validate_and_execute`
        // runs in the Wasm sandbox by default, no opt-in. Flipping
        // the raw gate must have zero effect on the validate-path
        // behaviour.
        let _env_guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        init_python();
        let executor = ToolExecutor::new(None, 10);
        let safe_code = r#"print("probe")"#;

        // State A: raw gate closed.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        let a_raw = Python::with_gil(|py| executor.execute_raw(py, safe_code, "{}"));
        assert_ne!(a_raw.exit_code, 0, "raw must deny without its opt-in");
        assert!(a_raw.stderr.contains("SAGE_UNSAFE_RAW_EXEC"));
        let a_val = Python::with_gil(|py| executor.validate_and_execute(py, safe_code, "{}"))
            .expect("validate should return Ok");
        assert_eq!(
            a_val.exit_code, 0,
            "validate-path runs sandboxed regardless of raw gate; stderr: {}",
            a_val.stderr
        );
        assert!(a_val.stdout.contains("probe"));

        // State B: raw gate open.
        std::env::set_var("SAGE_UNSAFE_RAW_EXEC", "1");
        let b_raw = Python::with_gil(|py| executor.execute_raw(py, safe_code, "{}"));
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
        assert_eq!(
            b_raw.exit_code, 0,
            "raw should now run (gate open); stderr: {}",
            b_raw.stderr
        );
        assert!(b_raw.stdout.contains("probe"));
        let b_val = Python::with_gil(|py| executor.validate_and_execute(py, safe_code, "{}"))
            .expect("validate ok");
        assert_eq!(
            b_val.exit_code, 0,
            "validate-path still runs regardless of raw gate state"
        );

        // Clean up so other parallel tests aren't affected.
        std::env::remove_var("SAGE_UNSAFE_RAW_EXEC");
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
