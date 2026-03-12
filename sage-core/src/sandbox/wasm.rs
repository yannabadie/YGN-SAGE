use pyo3::prelude::*;
use std::collections::HashMap;
use wasmtime::component::*;
use wasmtime::{Config, Engine, Store};

// Generate host bindings from our WIT file
bindgen!({
    world: "tool-env",
    path: "interface.wit",
});

/// WASI host state: holds WasiCtx + ResourceTable for component execution.
/// Capabilities are deny-by-default:
/// - NO filesystem access (no preopened_dir)
/// - NO env var access (no inherit_env)
/// - NO network access (not available in WASI preview2)
/// - NO subprocess spawning (not available in WASI preview2)
/// - stdout/stderr inherited for output capture
pub struct WasiState {
    ctx: wasmtime_wasi::WasiCtx,
    table: ResourceTable,
}

impl wasmtime_wasi::WasiView for WasiState {
    fn ctx(&mut self) -> wasmtime_wasi::WasiCtxView<'_> {
        wasmtime_wasi::WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

impl WasiState {
    /// Create a restrictive WASI state — deny-by-default.
    /// Only stdout and stderr are inherited (for output capture).
    /// Blocked: filesystem, env vars, network, subprocess.
    pub fn new_restrictive() -> Self {
        let ctx = wasmtime_wasi::WasiCtxBuilder::new()
            .inherit_stdout()
            .inherit_stderr()
            // Deliberately NOT calling:
            // - .inherit_env()     → blocks env var access
            // - .preopened_dir()   → blocks filesystem access
            // - .inherit_stdin()   → blocks stdin injection
            // Network and subprocess are not available in WASI preview2
            .build();
        Self {
            ctx,
            table: ResourceTable::new(),
        }
    }
}

/// Component-based Wasm sandbox for structured tool execution.
/// Uses the Component Model for typed I/O (WIT-defined tool-env world).
///
/// On Windows (no cranelift feature), only pre-compiled components are
/// supported via `execute_precompiled()`. On Linux CI with cranelift,
/// `execute()` compiles Wasm on the fly.
///
/// WASI-aware execution (`run_component_wasi`) is available for CPython
/// components built with componentize-py. Capabilities are deny-by-default.
#[pyclass]
pub struct WasmSandbox {
    engine: Engine,
    component_cache: dashmap::DashMap<String, Component>,
}

#[pymethods]
impl WasmSandbox {
    #[new]
    pub fn new() -> PyResult<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);

        let engine = Engine::new(&config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Wasmtime Engine error: {}",
                e
            ))
        })?;

        Ok(Self {
            engine,
            component_cache: dashmap::DashMap::new(),
        })
    }

    /// Run a pre-compiled Wasm component (works without cranelift).
    /// compiled_bytes: Output of `Component::serialize()` (pre-compiled on Linux CI).
    pub fn execute_precompiled(
        &self,
        name: String,
        compiled_bytes: Vec<u8>,
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        // Include content hash in cache key to detect stale entries.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        compiled_bytes.hash(&mut hasher);
        let cache_key = format!("{}_{:x}", name, hasher.finish());

        let component_result = self.component_cache.entry(cache_key).or_try_insert_with(
            || -> PyResult<Component> {
                // SAFETY: compiled_bytes must come from Component::serialize()
                // produced by the same version of wasmtime with the same Engine config.
                #[allow(unsafe_code)]
                unsafe {
                    Component::deserialize(&self.engine, &compiled_bytes).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to deserialize Wasm component: {}",
                            e
                        ))
                    })
                }
            },
        );

        let component = match component_result {
            Ok(entry) => entry.value().clone(),
            Err(e) => return Err(e),
        };

        self.run_component(component, tool_name, args_json, env)
    }

    /// Run a pre-compiled WASI component with deny-by-default capabilities.
    /// For CPython components built with componentize-py.
    /// WASI capabilities blocked: filesystem, env vars, network, subprocess.
    pub fn execute_precompiled_wasi(
        &self,
        name: String,
        compiled_bytes: Vec<u8>,
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        // Include content hash in cache key to detect stale entries.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        compiled_bytes.hash(&mut hasher);
        let cache_key = format!("{}_{:x}", name, hasher.finish());

        let component_result = self.component_cache.entry(cache_key).or_try_insert_with(
            || -> PyResult<Component> {
                // SAFETY: compiled_bytes must come from Component::serialize()
                // produced by the same version of wasmtime with the same Engine config.
                #[allow(unsafe_code)]
                unsafe {
                    Component::deserialize(&self.engine, &compiled_bytes).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to deserialize WASI component: {}",
                            e
                        ))
                    })
                }
            },
        );

        let component = match component_result {
            Ok(entry) => entry.value().clone(),
            Err(e) => return Err(e),
        };

        self.run_component_wasi(component, tool_name, args_json, env)
    }

    /// Compile and run a Wasm component (requires cranelift feature).
    /// wasm_bytes: Raw .wasm component bytes.
    #[cfg(feature = "cranelift")]
    pub fn execute(
        &self,
        name: String,
        wasm_bytes: Vec<u8>,
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        let component_result = self.component_cache.entry(name.clone()).or_try_insert_with(
            || -> PyResult<Component> {
                Component::new(&self.engine, &wasm_bytes).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to compile Wasm component: {}",
                        e
                    ))
                })
            },
        );

        let component = match component_result {
            Ok(entry) => entry.value().clone(),
            Err(e) => return Err(e),
        };

        self.run_component(component, tool_name, args_json, env)
    }

    /// Fallback when cranelift is not available — returns a clear error.
    #[cfg(not(feature = "cranelift"))]
    pub fn execute(
        &self,
        _name: String,
        _wasm_bytes: Vec<u8>,
        _tool_name: String,
        _args_json: String,
        _env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "JIT compilation requires the 'cranelift' feature. \
             Use execute_precompiled() with pre-compiled components, \
             or build with: cargo build --features sandbox,cranelift",
        ))
    }
}

impl WasmSandbox {
    /// Run a component WITHOUT WASI imports (for simple components like the
    /// Phase 2 expression evaluator that don't need WASI).
    fn run_component(
        &self,
        component: Component,
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);

        let instance = ToolEnv::instantiate(&mut store, &component, &linker).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Wasm instantiation error: {}",
                e
            ))
        })?;

        let input = ToolInput {
            name: tool_name,
            args: args_json,
            env: env.into_iter().collect(),
        };

        let output = instance.call_run(&mut store, &input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Wasm execution error: {}",
                e
            ))
        })?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("stdout", output.stdout)?;
            dict.set_item("stderr", output.stderr)?;
            dict.set_item("exit_code", output.exit_code)?;
            dict.set_item("result", output.result_json)?;
            Ok(dict.into())
        })
    }

    /// Run a component WITH WASI imports (for CPython WASI components).
    /// Capabilities are deny-by-default: no filesystem, no env vars,
    /// no network, no subprocess. Only stdout/stderr inherited.
    fn run_component_wasi(
        &self,
        component: Component,
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>,
    ) -> PyResult<PyObject> {
        let wasi_state = WasiState::new_restrictive();
        let mut store = Store::new(&self.engine, wasi_state);

        let mut linker = Linker::<WasiState>::new(&self.engine);

        // Add WASI p2 imports to the linker (sync version for non-async execution).
        // This provides the WASI host functions the CPython component needs to initialize.
        wasmtime_wasi::p2::add_to_linker_sync(&mut linker).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to add WASI imports to linker: {}",
                e
            ))
        })?;

        let instance = ToolEnv::instantiate(&mut store, &component, &linker).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "WASI component instantiation error: {}",
                e
            ))
        })?;

        let input = ToolInput {
            name: tool_name,
            args: args_json,
            env: env.into_iter().collect(),
        };

        let output = instance.call_run(&mut store, &input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "WASI component execution error: {}",
                e
            ))
        })?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("stdout", output.stdout)?;
            dict.set_item("stderr", output.stderr)?;
            dict.set_item("exit_code", output.exit_code)?;
            dict.set_item("result", output.result_json)?;
            Ok(dict.into())
        })
    }
}

/// Standalone WASI-aware execution for use by ToolExecutor.
/// Takes pre-deserialized component bytes (pre-compiled) and runs with WASI imports.
/// Returns (stdout, stderr, exit_code) or error.
pub fn execute_wasi_component(
    engine: &Engine,
    component: &Component,
    tool_name: &str,
    args_json: &str,
) -> Result<(String, String, i32), String> {
    let wasi_state = WasiState::new_restrictive();
    let mut store = Store::new(engine, wasi_state);

    let mut linker = Linker::<WasiState>::new(engine);
    wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
        .map_err(|e| format!("WASI linker: {}", e))?;

    let instance = ToolEnv::instantiate(&mut store, component, &linker)
        .map_err(|e| format!("WASI instantiate: {}", e))?;

    let input = ToolInput {
        name: tool_name.to_string(),
        args: args_json.to_string(),
        env: Vec::new(),
    };

    let output = instance
        .call_run(&mut store, &input)
        .map_err(|e| format!("WASI call_run: {}", e))?;

    Ok((output.stdout, output.stderr, output.exit_code))
}

/// Standalone non-WASI execution for use by ToolExecutor.
/// For simple components that don't need WASI imports.
pub fn execute_bare_component(
    engine: &Engine,
    component: &Component,
    tool_name: &str,
    args_json: &str,
) -> Result<(String, String, i32), String> {
    let mut store = Store::new(engine, ());
    let linker = Linker::<()>::new(engine);

    let instance = ToolEnv::instantiate(&mut store, component, &linker)
        .map_err(|e| format!("Instantiate: {}", e))?;

    let input = ToolInput {
        name: tool_name.to_string(),
        args: args_json.to_string(),
        env: Vec::new(),
    };

    let output = instance
        .call_run(&mut store, &input)
        .map_err(|e| format!("call_run: {}", e))?;

    Ok((output.stdout, output.stderr, output.exit_code))
}
