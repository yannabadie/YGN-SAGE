use pyo3::prelude::*;
use std::collections::HashMap;
use wasmtime::component::*;
use wasmtime::{Config, Engine, Store};

// Generate host bindings from our WIT file
bindgen!({
    world: "tool-env",
    path: "interface.wit",
});

/// Component-based Wasm sandbox for structured tool execution.
/// Uses the Component Model for typed I/O (WIT-defined tool-env world).
///
/// On Windows (no cranelift feature), only pre-compiled components are
/// supported via `execute_precompiled()`. On Linux CI with cranelift,
/// `execute()` compiles Wasm on the fly.
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
        let component_result = self.component_cache.entry(name.clone()).or_try_insert_with(
            || -> PyResult<Component> {
                // SAFETY: compiled_bytes must come from Component::serialize()
                // produced by the same version of wasmtime with the same Engine config.
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
             or build with: cargo build --features sandbox,cranelift"
        ))
    }
}

impl WasmSandbox {
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
}
