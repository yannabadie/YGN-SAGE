use pyo3::prelude::*;
use wasmtime::component::*;
use wasmtime::{Config, Engine, Store};
use std::collections::HashMap;

// Generate host bindings from our WIT file
bindgen!({
    world: "tool-env",
    path: "interface.wit",
});

/// SOTA 2026: Component-based Wasm sandbox for lightning-fast tool execution.
/// Uses the WASI 0.2/0.3 standard for secure, zero-copy data exchange.
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
        config.wasm_component_model(true); // Enable SOTA 2026 Component Model
        config.cranelift_opt_level(wasmtime::OptLevel::Speed);
        config.async_support(false); // Keep it sync for low-latency host calls
        
        let engine = Engine::new(&config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Wasmtime Engine error: {}", e)))?;
            
        Ok(Self {
            engine,
            component_cache: dashmap::DashMap::new(),
        })
    }

    /// Run a Wasm tool with structured input/output.
    /// name: Module identifier for caching
    /// wasm_bytes: The compiled .wasm component
    /// tool_name: Name of the tool to invoke
    /// args_json: Arguments as a JSON string
    /// env: Environment variables
    pub fn execute(
        &self, 
        name: String, 
        wasm_bytes: Vec<u8>, 
        tool_name: String,
        args_json: String,
        env: HashMap<String, String>
    ) -> PyResult<PyObject> {
        let component_result = self.component_cache.entry(name.clone()).or_try_insert_with(|| -> PyResult<Component> {
            Component::new(&self.engine, &wasm_bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to compile Wasm component: {}", e)))
        });
        
        let component = match component_result {
            Ok(entry) => entry.value().clone(),
            Err(e) => return Err(e),
        };

        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);
        
        // Instantiate the component
        let instance = ToolEnv::instantiate(&mut store, &component, &linker)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Wasm instantiation error: {}", e)))?;
            
        // Prepare input record
        let input = ToolInput {
            name: tool_name,
            args: args_json,
            env: env.into_iter().collect(),
        };

        // Call the exported function (defined in interface.wit)
        let output = instance.call_run(&mut store, &input)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Wasm execution error: {}", e)))?;
            
        // Return result as a Python dict for easy integration
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
