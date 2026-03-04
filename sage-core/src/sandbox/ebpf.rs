use pyo3::prelude::*;
use solana_rbpf::vm::Config;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EbpfError {
    #[error("Execution failed: {0}")]
    ExecutionError(String),
}

impl From<EbpfError> for PyErr {
    fn from(err: EbpfError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
    }
}

#[pyclass]
pub struct EbpfSandbox {
    _config: Config,
}

#[pymethods]
impl EbpfSandbox {
    #[new]
    pub fn new() -> Self {
        Self {
            _config: Config::default(),
        }
    }

    pub fn execute(&self, _program_data: &[u8], _input: Vec<u8>) -> PyResult<u64> {
        Ok(0)
    }
}
