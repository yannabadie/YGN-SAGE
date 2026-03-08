//! ONNX Runtime embedder for S-MMU semantic edges.
//!
//! Behind the `onnx` feature flag. Loads all-MiniLM-L6-v2 ONNX model
//! and produces 384-dim L2-normalized embeddings.
//!
//! Uses `load-dynamic` strategy: onnxruntime DLL is loaded at runtime
//! via `ORT_DYLIB_PATH` env var or auto-discovered next to the model.
//! This avoids static linking issues on Windows MSVC (LNK1120).

use ort::{inputs, session::Session, value::TensorRef};
use pyo3::prelude::*;
use std::sync::Once;
use tokenizers::Tokenizer;

const EMBEDDING_DIM: usize = 384;

static ORT_INIT: Once = Once::new();

/// Initialize ORT runtime from a dylib path. Must be called before any Session usage.
/// Searches: ORT_DYLIB_PATH env var > sibling of model_path > common venv paths.
///
/// NOTE: Do NOT call Python subprocess from here — this runs inside a PyO3 call,
/// and spawning a new Python process would deadlock on the GIL.
fn ensure_ort_initialized(model_path: &str) {
    ORT_INIT.call_once(|| {
        // If ORT_DYLIB_PATH is set, ort will use it automatically.
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            return;
        }

        #[cfg(target_os = "windows")]
        let dll_name = "onnxruntime.dll";
        #[cfg(target_os = "linux")]
        let dll_name = "libonnxruntime.so";
        #[cfg(target_os = "macos")]
        let dll_name = "libonnxruntime.dylib";

        // Check sibling of model file first.
        if let Some(parent) = std::path::Path::new(model_path).parent() {
            let candidate = parent.join(dll_name);
            if candidate.exists() {
                std::env::set_var("ORT_DYLIB_PATH", &candidate);
                return;
            }
        }

        // On Windows, try the pip-installed onnxruntime capi directory.
        // Walk up from model_path to find a .venv or venv with onnxruntime installed.
        #[cfg(target_os = "windows")]
        {
            // Try sys.prefix-based discovery via VIRTUAL_ENV env var
            if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
                let candidate = std::path::Path::new(&venv)
                    .join("Lib")
                    .join("site-packages")
                    .join("onnxruntime")
                    .join("capi")
                    .join(dll_name);
                if candidate.exists() {
                    std::env::set_var("ORT_DYLIB_PATH", &candidate);
                }
            }
        }
    });
}

#[pyclass]
pub struct RustEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

#[pymethods]
impl RustEmbedder {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path))]
    pub fn new(model_path: String, tokenizer_path: String) -> PyResult<Self> {
        // Initialize ORT dynamic library before creating session
        ensure_ort_initialized(&model_path);

        let session = Session::builder()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "ORT session builder error: {e}"
                ))
            })?
            .commit_from_file(&model_path)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("ORT model load error: {e}"))
            })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenizer load error: {e}"))
        })?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text string. Returns a 384-dim L2-normalized vector.
    pub fn embed(&mut self, text: &str) -> PyResult<Vec<f32>> {
        let batch = self.embed_batch(vec![text.to_string()])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of text strings. Returns a list of 384-dim L2-normalized vectors.
    pub fn embed_batch(&mut self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        let encodings = self
            .tokenizer
            .encode_batch(str_refs, true)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenization error: {e}"))
            })?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Flatten token IDs and attention masks into contiguous buffers
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids[i * max_len + j] = id as i64;
            }
            for (j, &mask) in enc.get_attention_mask().iter().enumerate() {
                attention_mask[i * max_len + j] = mask as i64;
            }
        }

        // token_type_ids: all zeros (single-sentence, required by BERT-like models)
        let token_type_ids = vec![0i64; batch_size * max_len];

        // Build tensors using (shape, &[T]) tuple form accepted by ort 2.x
        let shape = vec![batch_size, max_len];

        let id_tensor =
            TensorRef::from_array_view((shape.clone(), &*input_ids)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error (ids): {e}"))
            })?;

        let mask_tensor =
            TensorRef::from_array_view((shape.clone(), &*attention_mask)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error (mask): {e}"))
            })?;

        let type_tensor =
            TensorRef::from_array_view((shape, &*token_type_ids)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error (type_ids): {e}"))
            })?;

        let session_inputs = inputs![
            "input_ids" => id_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_tensor
        ];

        let outputs = self.session.run(session_inputs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ORT inference error: {e}"))
        })?;

        // Extract output — for all-MiniLM-L6-v2 shape is [batch, seq_len, 384]
        // try_extract_tensor returns (&Shape, &[f32]) where Shape derefs to [i64]
        let (out_shape, out_data) =
            outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Output extraction error: {e}"
                ))
            })?;

        // Determine dimensions from the output shape
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();
        let hidden_dim = if dims.len() == 3 { dims[2] } else { EMBEDDING_DIM };

        // Mean pooling with attention mask + L2 normalization
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_dim];
            let mut mask_sum = 0.0f32;

            for j in 0..max_len {
                let m = attention_mask[i * max_len + j] as f32;
                mask_sum += m;
                // Index into flat output: out_data[i * max_len * hidden_dim + j * hidden_dim + k]
                let offset = i * max_len * hidden_dim + j * hidden_dim;
                for k in 0..hidden_dim {
                    pooled[k] += out_data[offset + k] * m;
                }
            }

            // Mean pool
            if mask_sum > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= mask_sum;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }

            results.push(pooled);
        }

        Ok(results)
    }

    /// Embedding dimensionality (384 for all-MiniLM-L6-v2).
    #[getter]
    pub fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Whether this embedder produces semantic (not hash-based) embeddings.
    #[getter]
    pub fn is_semantic(&self) -> bool {
        true
    }
}
