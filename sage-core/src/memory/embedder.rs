//! ONNX Runtime embedder for S-MMU semantic edges.
//!
//! Behind the `onnx` feature flag. Loads snowflake-arctic-embed-m ONNX model
//! and produces 768-dim L2-normalized embeddings.
//!
//! Uses `load-dynamic` strategy: onnxruntime DLL is loaded at runtime
//! via `ort::init_from()` (ort 2.x API) or `ORT_DYLIB_PATH` env var.
//! This avoids static linking issues on Windows MSVC (LNK1120).

use ort::{inputs, session::Session, value::TensorRef};
use pyo3::prelude::*;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

const EMBEDDING_DIM: usize = 768;

/// Cached ORT dylib path — resolved once, reused everywhere.
static ORT_DYLIB_RESOLVED: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();

/// Resolve and cache the ORT dylib path (thread-safe, one-time).
/// Returns a reference to the cached path if found.
pub fn resolve_ort_dylib_once(model_path: &str, sys_prefix: Option<&str>) -> Option<&'static std::path::PathBuf> {
    let resolved = ORT_DYLIB_RESOLVED.get_or_init(|| {
        if std::env::var("ORT_DYLIB_PATH").is_ok() {
            return None; // User already set it
        }
        discover_ort_dylib(model_path, sys_prefix)
    });
    resolved.as_ref()
}

/// Auto-discover the ONNX Runtime shared library on the filesystem.
///
/// Search order: sibling of model > sys.prefix > VIRTUAL_ENV >
/// user site-packages (%APPDATA%) > system Python (C:\Python3*).
pub(crate) fn discover_ort_dylib(model_path: &str, sys_prefix: Option<&str>) -> Option<std::path::PathBuf> {
    #[cfg(target_os = "windows")]
    let dll_name = "onnxruntime.dll";
    #[cfg(target_os = "linux")]
    let dll_name = "libonnxruntime.so";
    #[cfg(target_os = "macos")]
    let dll_name = "libonnxruntime.dylib";

    // 1. Sibling of model file.
    if let Some(parent) = std::path::Path::new(model_path).parent() {
        let candidate = parent.join(dll_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Platform-specific search paths (Windows pip installs).
    #[cfg(target_os = "windows")]
    {
        let capi_tail = std::path::Path::new("Lib")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi")
            .join(dll_name);

        // 2. Python sys.prefix (works inside venv even without `activate`).
        // This is the most reliable path when running from a venv.
        if let Some(prefix) = sys_prefix {
            let candidate = std::path::Path::new(prefix).join(&capi_tail);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        // 3. VIRTUAL_ENV env var (set by `activate` scripts)
        if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
            let candidate = std::path::Path::new(&venv).join(&capi_tail);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        // 4. User site-packages: %APPDATA%\Python\Python3*\site-packages
        if let Ok(appdata) = std::env::var("APPDATA") {
            let user_python = std::path::Path::new(&appdata).join("Python");
            if let Ok(entries) = std::fs::read_dir(&user_python) {
                let user_capi_tail = std::path::Path::new("site-packages")
                    .join("onnxruntime")
                    .join("capi")
                    .join(dll_name);
                let mut dirs: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.file_name()
                            .to_str()
                            .is_some_and(|n| n.starts_with("Python3"))
                    })
                    .collect();
                dirs.sort_by_key(|b| std::cmp::Reverse(b.file_name()));
                for entry in dirs {
                    let candidate = entry.path().join(&user_capi_tail);
                    if candidate.exists() {
                        return Some(candidate);
                    }
                }
            }
        }

        // 5. System Python: scan C:\Python3* directories
        if let Ok(entries) = std::fs::read_dir("C:\\") {
            let mut pythons: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("Python3"))
                })
                .collect();
            pythons.sort_by_key(|b| std::cmp::Reverse(b.file_name()));
            for entry in pythons {
                let candidate = entry.path().join(&capi_tail);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
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
    pub fn new(py: Python<'_>, model_path: String, tokenizer_path: String) -> PyResult<Self> {
        // Resolve ORT dylib path once (thread-safe via OnceLock).
        // This MUST happen before Session::builder() because ort caches
        // its DLL handle in a OnceLock — once read, it never re-checks env.
        // Python side (Embedder._ensure_ort_dylib_path) also does this,
        // but this covers direct `sage_core.RustEmbedder(...)` usage.
        let sys_prefix: Option<String> = py
            .import("sys")
            .ok()
            .and_then(|sys| sys.getattr("prefix").ok())
            .and_then(|p| p.extract().ok());
        if let Some(path) = resolve_ort_dylib_once(&model_path, sys_prefix.as_deref()) {
            if std::env::var("ORT_DYLIB_PATH").is_err() {
                std::env::set_var("ORT_DYLIB_PATH", path);
            }
        }
        // Release the GIL before loading ORT — on Windows, LoadLibraryW runs
        // onnxruntime.dll's DllMain which may attempt GIL acquisition, causing
        // deadlock if we're still holding it from this #[pymethods] call.
        let (session, tokenizer) = py.allow_threads(|| -> Result<(Session, Tokenizer), String> {
            let session = Session::builder()
                .map_err(|e| format!("ORT session builder error: {e}"))?
                .commit_from_file(&model_path)
                .map_err(|e| format!("ORT model load error: {e}"))?;

            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("Tokenizer load error: {e}"))?;

            Ok((session, tokenizer))
        }).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text string. Returns a 768-dim L2-normalized vector.
    pub fn embed(&mut self, text: &str) -> PyResult<Vec<f32>> {
        let batch = self.embed_batch(vec![text.to_string()])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of text strings. Returns a list of 768-dim L2-normalized vectors.
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

        // Check if the ONNX model expects token_type_ids (standard BERT input).
        // Some models (e.g. snowflake-arctic-embed-m) may not include it.
        let has_token_type = self.session.inputs.iter().any(|input| input.name == "token_type_ids");

        let session_inputs = if has_token_type {
            let token_type_ids = vec![0i64; batch_size * max_len];
            let type_tensor =
                TensorRef::from_array_view((shape, &*token_type_ids)).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error (type_ids): {e}"))
                })?;
            inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor
            ]
        } else {
            inputs![
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor
            ]
        };

        let outputs = self.session.run(session_inputs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("ORT inference error: {e}"))
        })?;

        // Extract output — shape is [batch, seq_len, hidden_dim] (768 for arctic-embed-m)
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

    /// Embedding dimensionality (768 for snowflake-arctic-embed-m).
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
