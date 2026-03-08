# Rust ONNX Embedder in sage-core — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a native ONNX Runtime embedder to sage-core (behind an optional `onnx` feature flag) that replaces the Python hash fallback with real 384-dim embeddings, callable from Python via PyO3.

**Architecture:** The `ort` crate (ONNX Runtime Rust bindings) loads a pre-exported all-MiniLM-L6-v2 ONNX model at init time. A `RustEmbedder` PyO3 class exposes `embed(text) -> list[float]` and `embed_batch(texts) -> list[list[float]]`. The Python `Embedder` adapter auto-detects `sage_core.RustEmbedder` availability and delegates to it, falling back to sentence-transformers then hash. The entire ort dependency is behind `features = ["onnx"]` so the default build is unchanged.

**Tech Stack:** Rust (ort 2.0.0-rc.12, tokenizers 0.21, ndarray 0.16), PyO3 0.25, Python 3.12+

**Oracle Consensus (Gemini 3.1 Pro + GPT-5.4 + Context7):**
- ort + PyO3 0.25 in same cdylib: no linking conflict (confirmed)
- ndarray version mismatch: ort may use 0.17, numpy uses 0.16 — bridge via slices (confirmed)
- Windows System32 DLL gotcha: use `os.add_dll_directory()` in Python (confirmed)
- Model: all-MiniLM-L6-v2 has `onnx/model.onnx` on HuggingFace (confirmed)
- Mean pooling + L2 norm: must be implemented manually in Rust (confirmed)

---

## Task 1: Add ort + tokenizers to Cargo.toml (optional feature)

**Files:**
- Modify: `sage-core/Cargo.toml`

**Step 1: Add dependencies behind onnx feature**

```toml
# In [dependencies]:
ort = { version = "2.0.0-rc.12", optional = true, features = ["copy-dylibs"] }
tokenizers = { version = "0.21", optional = true, default-features = false, features = ["progressbar"] }

# In [features]:
onnx = ["dep:ort", "dep:tokenizers"]
```

**Step 2: Verify it compiles without the feature**

Run: `cd sage-core && cargo check --no-default-features`
Expected: PASS (ort not compiled)

**Step 3: Verify it compiles WITH the feature**

Run: `cd sage-core && cargo check --features onnx`
Expected: PASS (ort downloaded and linked)

**Step 4: Commit**

```bash
git add sage-core/Cargo.toml
git commit -m "build(core): add ort + tokenizers as optional onnx feature"
```

---

## Task 2: Implement RustEmbedder in sage-core

**Files:**
- Create: `sage-core/src/memory/embedder.rs`
- Modify: `sage-core/src/memory/mod.rs` (add module)
- Modify: `sage-core/src/lib.rs` (expose to PyO3)

**Step 1: Write the Rust embedder module**

```rust
// sage-core/src/memory/embedder.rs
//! ONNX Runtime embedder for S-MMU semantic edges.
//!
//! Behind the `onnx` feature flag. Loads all-MiniLM-L6-v2 ONNX model
//! and produces 384-dim L2-normalized embeddings.

#[cfg(feature = "onnx")]
use ort::{inputs, session::Session, value::TensorRef};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;
#[cfg(feature = "onnx")]
use pyo3::prelude::*;
#[cfg(feature = "onnx")]
use std::path::PathBuf;

#[cfg(feature = "onnx")]
const EMBEDDING_DIM: usize = 384;

#[cfg(feature = "onnx")]
#[pyclass]
pub struct RustEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl RustEmbedder {
    /// Create a new RustEmbedder from ONNX model and tokenizer paths.
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path))]
    pub fn new(model_path: String, tokenizer_path: String) -> PyResult<Self> {
        let session = Session::builder()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("ORT session builder error: {e}")
            ))?
            .commit_from_file(&model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("ORT model load error: {e}")
            ))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Tokenizer load error: {e}")
            ))?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text, returns list of 384 floats (L2-normalized).
    pub fn embed(&self, text: &str) -> PyResult<Vec<f32>> {
        let batch = self.embed_batch(vec![text.to_string()])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts, returns list of list of 384 floats.
    pub fn embed_batch(&self, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Tokenize
        let encodings = self.tokenizer.encode_batch(str_refs, true)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Tokenization error: {e}")
            ))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build input tensors
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

        // Run ONNX inference
        let id_shape = [batch_size, max_len];
        let id_tensor = TensorRef::from_array_view(
            &ndarray::ArrayView2::from_shape(id_shape, &input_ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Shape error: {e}")
                ))?
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Tensor error: {e}")
        ))?;

        let mask_tensor = TensorRef::from_array_view(
            &ndarray::ArrayView2::from_shape(id_shape, &attention_mask)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("Shape error: {e}")
                ))?
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Tensor error: {e}")
        ))?;

        let outputs = self.session.run(inputs![
            "input_ids" => id_tensor,
            "attention_mask" => mask_tensor
        ].map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("Input error: {e}")
        ))?)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("ORT inference error: {e}")
        ))?;

        // Extract token embeddings [batch, seq_len, 384]
        let token_emb = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Output extraction error: {e}")
            ))?;
        let (shape, data) = token_emb;

        // Mean pooling + L2 normalization
        let hidden_dim = if shape.len() == 3 { shape[2] } else { EMBEDDING_DIM };
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_dim];
            let mut mask_sum = 0.0f32;

            for j in 0..max_len {
                let m = attention_mask[i * max_len + j] as f32;
                mask_sum += m;
                let offset = i * max_len * hidden_dim + j * hidden_dim;
                for k in 0..hidden_dim {
                    pooled[k] += data[offset + k] * m;
                }
            }

            // Mean
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

    /// Embedding dimension (always 384 for all-MiniLM-L6-v2).
    #[getter]
    pub fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Whether this is a semantic embedder (always true for ONNX).
    #[getter]
    pub fn is_semantic(&self) -> bool {
        true
    }
}
```

**Step 2: Register module in mod.rs and lib.rs**

In `sage-core/src/memory/mod.rs`, add:
```rust
#[cfg(feature = "onnx")]
pub mod embedder;
```

In `sage-core/src/lib.rs`, inside the `#[pymodule]` function, add:
```rust
#[cfg(feature = "onnx")]
m.add_class::<memory::embedder::RustEmbedder>()?;
```

**Step 3: Verify compilation**

Run: `cd sage-core && cargo check --features onnx`
Expected: PASS

Run: `cd sage-core && cargo check --no-default-features`
Expected: PASS (onnx not compiled, no errors)

**Step 4: Commit**

```bash
git add sage-core/src/memory/embedder.rs sage-core/src/memory/mod.rs sage-core/src/lib.rs
git commit -m "feat(core): add RustEmbedder via ort ONNX Runtime (behind onnx feature)"
```

---

## Task 3: Download and bundle ONNX model + tokenizer

**Files:**
- Create: `sage-core/models/download_model.py` (one-time script)
- Create: `sage-core/models/.gitkeep`

**Step 1: Write download script**

```python
#!/usr/bin/env python3
"""Download all-MiniLM-L6-v2 ONNX model and tokenizer for sage-core."""
from huggingface_hub import hf_hub_download
import os

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

files = {
    "onnx/model.onnx": "model.onnx",
    "tokenizer.json": "tokenizer.json",
}

for hf_path, local_name in files.items():
    local_path = os.path.join(OUT_DIR, local_name)
    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        continue
    print(f"Downloading {hf_path}...")
    hf_hub_download(
        repo_id=MODEL_ID,
        filename=hf_path,
        local_dir=OUT_DIR,
        local_dir_use_symlinks=False,
    )
    # Flatten: move from subdir if needed
    src = os.path.join(OUT_DIR, hf_path)
    if os.path.exists(src) and src != local_path:
        os.rename(src, local_path)
        print(f"  -> {local_path}")

print("Done. Files in:", OUT_DIR)
```

**Step 2: Add models/ to .gitignore** (large binaries)

```
# In sage-core/.gitignore or root .gitignore:
sage-core/models/*.onnx
sage-core/models/tokenizer.json
```

**Step 3: Run download**

Run: `cd sage-core/models && python download_model.py`
Expected: `model.onnx` (~91 MB) and `tokenizer.json` (~700 KB) downloaded

**Step 4: Commit**

```bash
git add sage-core/models/download_model.py sage-core/models/.gitkeep .gitignore
git commit -m "build(core): add ONNX model download script for all-MiniLM-L6-v2"
```

---

## Task 4: Rust unit tests for RustEmbedder

**Files:**
- Create: `sage-core/tests/test_embedder.rs`

**Step 1: Write tests (gated behind onnx feature)**

```rust
// sage-core/tests/test_embedder.rs
#[cfg(feature = "onnx")]
mod onnx_embedder_tests {
    use sage_core::memory::embedder::RustEmbedder;

    fn model_path() -> String {
        let base = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        format!("{}/models/model.onnx", base)
    }

    fn tokenizer_path() -> String {
        let base = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        format!("{}/models/tokenizer.json", base)
    }

    fn skip_if_no_model() -> bool {
        !std::path::Path::new(&model_path()).exists()
    }

    #[test]
    fn test_embed_single() {
        if skip_if_no_model() { return; }
        let emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vec = emb.embed("Hello world").unwrap();
        assert_eq!(vec.len(), 384);
        // L2 normalized: norm should be ~1.0
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embed_batch() {
        if skip_if_no_model() { return; }
        let emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vecs = emb.embed_batch(vec![
            "Hello".into(), "World".into(), "Rust is fast".into()
        ]).unwrap();
        assert_eq!(vecs.len(), 3);
        assert!(vecs.iter().all(|v| v.len() == 384));
    }

    #[test]
    fn test_embed_deterministic() {
        if skip_if_no_model() { return; }
        let emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let v1 = emb.embed("deterministic test").unwrap();
        let v2 = emb.embed("deterministic test").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_embed_empty_batch() {
        if skip_if_no_model() { return; }
        let emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let vecs = emb.embed_batch(vec![]).unwrap();
        assert!(vecs.is_empty());
    }

    #[test]
    fn test_similar_texts_closer() {
        if skip_if_no_model() { return; }
        let emb = RustEmbedder::new(model_path(), tokenizer_path()).unwrap();
        let cat = emb.embed("I love cats").unwrap();
        let dog = emb.embed("I love dogs").unwrap();
        let code = emb.embed("fn main() { println!(\"hello\"); }").unwrap();

        let sim_cat_dog: f32 = cat.iter().zip(&dog).map(|(a, b)| a * b).sum();
        let sim_cat_code: f32 = cat.iter().zip(&code).map(|(a, b)| a * b).sum();

        // "cats" and "dogs" should be more similar than "cats" and "code"
        assert!(sim_cat_dog > sim_cat_code,
            "Expected cat-dog ({:.3}) > cat-code ({:.3})", sim_cat_dog, sim_cat_code);
    }
}
```

**Step 2: Run tests**

Run: `cd sage-core && cargo test --features onnx -- onnx_embedder`
Expected: 5 tests pass (or skip if model not downloaded)

**Step 3: Commit**

```bash
git add sage-core/tests/test_embedder.rs
git commit -m "test(core): add RustEmbedder unit tests (ONNX feature-gated)"
```

---

## Task 5: Wire Python Embedder to auto-detect RustEmbedder

**Files:**
- Modify: `sage-python/src/sage/memory/embedder.py`
- Test: `sage-python/tests/test_embedder_rust_fallback.py`

**Step 1: Write the failing test**

```python
# sage-python/tests/test_embedder_rust_fallback.py
import pytest
from unittest.mock import MagicMock, patch

def test_embedder_prefers_rust_when_available():
    """When sage_core.RustEmbedder exists, Embedder should use it."""
    mock_rust = MagicMock()
    mock_rust.embed.return_value = [0.1] * 384
    mock_rust.embed_batch.return_value = [[0.1] * 384]
    mock_rust.is_semantic = True

    with patch("sage.memory.embedder._try_rust_embedder", return_value=mock_rust):
        from sage.memory.embedder import Embedder
        emb = Embedder()
        # Should delegate to Rust
        vec = emb.embed("test")
        assert len(vec) == 384

def test_embedder_falls_back_without_rust():
    """When sage_core.RustEmbedder is unavailable, falls back to hash."""
    with patch("sage.memory.embedder._try_rust_embedder", return_value=None):
        from sage.memory.embedder import Embedder
        emb = Embedder(force_hash=True)
        vec = emb.embed("test")
        assert len(vec) == 384
```

**Step 2: Add auto-detection to embedder.py**

Add at the top of `embedder.py`:

```python
def _try_rust_embedder() -> Any | None:
    """Try to create a RustEmbedder from sage_core (ONNX feature)."""
    try:
        import sage_core
        if not hasattr(sage_core, "RustEmbedder"):
            return None
        # Look for model files
        import importlib.resources
        from pathlib import Path
        model_dirs = [
            Path.home() / ".sage" / "models",
            Path(__file__).parent.parent.parent.parent.parent / "sage-core" / "models",
        ]
        for d in model_dirs:
            model_path = d / "model.onnx"
            tokenizer_path = d / "tokenizer.json"
            if model_path.exists() and tokenizer_path.exists():
                return sage_core.RustEmbedder(str(model_path), str(tokenizer_path))
        return None
    except Exception:
        return None
```

Update `Embedder.__init__` to try Rust first:

```python
def __init__(self, force_hash: bool = False):
    if force_hash:
        self._backend = _HashEmbedder()
        self._is_semantic = False
        return

    # Priority: RustEmbedder (ONNX) > sentence-transformers > hash
    rust = _try_rust_embedder()
    if rust is not None:
        self._backend = rust
        self._is_semantic = True
        log.info("Embedder: using RustEmbedder (ONNX, native)")
        return

    try:
        self._backend = _SentenceTransformerEmbedder()
        self._is_semantic = True
        log.info("Embedder: using sentence-transformers (semantic)")
    except ImportError:
        self._backend = _HashEmbedder()
        self._is_semantic = False
        log.warning("No embedding backend available — using hash fallback")
```

**Step 3: Run tests**

Run: `cd sage-python && python -m pytest tests/test_embedder_rust_fallback.py tests/test_embedder.py -v`
Expected: All pass

**Step 4: Commit**

```bash
git add sage-python/src/sage/memory/embedder.py sage-python/tests/test_embedder_rust_fallback.py
git commit -m "feat(memory): Embedder auto-detects RustEmbedder (ONNX) with 3-tier fallback"
```

---

## Task 6: Windows DLL safety + CI update

**Files:**
- Modify: `sage-python/src/sage/memory/working.py` (add DLL directory for Windows)
- Modify: `.github/workflows/ci.yml` (add onnx feature test job)

**Step 1: Add DLL safety to working.py**

At the very top of `working.py`, before `import sage_core`:

```python
import sys, os
if sys.platform == "win32":
    # Ensure ONNX Runtime DLL is found before System32 fallback
    _ort_dll_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "sage-core", "target", "debug"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "sage-core", "target", "release"),
    ]
    for d in _ort_dll_dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
```

**Step 2: Add CI job for onnx feature**

In `.github/workflows/ci.yml`, add after the existing Rust test step:

```yaml
    - name: Rust tests (with onnx feature)
      if: runner.os == 'Linux'  # Only on Linux for now
      run: |
        cd sage-core
        cargo test --features onnx --no-default-features 2>&1 || echo "ONNX tests skipped (model not downloaded)"
```

**Step 3: Commit**

```bash
git add sage-python/src/sage/memory/working.py .github/workflows/ci.yml
git commit -m "build: Windows DLL safety for ONNX Runtime + CI onnx feature job"
```

---

## Task 7: Update documentation + MEMORY.md

**Files:**
- Modify: `CLAUDE.md` (add RustEmbedder, onnx feature)
- Modify: `ARCHITECTURE.md` (add ONNX embedder to memory section)

Key additions:
- New optional feature `onnx` in Cargo.toml
- `RustEmbedder` PyO3 class: loads all-MiniLM-L6-v2 ONNX, 384-dim, L2-normalized
- 3-tier fallback: RustEmbedder (ONNX) > sentence-transformers > hash
- Model download: `python sage-core/models/download_model.py`
- Build with ONNX: `cd sage-core && maturin develop --features onnx`

**Commit:**

```bash
git add CLAUDE.md ARCHITECTURE.md
git commit -m "docs: document RustEmbedder ONNX feature + 3-tier embedding fallback"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Cargo.toml: ort + tokenizers optional | `Cargo.toml` |
| 2 | RustEmbedder implementation | `embedder.rs`, `mod.rs`, `lib.rs` |
| 3 | ONNX model download script | `models/download_model.py` |
| 4 | Rust unit tests (feature-gated) | `tests/test_embedder.rs` |
| 5 | Python Embedder 3-tier fallback | `embedder.py` + test |
| 6 | Windows DLL safety + CI | `working.py`, `ci.yml` |
| 7 | Documentation | `CLAUDE.md`, `ARCHITECTURE.md` |

**Total: 7 tasks, ~7 commits**

**Dependencies:** Tasks 1→2→3→4 sequential. Task 5 depends on 2. Tasks 6-7 depend on all.
