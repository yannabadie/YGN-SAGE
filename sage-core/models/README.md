# models/

ONNX model assets for the RustEmbedder (behind the `onnx` feature flag). These files are required only when using the native Rust embedding path.

## Files

### `download_model.py`

Downloads the all-MiniLM-L6-v2 ONNX model and tokenizer from HuggingFace Hub:

```bash
python sage-core/models/download_model.py
```

Downloads two files into this directory:
- `model.onnx` -- ONNX model for sentence embeddings (384-dim output).
- `tokenizer.json` -- HuggingFace tokenizer configuration.

Requires the `huggingface_hub` Python package. Cleans up intermediate subdirectories created by `hf_hub_download`.

### `tokenizer.json`

HuggingFace tokenizer configuration for all-MiniLM-L6-v2. Used by the `tokenizers` Rust crate to tokenize input text before ONNX inference.

### `model.onnx` (gitignored)

The ONNX model file (~90 MB). Not checked into version control. Download via `download_model.py`.

### `onnxruntime.dll` (gitignored)

ONNX Runtime shared library for Windows. Automatically copied by the `ort` crate's `copy-dylibs` feature during build. Not checked into version control.

## gitignore Rules

The following patterns are in the root `.gitignore`:

```
sage-core/models/*.onnx
sage-core/models/tokenizer.json
sage-core/models/*.dll
```

## Model Details

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Output dimensionality**: 384
- **Normalization**: L2-normalized embeddings
- **Inference**: Mean pooling over token outputs with attention mask, followed by L2 normalization
- **Usage**: S-MMU semantic edges (cosine similarity > 0.5 threshold)
