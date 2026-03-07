#!/usr/bin/env python3
"""Download all-MiniLM-L6-v2 ONNX model and tokenizer for sage-core."""
from huggingface_hub import hf_hub_download
import os
import shutil

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
    )
    # Flatten: move from subdir if needed
    src = os.path.join(OUT_DIR, hf_path)
    if os.path.exists(src) and src != local_path:
        os.rename(src, local_path)
        print(f"  -> {local_path}")

# Clean up subdirectories created by hf_hub_download
for subdir in ("onnx", ".huggingface", ".cache"):
    path = os.path.join(OUT_DIR, subdir)
    if os.path.isdir(path):
        shutil.rmtree(path)

print("Done. Files in:", OUT_DIR)
