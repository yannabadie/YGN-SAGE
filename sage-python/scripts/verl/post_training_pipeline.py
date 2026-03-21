#!/usr/bin/env python3
"""Post-training pipeline: export → merge → push HuggingFace → quantize Q8 GGUF.

Run ON THE POD after training completes:

    # Step 1: Export LoRA from veRL checkpoint
    python scripts/verl/post_training_pipeline.py export

    # Step 2: Merge LoRA into base Qwen3.5-9B
    python scripts/verl/post_training_pipeline.py merge

    # Step 3: Push merged model to HuggingFace
    python scripts/verl/post_training_pipeline.py push

    # Step 4: Quantize to Q8 GGUF (for local 12GB inference)
    python scripts/verl/post_training_pipeline.py quantize

    # All steps:
    python scripts/verl/post_training_pipeline.py all
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("post_train")

# --- Config ---
BASE_MODEL = "Qwen/Qwen3.5-9B"
HF_REPO = "yannabadie/sage-topology-policy-v2"
GGUF_QUANT = "Q8_0"  # Q8 for 12GB GPU (~9.5GB)

CHECKPOINT_DIR = "models/topology_verl_gigpo/"
LORA_DIR = "models/topology_verl_lora/"
MERGED_DIR = "models/topology_verl_merged/"
GGUF_DIR = "models/topology_verl_gguf/"


def step_export():
    """Extract LoRA adapter from veRL checkpoint."""
    log.info("=== STEP 1: Export LoRA from veRL checkpoint ===")

    ckpt = Path(CHECKPOINT_DIR)
    out = Path(LORA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Find latest checkpoint
    step_dirs = sorted(ckpt.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if step_dirs:
        latest = step_dirs[-1]
        log.info("Latest checkpoint: %s", latest)
    else:
        latest = ckpt
        log.info("Using checkpoint dir: %s", latest)

    # veRL saves actor model separately
    actor_dir = latest / "actor"
    model_path = actor_dir if actor_dir.exists() else latest

    # Copy adapter files
    copied = 0
    for f in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        src = model_path / f
        if src.exists():
            shutil.copy2(src, out / f)
            log.info("Copied %s (%.1f MB)", f, src.stat().st_size / 1e6)
            copied += 1

    # Copy tokenizer
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.model", "chat_template.jinja"]:
        src = model_path / f
        if not src.exists():
            # Try base model cache
            from huggingface_hub import hf_hub_download
            try:
                downloaded = hf_hub_download(BASE_MODEL, f)
                shutil.copy2(downloaded, out / f)
            except Exception:
                pass
        else:
            shutil.copy2(src, out / f)

    if copied == 0:
        log.error("No adapter files found in %s", model_path)
        sys.exit(1)

    log.info("LoRA exported to %s", out)


def step_merge():
    """Merge LoRA adapter into base model (full float16)."""
    log.info("=== STEP 2: Merge LoRA into %s ===", BASE_MODEL)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lora = Path(LORA_DIR)
    out = Path(MERGED_DIR)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model %s...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    log.info("Loading LoRA from %s...", lora)
    model = PeftModel.from_pretrained(model, str(lora))

    log.info("Merging LoRA into base model...")
    model = model.merge_and_unload()

    log.info("Saving merged model to %s...", out)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))

    size_gb = sum(f.stat().st_size for f in out.rglob("*.safetensors")) / 1e9
    log.info("Merged model saved: %.1f GB", size_gb)


def step_push():
    """Push merged model to HuggingFace Hub."""
    log.info("=== STEP 3: Push to HuggingFace %s ===", HF_REPO)

    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN not set. Export it: export HF_TOKEN=hf_...")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if needed
    try:
        create_repo(HF_REPO, token=token, exist_ok=True, private=False)
        log.info("Repository %s ready", HF_REPO)
    except Exception as e:
        log.warning("create_repo: %s", e)

    # Upload merged model
    merged = Path(MERGED_DIR)
    if merged.exists():
        log.info("Uploading merged model (this may take a while)...")
        api.upload_folder(
            folder_path=str(merged),
            repo_id=HF_REPO,
            commit_message="feat: SAGE V2 topology policy — Qwen3.5-9B GiGPO trained",
        )
        log.info("Merged model pushed to %s", HF_REPO)
    else:
        # Upload LoRA only
        lora = Path(LORA_DIR)
        log.info("No merged model found, uploading LoRA adapter...")
        api.upload_folder(
            folder_path=str(lora),
            repo_id=HF_REPO,
            path_in_repo="lora/",
            commit_message="feat: SAGE V2 LoRA adapter — Qwen3.5-9B GiGPO",
        )
        log.info("LoRA adapter pushed to %s/lora/", HF_REPO)

    # Upload GGUF if available
    gguf = Path(GGUF_DIR)
    if gguf.exists() and list(gguf.glob("*.gguf")):
        log.info("Uploading GGUF quantized model...")
        api.upload_folder(
            folder_path=str(gguf),
            repo_id=HF_REPO,
            path_in_repo="gguf/",
            commit_message=f"feat: {GGUF_QUANT} GGUF quantization for local inference",
        )
        log.info("GGUF uploaded to %s/gguf/", HF_REPO)


def step_quantize():
    """Quantize merged model to GGUF Q8_0 for local 12GB inference."""
    log.info("=== STEP 4: Quantize to %s GGUF ===", GGUF_QUANT)

    merged = Path(MERGED_DIR)
    out = Path(GGUF_DIR)
    out.mkdir(parents=True, exist_ok=True)

    if not merged.exists():
        log.error("Merged model not found at %s. Run 'merge' first.", merged)
        sys.exit(1)

    # Check if llama.cpp convert script is available
    llama_cpp = shutil.which("llama-quantize")
    convert_script = None

    # Try to find llama.cpp tools
    for candidate in [
        "/workspace/llama.cpp/convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
        shutil.which("convert_hf_to_gguf.py"),
    ]:
        if candidate and Path(candidate).exists():
            convert_script = candidate
            break

    if convert_script is None:
        log.info("llama.cpp not found. Installing...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/ggml-org/llama.cpp.git",
            "/workspace/llama.cpp"
        ], check=True)
        subprocess.run(["pip", "install", "-r", "/workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"], check=True)
        convert_script = "/workspace/llama.cpp/convert_hf_to_gguf.py"

        # Build quantize tool
        subprocess.run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd="/workspace/llama.cpp", check=True)
        subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd="/workspace/llama.cpp", check=True)
        llama_cpp = "/workspace/llama.cpp/build/bin/llama-quantize"

    # Step 1: Convert to GGUF F16
    f16_path = out / "model-f16.gguf"
    log.info("Converting to GGUF F16...")
    subprocess.run([
        sys.executable, convert_script,
        str(merged),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ], check=True)
    log.info("F16 GGUF: %.1f GB", f16_path.stat().st_size / 1e9)

    # Step 2: Quantize to Q8_0
    q8_path = out / f"sage-topology-v2-{GGUF_QUANT}.gguf"
    if llama_cpp and Path(llama_cpp).exists():
        log.info("Quantizing to %s...", GGUF_QUANT)
        subprocess.run([llama_cpp, str(f16_path), str(q8_path), GGUF_QUANT], check=True)
        log.info("Q8 GGUF: %.1f GB", q8_path.stat().st_size / 1e9)

        # Clean up F16
        f16_path.unlink()
        log.info("Cleaned up F16 intermediate")
    else:
        log.warning("llama-quantize not found. F16 GGUF available at %s", f16_path)
        log.warning("Quantize manually: llama-quantize %s %s %s", f16_path, q8_path, GGUF_QUANT)

    log.info("GGUF ready for local inference at %s", out)
    log.info("")
    log.info("To run locally (12GB GPU):")
    log.info("  llama-cli -m %s -p 'Design a topology for: sort a list'", q8_path)
    log.info("  # Or via llama-cpp-python:")
    log.info("  from llama_cpp import Llama")
    log.info("  model = Llama(model_path='%s', n_gpu_layers=-1, n_ctx=2048)", q8_path)


def main():
    parser = argparse.ArgumentParser(description="Post-training pipeline")
    parser.add_argument("step", choices=["export", "merge", "push", "quantize", "all"],
                        help="Pipeline step to run")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--quant", default=GGUF_QUANT, choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"])
    args = parser.parse_args()

    global BASE_MODEL, HF_REPO, GGUF_QUANT
    BASE_MODEL = args.base_model
    HF_REPO = args.hf_repo
    GGUF_QUANT = args.quant

    steps = {
        "export": [step_export],
        "merge": [step_merge],
        "push": [step_push],
        "quantize": [step_quantize],
        "all": [step_export, step_merge, step_push, step_quantize],
    }

    for fn in steps[args.step]:
        fn()

    log.info("Done.")


if __name__ == "__main__":
    main()
