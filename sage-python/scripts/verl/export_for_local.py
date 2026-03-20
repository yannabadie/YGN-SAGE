"""Export veRL-trained LoRA adapter for local 12GB inference.

After training on H100, the LoRA adapter needs to be:
1. Extracted from veRL checkpoint format
2. Saved in HuggingFace PEFT format
3. Optionally merged + quantized (GPTQ 4-bit) for 12GB local GPU

Usage:
    python scripts/verl/export_for_local.py \
        --checkpoint models/topology_verl_gigpo/ \
        --output models/topology_verl_local/
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("export")


def export_lora(checkpoint_dir: str, output_dir: str):
    """Extract and save LoRA adapter from veRL checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = Path(checkpoint_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # veRL saves checkpoints as: {checkpoint_dir}/global_step_{N}/
    # Find the latest checkpoint
    step_dirs = sorted(ckpt.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    if step_dirs:
        latest = step_dirs[-1]
        log.info("Using latest checkpoint: %s", latest)
    else:
        latest = ckpt
        log.info("Using checkpoint dir directly: %s", latest)

    # Check for actor model (veRL saves actor separately)
    actor_dir = latest / "actor"
    if actor_dir.exists():
        model_path = actor_dir
    else:
        model_path = latest

    # Copy adapter files
    for f in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]:
        src = model_path / f
        if src.exists():
            shutil.copy2(src, out / f)
            log.info("Copied %s", f)

    # Copy tokenizer
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "tokenizer.model", "vocab.json", "merges.txt"]:
        src = model_path / f
        if src.exists():
            shutil.copy2(src, out / f)

    log.info("LoRA adapter exported to %s", out)
    log.info("")
    log.info("To use locally (12GB GPU):")
    log.info("  from peft import PeftModel")
    log.info("  from transformers import AutoModelForCausalLM")
    log.info("  base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-9B', load_in_4bit=True)")
    log.info("  model = PeftModel.from_pretrained(base, '%s')", out)


def main():
    parser = argparse.ArgumentParser(description="Export veRL checkpoint for local inference")
    parser.add_argument("--checkpoint", default="models/topology_verl_gigpo/")
    parser.add_argument("--output", default="models/topology_verl_local/")
    args = parser.parse_args()
    export_lora(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
