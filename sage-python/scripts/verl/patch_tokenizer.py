#!/usr/bin/env python3
"""Patch Qwen3.5 tokenizer to disable thinking mode for veRL/vLLM rollouts.

BLOCKER FIX: vLLM loads the tokenizer from HuggingFace with the original
chat_template that includes <think> blocks. TRL's GRPOTrainer has a local
patch in train_local_grpo.py, but veRL's vLLM rollout engine bypasses it.

This script:
  1. Loads the Qwen3.5 tokenizer from HuggingFace
  2. Removes the thinking mode from the Jinja chat_template
  3. Saves the patched tokenizer to a local directory
  4. The train script points actor_rollout_ref.model.path to this directory

The model weights are NOT copied — only tokenizer files. vLLM will load
weights from the HF cache and tokenizer from the local patched directory.

Usage:
    python scripts/verl/patch_tokenizer.py [--model Qwen/Qwen3.5-9B] [--output /workspace/patched_tokenizer]
    
    # Then in train_topology.sh:
    #   MODEL=/workspace/patched_model  (contains tokenizer + symlinked weights)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("patch_tokenizer")


def patch_chat_template(template: str) -> str:
    """Remove thinking mode from Qwen3.5 Jinja chat_template.
    
    Qwen3.5's template has a conditional block that inserts <think>\\n\\n</think>\\n\\n
    even when enable_thinking=False. This makes the model waste all generation
    budget on thinking tokens instead of producing YAML.
    
    Strategy: Remove the entire enable_thinking conditional block from the Jinja
    template. This forces the assistant turn to start directly without any think tags.
    """
    original = template
    
    # Strategy 1: Remove the full Jinja conditional for thinking
    # Pattern: {%- if enable_thinking ... %} ... {%- endif %}
    patched = re.sub(
        r'\{%-?\s*if\s+enable_thinking.*?%\}.*?\{%-?\s*endif\s*-?%\}',
        '', template, flags=re.DOTALL
    )
    
    # Strategy 2: If regex didn't catch it, brute-force remove think tags
    if "<think>" in patched:
        patched = patched.replace("<think>\\n", "")
        patched = patched.replace("</think>\\n\\n", "")
        patched = patched.replace("<think>\n", "")
        patched = patched.replace("</think>\n\n", "")
        patched = patched.replace("<think>", "")
        patched = patched.replace("</think>", "")
    
    if patched != original:
        log.info("Thinking mode removed from chat_template (%d chars removed)",
                 len(original) - len(patched))
    else:
        log.warning("No thinking mode found in chat_template — template unchanged")
    
    return patched


def patch_tokenizer(model_name: str, output_dir: str):
    """Load, patch, and save tokenizer."""
    from transformers import AutoTokenizer, AutoConfig
    
    log.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Check if thinking mode is present
    if tokenizer.chat_template and "<think>" in tokenizer.chat_template:
        log.info("Thinking mode detected in chat_template — patching...")
        tokenizer.chat_template = patch_chat_template(tokenizer.chat_template)
    else:
        log.info("No thinking mode in chat_template — no patch needed")
    
    # Verify the patch works
    test_messages = [
        {"role": "system", "content": "You are a topology designer."},
        {"role": "user", "content": "Design a topology for: hello world"},
    ]
    rendered = tokenizer.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    assert "<think>" not in rendered, f"Patch failed! <think> still in rendered output:\n{rendered[-200:]}"
    log.info("Patch verified — no <think> tags in rendered output")
    log.info("Rendered ends with: ...%s", rendered[-80:])
    
    # Save patched tokenizer
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out))
    config.save_pretrained(str(out))
    log.info("Saved patched tokenizer to %s", output_dir)
    
    # Create a symlink script for model weights (user runs on pod)
    symlink_script = out / "link_weights.sh"
    symlink_script.write_text(f"""#!/bin/bash
# Symlink model weights from HF cache into patched tokenizer directory
# Run this AFTER the model is downloaded (e.g., after first vLLM load)
HF_CACHE=$(python3 -c "from huggingface_hub import scan_cache_dir; \
    info = scan_cache_dir(); \
    revs = [r for repo in info.repos if '{model_name.split('/')[-1]}' in repo.repo_id for r in repo.revisions]; \
    print(revs[0].snapshot_path if revs else '')")

if [ -z "$HF_CACHE" ]; then
    echo "Model not in HF cache yet. Download it first:"
    echo "  python -c \\"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('{model_name}')\\""
    exit 1
fi

echo "Linking weights from $HF_CACHE"
for f in "$HF_CACHE"/model*.safetensors "$HF_CACHE"/model.safetensors.index.json; do
    [ -f "$f" ] && ln -sf "$f" "{output_dir}/$(basename $f)"
done
echo "Done. Model ready at {output_dir}"
""")
    os.chmod(str(symlink_script), 0o755)
    log.info("Created weight symlink script: %s", symlink_script)
    
    return str(out)


def main():
    parser = argparse.ArgumentParser(description="Patch Qwen3.5 tokenizer for veRL")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model name")
    parser.add_argument("--output", default="/workspace/patched_model",
                        help="Output directory for patched tokenizer")
    args = parser.parse_args()
    
    patch_tokenizer(args.model, args.output)
    log.info("Done. Use --model %s in train_topology.sh", args.output)


if __name__ == "__main__":
    main()
