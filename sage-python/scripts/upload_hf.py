"""Upload SFT topology policy to HuggingFace Hub.

Usage:
    python scripts/upload_hf.py
    python scripts/upload_hf.py --repo yannabadie/sage-topology-policy --model-dir models/topology_sft/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("upload_hf")

REPO_ID = "yannabadie/sage-topology-policy"
MODEL_DIR = "models/topology_sft/"

MODEL_CARD = """---
library_name: peft
base_model: microsoft/Phi-4-mini-instruct
license: mit
tags:
  - topology-generation
  - multi-agent
  - agent-orchestration
  - qlora
  - phi-4
datasets:
  - custom (GPT-5.4 distilled)
language:
  - en
metrics:
  - loss
  - accuracy
pipeline_tag: text-generation
---

# SAGE Topology Policy

**Task:** Generate optimal multi-agent topologies (DAGs) for coding and reasoning tasks.

This is a QLoRA adapter for [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) (3.8B), trained to generate YAML topology specifications for the [YGN-SAGE](https://github.com/yannabadie/YGN-SAGE) Agent Development Kit.

## Training

| Parameter | Value |
|-----------|-------|
| Base model | microsoft/Phi-4-mini-instruct (3.8B) |
| Method | QLoRA 4-bit NF4 + double quantization |
| LoRA rank | r=16, alpha=32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Training data | 2,624 topologies distilled from GPT-5.4 (reasoning=high) |
| Data sources | BigCodeBench (1,140), GSM8K (1,319), Code Contests (165) |
| Data format | Double-quote YAML (no multiline wrapping) |
| Epochs | 5 |
| Batch size | 1 (gradient accumulation 8) |
| Learning rate | 2e-4 |
| Max length | 1,280 tokens |
| Hardware | NVIDIA RTX 3500 Ada (12GB VRAM) |
| Training time | ~2 hours |

## Metrics

| Metric | Value |
|--------|-------|
| Train loss | 0.896 |
| Token accuracy | 77.2% |
| YAML validity | 70% (SFT only; GRPO expected to reach 100%) |

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model + adapter
tokenizer = AutoTokenizer.from_pretrained("yannabadie/sage-topology-policy")
base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    trust_remote_code=False,
    dtype=torch.float16,
    device_map="cpu",
)
model = PeftModel.from_pretrained(base, "yannabadie/sage-topology-policy")
model = model.to("cuda:0")  # or keep on CPU
model.eval()

# Generate a topology
prompt = (
    '<|system|>You are a multi-agent topology designer. '
    'Given a task, generate an optimal agent topology in YAML format.<|end|>\\n'
    '<|user|>Write a merge sort function<|end|>\\n'
    '<|assistant|>'
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=500, temperature=0.3, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Output Format

The model generates YAML topologies with this structure:

```yaml
"difficulty": "moderate"
"edges":
- "flow_type": "control"
  "from_idx": 0
  "to_idx": 1
"nodes":
- "model_tier": "reasoner"
  "prompt": "Design the algorithm and plan the implementation..."
  "role": "planner"
- "model_tier": "fast"
  "prompt": "Implement the planned solution in Python..."
  "role": "coder"
"reasoning": "This task requires planning then implementation..."
```

## Part of YGN-SAGE

This model is the topology generation policy for [YGN-SAGE](https://github.com/yannabadie/YGN-SAGE), a Self-Adaptive Generation Engine built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

The next step is GRPO training with formally verified dense rewards (OxiZ SMT + HybridVerifier + S_complex density) to reach 100% YAML validity.

## License

MIT
"""


def main():
    parser = argparse.ArgumentParser(description="Upload topology policy to HuggingFace")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        log.error("Model directory not found: %s", model_path)
        return

    # Check required files
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for f in required:
        if not (model_path / f).exists():
            log.error("Missing required file: %s", model_path / f)
            return

    # Write model card
    readme_path = model_path / "README.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")
    log.info("Model card written to %s", readme_path)

    # Upload
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("pip install huggingface_hub")
        return

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(repo_id=args.repo, exist_ok=True, private=args.private)
        log.info("Repo %s ready", args.repo)
    except Exception as exc:
        log.warning("Repo creation: %s", exc)

    # Upload all files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo,
        ignore_patterns=["checkpoint-*", "*.pt", "*.pth", "training_args.bin"],
    )
    log.info("Uploaded to https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
