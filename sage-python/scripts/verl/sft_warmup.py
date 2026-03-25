#!/usr/bin/env python3
"""Quick SFT warmup to teach Nemotron-Orchestrator-8B the YAML topology format.

This runs 1-2 epochs of supervised fine-tuning on ground-truth YAML topologies
so the model learns the output format BEFORE we do GRPO/GiGPO.
Without this, GRPO gets zero variance in rewards (all completions get ~0)
and can't learn.

Usage:
    python3 scripts/verl/sft_warmup.py
"""
import json
import os
import sys
import logging

import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SFT] %(message)s")
log = logging.getLogger("sft_warmup")

# Config
MODEL_PATH = os.environ.get("SAGE_MODEL", "/workspace/patched_nemotron_orchestrator")
OUTPUT_DIR = "/workspace/sft_warmup_output"
JSONL_PATH = "data/topology_sft_v2_combined.jsonl"
LR = 2e-5
EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 8  # effective batch = 32
MAX_SEQ_LEN = 768
LORA_R = 64
LORA_ALPHA = 32


def load_sft_data(jsonl_path: str, tokenizer):
    """Load SFT data from JSONL with system/user/assistant messages."""
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, examples, tokenizer, max_len):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            messages = ex["messages"]
            # Build full text manually (Qwen3 format, avoids patched template issues)
            parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            text = "\n".join(parts)
            # Tokenize
            enc = self.tokenizer(
                text, truncation=True, max_length=self.max_len,
                padding="max_length", return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            # Labels = input_ids (shift handled by model)
            labels = input_ids.clone()
            # Mask padding
            labels[attention_mask == 0] = -100
            # Mask everything before the assistant response
            assistant_marker = "<|im_start|>assistant\n"
            marker_pos = text.rfind(assistant_marker)
            if marker_pos >= 0:
                prefix = text[:marker_pos + len(assistant_marker)]
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                mask_len = min(len(prefix_ids), len(labels))
                labels[:mask_len] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    SYSTEM_PROMPT = (
        "You are a multi-agent topology designer for the YGN-SAGE framework. "
        "Given a coding task, design an optimal agent topology as a YAML DAG. "
        "Include: difficulty, reasoning, nodes (role + prompt + model_tier), "
        "edges (from_idx + to_idx + flow_type). The LAST node must be a "
        "synthesizer that returns the final answer."
    )
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line.strip())
            if "messages" in data:
                examples.append(data)
            elif "topology_yaml" in data and "prompt" in data:
                # SAGE format: prompt + topology_yaml
                yaml_text = data["topology_yaml"]
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": data["prompt"]},
                        {"role": "assistant", "content": yaml_text},
                    ]
                })
            elif "prompt" in data and "completion" in data:
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": data["prompt"]},
                        {"role": "assistant", "content": data["completion"]},
                    ]
                })

    log.info(f"Loaded {len(examples)} SFT examples from {jsonl_path}")
    return SFTDataset(examples, tokenizer, MAX_SEQ_LEN)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    log.info(f"Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda")
    log.info(f"Model loaded: {model.config.model_type}")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # Required for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    # Load data
    dataset = load_sft_data(JSONL_PATH, tokenizer)
    log.info(f"Dataset: {len(dataset)} examples")

    # Training
    from transformers import Trainer

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    log.info("Starting SFT warmup training...")
    trainer.train()
    log.info("SFT warmup complete!")

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info(f"Model saved to {OUTPUT_DIR}")

    # Verify: generate a sample
    log.info("Generating sample output...")
    model.eval()
    test_msgs = [
        {"role": "system", "content": "You are a multi-agent topology designer for the YGN-SAGE framework. Given a coding task, design an optimal agent topology as a YAML DAG."},
        {"role": "user", "content": "Write a function that checks if a string is a palindrome."},
    ]
    test_input = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, temperature=0.3, do_sample=True)
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info(f"Sample output:\n{generated[:500]}")

    # Check if it looks like YAML
    import yaml
    try:
        parsed = yaml.safe_load(generated.split("```")[0] if "```" in generated else generated)
        if parsed and isinstance(parsed, dict):
            log.info("SUCCESS: Output parses as valid YAML!")
        else:
            log.warning("Output parsed but is not a dict")
    except Exception:
        log.warning("Output does NOT parse as YAML (may improve with more training)")


if __name__ == "__main__":
    main()
