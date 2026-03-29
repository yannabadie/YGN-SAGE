#!/usr/bin/env python3
"""Local training: Qwen3-4B GRPO via TRL+PEFT+bitsandbytes on RTX 3500 Ada (12 GB).

Uses bitsandbytes NF4 quantization + PEFT LoRA + TRL GRPOTrainer.
No Unsloth/triton (incompatible on Windows).
Same reward function as the pod (compute_score from reward.py).
Same dataset (verl_topology_train.parquet).

This is a fast iteration loop for testing reward/data changes
before deploying to the H100 pod.

Usage:
    python scripts/train_local_qwen3_4b.py --smoke  # 2 steps, tiny batch
    python scripts/train_local_qwen3_4b.py           # full training
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("local_training")

# YGN-SAGE topology system prompt (same as pod training)
SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "Given a coding task, design an optimal agent topology as a YAML DAG. "
    "Include: difficulty, reasoning, nodes (role + prompt + model_tier), "
    "edges (from_idx + to_idx + flow_type). The LAST node must be a "
    "synthesizer that returns the final answer."
)


def load_dataset_from_parquet(path: str, max_samples: int = 0):
    """Load prompts from verl parquet format."""
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(path)
    log.info("Loaded %d entries from %s", len(df), path)

    samples = []
    for _, row in df.iterrows():
        prompt_data = row.get("prompt")
        if isinstance(prompt_data, (list, np.ndarray)):
            for m in prompt_data:
                if isinstance(m, dict) and m.get("role") == "user":
                    samples.append(m["content"])
                    break
        elif isinstance(prompt_data, str):
            samples.append(prompt_data)

    if max_samples > 0:
        samples = samples[:max_samples]

    log.info("Extracted %d prompts", len(samples))
    return samples


def build_reward_fn():
    """Build reward function wrapping SAGE's compute_score."""
    os.environ.setdefault("SAGE_VERL_EXEC", "0")
    os.environ.setdefault("SAGE_TRAINING_PHASE", "A")

    from sage.verl.reward import compute_score

    def reward_func(prompts, completions, **kwargs):
        """TRL GRPO reward function signature."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Extract text from completion
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)

            score = compute_score("sage_topology", text, "", {})
            rewards.append(float(score))
        return rewards

    return reward_func


def main():
    parser = argparse.ArgumentParser(description="Local Qwen3-4B GRPO training")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 2 steps")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Model name (default: Qwen/Qwen3-4B)")
    parser.add_argument("--data", default="data/verl_topology_train.parquet")
    parser.add_argument("--output", default="models/local_qwen3_4b_grpo")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max training samples (0=all)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="K rollouts per prompt for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=1024,
                        help="Max tokens per completion")
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 4
        args.epochs = 1
        args.num_generations = 2
        log.info("=== SMOKE MODE ===")

    # ── Load model with bitsandbytes 4-bit + PEFT LoRA ─────────
    log.info("Loading %s with bitsandbytes NF4 + LoRA rank %d...", args.model, args.lora_rank)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ── Set chat template for topology generation ───────────────
    tokenizer.chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{ '{SYSTEM_PROMPT}' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '' }}{% endif %}"
    )

    # ── Load dataset ────────────────────────────────────────────
    prompts = load_dataset_from_parquet(args.data, args.max_samples)

    # Convert to TRL format
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]
    })
    log.info("Dataset: %d samples", len(dataset))

    # ── Reward function ─────────────────────────────────────────
    reward_fn = build_reward_fn()

    # ── GRPO Training ───────────────────────────────────────────
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    metrics_file = os.path.join(args.output, "live_metrics.jsonl")

    class LogAndCleanCallback(TrainerCallback):
        """Log rewards to file + clear CUDA cache to prevent fragmentation."""
        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                entry = {
                    "step": state.global_step,
                    "reward": logs["reward"],
                    "loss": logs.get("loss", 0),
                    "grad_norm": logs.get("grad_norm", 0),
                    "step_time": logs.get("step_time", 0),
                    "clipped_ratio": logs.get("completions/clipped_ratio", 0),
                }
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        def on_step_end(self, _args, state, control, **kwargs):
            torch.cuda.empty_cache()

    os.makedirs(args.output, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        beta=0.0,  # No KL (Conductor-style)
        logging_steps=1,
        save_steps=50,
        seed=42,
        report_to="none",
        optim="adamw_8bit",
        bf16=True,
        disable_tqdm=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        callbacks=[LogAndCleanCallback()],
    )

    log.info("Starting GRPO training: %d samples, %d epochs, K=%d, lr=%s",
             len(dataset), args.epochs, args.num_generations, args.lr)
    log.info("GPU: %s", os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip())

    trainer.train()

    # ── Save ────────────────────────────────────────────────────
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    log.info("Model saved to %s", args.output)

    # ── Write manifest ──────────────────────────────────────────
    from sage.verl.manifest import TrainingManifest
    manifest = TrainingManifest(
        base_model=args.model,
        stage="grpo_local",
        format="lora",
        chat_template="qwen3",
        output_path=args.output,
        dataset=args.data,
        dataset_size=len(dataset),
        algorithm="grpo_trl_peft",
        lr=args.lr,
        epochs=args.epochs,
    )
    manifest.save(args.output)

    log.info("=== Local training complete ===")
    log.info("Next: test with SAGE_ENABLE_PATH6=1")


if __name__ == "__main__":
    main()
