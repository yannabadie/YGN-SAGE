#!/usr/bin/env python3
"""Two-phase local training: SFT warmup → GRPO on RTX 3500 Ada (12 GB).

Scientific justification (InstructGPT, The Conductor arXiv 2512.04388):
  Phase 1 (SFT): π_base → π_sft such that P(valid YAML | π_sft) >> 0
  Phase 2 (GRPO): π_sft → π* maximizing E[R(τ)] with non-zero gradient

Without SFT warmup, GRPO gradient ≈ 0 because all K rollouts produce
equally invalid YAML → advantage Â(a_i) ≈ 0 for all i.

Stack: bitsandbytes NF4 + PEFT LoRA + TRL (SFTTrainer + GRPOTrainer).

Usage:
    # Full pipeline: SFT warmup then GRPO
    python scripts/train_local_qwen3_4b.py --sft-data data/topology_sft_v2_combined.jsonl

    # SFT only (validate YAML generation before GRPO)
    python scripts/train_local_qwen3_4b.py --sft-data data/topology_sft_v2_combined.jsonl --sft-only

    # GRPO only (from existing SFT checkpoint)
    python scripts/train_local_qwen3_4b.py --adapter models/local_qwen3_4b_sft

    # Smoke test
    python scripts/train_local_qwen3_4b.py --sft-data data/topology_sft_v2_combined.jsonl --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("local_training")


def load_config(path: str) -> dict:
    """Load experiment config JSON. CLI args override config values."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# System prompt with 7 SAGE tool definitions for tool-call format
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════
# Data loaders
# ════════════════════════════════════════════════════════════════

def load_sft_dataset(path: str, max_samples: int = 0):
    """Load SFT data from JSONL → messages format for TRL SFTTrainer.

    Supports both YAML (topology_yaml field) and JSON (topology_json field).
    """
    from datasets import Dataset

    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            # Prefer tool-call format, fall back to JSON, then YAML
            topology_text = (entry.get("topology_toolcall")
                           or entry.get("topology_json")
                           or entry.get("topology_yaml", ""))
            # Use entry's system_prompt if available (has tool definitions baked in)
            sys_prompt = entry.get("system_prompt", SYSTEM_PROMPT)
            if not prompt or not topology_text:
                continue
            messages.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": topology_text},
            ])

    if max_samples > 0:
        messages = messages[:max_samples]

    log.info("Loaded %d SFT examples from %s", len(messages), path)
    return Dataset.from_dict({"messages": messages})


def load_grpo_dataset(path: str, max_samples: int = 0):
    """Load prompts from verl parquet → TRL GRPO format."""
    import numpy as np
    import pandas as pd
    from datasets import Dataset

    df = pd.read_parquet(path)
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

    log.info("Loaded %d GRPO prompts from %s", len(samples), path)
    return Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in samples
        ]
    })


# ════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════

def load_model(model_name: str, lora_rank: int, adapter_path: str | None = None):
    """Load base model in 4-bit NF4 + PEFT LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if adapter_path and os.path.exists(adapter_path):
        log.info("Loading pre-trained LoRA from %s", adapter_path)
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Qwen3-4B-Instruct has native tool-call chat template — do NOT override
    if tokenizer.chat_template is None:
        log.warning("No chat template found, model may not support tool-call format")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d trainable / %d total (%.2f%%)",
             trainable, total, 100 * trainable / total)

    return model, tokenizer


# ════════════════════════════════════════════════════════════════
# Phase 1: SFT Warmup
# ════════════════════════════════════════════════════════════════

def run_sft(model, tokenizer, dataset, args):
    """Supervised fine-tuning to teach YAML format.

    Justification: A base model has P(valid YAML) ≈ 0.
    SFT on expert demonstrations moves the policy to a region where
    GRPO can compute meaningful advantages (Ouyang et al. 2022).
    """
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback

    sft_output = os.path.join(args.output, "sft_checkpoint")
    metrics_file = os.path.join(args.output, "sft_metrics.jsonl")
    os.makedirs(sft_output, exist_ok=True)

    class SFTMetricsCallback(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                entry = {
                    "step": state.global_step,
                    "loss": logs.get("loss", 0),
                    "grad_norm": logs.get("grad_norm", 0),
                    "lr": logs.get("learning_rate", 0),
                }
                with open(metrics_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        def on_step_end(self, _args, state, control, **kwargs):
            torch.cuda.empty_cache()

    training_args = SFTConfig(
        output_dir=sft_output,
        learning_rate=args.sft_lr,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=768,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=100,
        seed=42,
        report_to="none",
        optim="adamw_8bit",
        bf16=True,
        disable_tqdm=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[SFTMetricsCallback()],
    )

    log.info("=== PHASE 1: SFT WARMUP ===")
    log.info("Dataset: %d examples, lr=%.1e, epochs=%d",
             len(dataset), args.sft_lr, args.sft_epochs)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Save adapter
    model.save_pretrained(sft_output)
    tokenizer.save_pretrained(sft_output)
    log.info("SFT complete in %.1f min. Adapter saved to %s", elapsed / 60, sft_output)

    return sft_output


# ════════════════════════════════════════════════════════════════
# Phase 2: GRPO
# ════════════════════════════════════════════════════════════════

def build_reward_fn():
    """Build reward function wrapping SAGE's compute_score."""
    # SAGE_VERL_EXEC=1 enables real execution reward (API calls)
    # SAGE_TRAINING_PHASE=C enables continuous bonuses (tier, checkpoints, providers)
    # Set via env or CLI before calling this script
    os.environ.setdefault("SAGE_VERL_EXEC", os.environ.get("SAGE_VERL_EXEC", "0"))
    os.environ.setdefault("SAGE_TRAINING_PHASE", os.environ.get("SAGE_TRAINING_PHASE", "A"))

    from sage.verl.reward import compute_score

    def reward_func(prompts, completions, **kwargs):
        """TRL GRPO reward function signature."""
        rewards = []
        for prompt, completion in zip(prompts, completions):
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


def run_grpo(model, tokenizer, dataset, args):
    """GRPO training on SFT-warmed model.

    Justification: With π_sft generating valid YAML,
    GRPO advantages Â(a_i) = (r_i - μ_r) / σ_r have non-zero σ_r,
    enabling policy gradient descent (Shao et al. 2024, arXiv 2402.03300).
    """
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    grpo_output = os.path.join(args.output, "grpo_checkpoint")
    metrics_file = os.path.join(args.output, "grpo_metrics.jsonl")
    os.makedirs(grpo_output, exist_ok=True)

    class GRPOMetricsCallback(TrainerCallback):
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

    reward_fn = build_reward_fn()

    training_args = GRPOConfig(
        output_dir=grpo_output,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,  # Conductor: more exploration post-SFT
        beta=0.0,  # No KL penalty (Conductor arXiv 2512.04388)
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
        callbacks=[GRPOMetricsCallback()],
    )

    log.info("=== PHASE 2: GRPO ===")
    log.info("Dataset: %d prompts, K=%d, lr=%.1e, β=0.0, T=1.0",
             len(dataset), args.num_generations, args.lr)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    model.save_pretrained(grpo_output)
    tokenizer.save_pretrained(grpo_output)
    log.info("GRPO complete in %.1f min. Model saved to %s", elapsed / 60, grpo_output)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase local training: SFT warmup → GRPO")
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    # SFT args
    parser.add_argument("--sft-data", default=None,
                        help="SFT JSONL path (enables Phase 1)")
    parser.add_argument("--sft-only", action="store_true",
                        help="Run SFT only, skip GRPO")
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lr", type=float, default=2e-5)
    parser.add_argument("--sft-max-samples", type=int, default=0)
    # GRPO args
    parser.add_argument("--adapter", default=None,
                        help="Pre-trained LoRA adapter path (skip SFT)")
    parser.add_argument("--data", default="data/verl_topology_train.parquet")
    parser.add_argument("--output", default="models/local_qwen3_4b_grpo")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="K rollouts per prompt for GRPO")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--config", default=None, help="JSON config file (CLI args override)")
    args = parser.parse_args()

    # ── Load config if provided ────────────────────────────────
    if args.config:
        config = load_config(args.config)
        # Compare against argparse defaults to detect explicit CLI overrides
        defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
        for key, value in config.items():
            key_attr = key.replace("-", "_")
            current = getattr(args, key_attr, None)
            default = defaults.get(key_attr)
            if current == default:  # User didn't explicitly override on CLI
                setattr(args, key_attr, value)
        log.info("Loaded config from %s", args.config)

    if args.smoke:
        args.sft_max_samples = 16
        args.sft_epochs = 1
        args.max_samples = 8
        args.num_generations = 2
        args.max_completion_length = 512
        args.grad_accum = 2
        log.info("=== SMOKE MODE ===")

    os.makedirs(args.output, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────
    log.info("Loading %s (4-bit NF4, LoRA rank %d)...", args.model, args.lora_rank)
    model, tokenizer = load_model(args.model, args.lora_rank, args.adapter)
    log.info("GPU: %s",
             os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
             .read().strip())

    # ── Phase 1: SFT Warmup ───────────────────────────────────
    # Runs SFT if sft_data is provided. With --adapter, this is "continued SFT"
    # (fine-tunes the existing adapter on new data). Without --adapter, fresh LoRA.
    if args.sft_data:
        sft_dataset = load_sft_dataset(args.sft_data, args.sft_max_samples)
        sft_path = run_sft(model, tokenizer, sft_dataset, args)

        if args.sft_only:
            log.info("=== SFT-only mode. Done. ===")
            return

    # ── Phase 2: GRPO ─────────────────────────────────────────
    if not args.sft_only:
        grpo_dataset = load_grpo_dataset(args.data, args.max_samples)
        run_grpo(model, tokenizer, grpo_dataset, args)

    # ── Final save + manifest ─────────────────────────────────
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    from sage.verl.manifest import TrainingManifest
    manifest = TrainingManifest(
        base_model=args.model,
        stage="sft_grpo_local",
        format="lora",
        chat_template="qwen3",
        output_path=args.output,
        dataset=args.data,
        dataset_size=0,
        algorithm="sft_then_grpo",
        lr=args.lr,
        epochs=args.epochs,
    )
    manifest.save(args.output)

    log.info("=== Training complete ===")
    log.info("Next: test with SAGE_ENABLE_PATH6=1")


if __name__ == "__main__":
    main()
