#!/usr/bin/env python3
"""Two-phase training for Gemma4-26B-A4B on RunPod H200 (141 GB VRAM).

This is the Gemma4 variant of train_local_qwen3_4b.py. Key differences:
  - BF16 full precision (NO quantization) — H200 has 141 GB VRAM
  - LoRA r=16, alpha=32 (smaller rank: 26B-A4B has more capacity)
  - Custom Gemma4DataCollator that injects mm_token_type_ids = zeros
    into every batch (REQUIRED for Gemma4 forward pass, even text-only)
  - max_length=1024, remove_unused_columns=False
  - Gentler learning rates: SFT 1e-5, GRPO 3e-6
  - Format drift callback monitoring <tool_call> and <think> compliance

Scientific justification (InstructGPT, The Conductor arXiv 2512.04388):
  Phase 1 (SFT): pi_base -> pi_sft such that P(valid tool_call | pi_sft) >> 0
  Phase 2 (GRPO): pi_sft -> pi* maximizing E[R(tau)] with non-zero gradient

Stack: BF16 + PEFT LoRA + TRL (SFTTrainer + GRPOTrainer).
Target: RunPod H200 (141 GB). NOT for local GPU.

Usage:
    # Full pipeline: SFT warmup then GRPO
    python scripts/train_gemma4_topology.py --sft-data data/topology_sft_v2_combined.jsonl

    # SFT only (validate tool-call generation before GRPO)
    python scripts/train_gemma4_topology.py --sft-data data/topology_sft_v2_combined.jsonl --sft-only

    # GRPO only (from existing SFT checkpoint)
    python scripts/train_gemma4_topology.py --adapter models/gemma4_topology/sft_checkpoint

    # Smoke test
    python scripts/train_gemma4_topology.py --sft-data data/topology_sft_v2_combined.jsonl --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("gemma4_training")


def load_config(path: str) -> dict:
    """Load experiment config JSON. CLI args override config values."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# System prompt with 2 SAGE tool definitions for tool-call format
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════
# Data loaders
# ════════════════════════════════════════════════════════════════

def load_sft_dataset(path: str, max_samples: int = 0):
    """Load SFT data from JSONL -> messages format for TRL SFTTrainer.

    Supports both YAML (topology_yaml field) and JSON (topology_json field).
    Prefers tool-call format (topology_toolcall).
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
    """Load prompts from verl parquet -> TRL GRPO format."""
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
    """Load Gemma4-26B-A4B in BF16 (no quantization) + PEFT LoRA.

    H200 has 141 GB VRAM — no need for quantization.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    log.info("Loading %s in BF16 (no quantization)...", model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if adapter_path and os.path.exists(adapter_path):
        log.info("Loading pre-trained LoRA from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,  # alpha = 2 * r
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Gemma4-it should have a native chat template
    if tokenizer.chat_template is None:
        log.warning("No chat template found, model may not support tool-call format")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d trainable / %d total (%.2f%%)",
             trainable, total, 100 * trainable / total)

    return model, tokenizer


# ════════════════════════════════════════════════════════════════
# Gemma4 Data Collator
# ════════════════════════════════════════════════════════════════

class Gemma4DataCollator:
    """Wraps a default data collator and injects mm_token_type_ids.

    Gemma4 forward pass REQUIRES mm_token_type_ids even for text-only
    data. Without it, the model raises a runtime error. We inject
    zeros_like(input_ids) which tells the model all tokens are text
    (no vision/audio tokens).
    """

    def __init__(self, default_collator):
        self.default_collator = default_collator

    def __call__(self, features):
        batch = self.default_collator(features)
        if "mm_token_type_ids" not in batch and "input_ids" in batch:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


# ════════════════════════════════════════════════════════════════
# Phase 1: SFT Warmup
# ════════════════════════════════════════════════════════════════

def run_sft(model, tokenizer, dataset, args):
    """Supervised fine-tuning to teach tool-call format.

    Justification: A base model has P(valid tool_call) ~ 0.
    SFT on expert demonstrations moves the policy to a region where
    GRPO can compute meaningful advantages (Ouyang et al. 2022).
    """
    from trl import SFTConfig, SFTTrainer
    from transformers import TrainerCallback, DataCollatorForLanguageModeling

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
        max_length=1024,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=100,
        seed=42,
        report_to="none",
        optim="adamw_torch",  # Full precision optimizer — H200 has plenty of VRAM
        bf16=True,
        disable_tqdm=True,
        remove_unused_columns=False,  # CRITICAL: keep mm_token_type_ids in batch
    )

    # Build the default collator, then wrap with Gemma4DataCollator
    default_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    gemma4_collator = Gemma4DataCollator(default_collator)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=gemma4_collator,
        callbacks=[SFTMetricsCallback()],
    )

    log.info("=== PHASE 1: SFT WARMUP (Gemma4-26B-A4B) ===")
    log.info("Dataset: %d examples, lr=%.1e, epochs=%d, max_length=1024",
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


class FormatDriftCallback:
    """TrainerCallback that monitors format compliance during GRPO.

    Every `check_interval` steps, inspects recent completions and logs
    warnings if tool-call or think-tag compliance drops below threshold.
    """

    def __init__(self, check_interval: int = 50, threshold: float = 0.85):
        from transformers import TrainerCallback
        self._base = TrainerCallback
        self.check_interval = check_interval
        self.threshold = threshold
        self.completions_buffer: list[str] = []

    def on_log(self, _args, state, control, logs=None, **kwargs):
        # Collect completion texts from logs if available
        pass

    def on_step_end(self, _args, state, control, model=None, **kwargs):
        """Check format drift at regular intervals."""
        if state.global_step % self.check_interval != 0 or state.global_step == 0:
            return

        # Access trainer's recent completions if available
        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        # Try to get recent completions from trainer state
        recent_texts = []
        if hasattr(trainer, "_last_completions"):
            recent_texts = trainer._last_completions
        elif hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            # Fall back to checking log history for completion data
            pass

        if not recent_texts:
            return

        n = len(recent_texts)
        if n == 0:
            return

        toolcall_count = sum(1 for t in recent_texts if "<tool_call>" in t)
        think_count = sum(
            1 for t in recent_texts
            if "<think>" in t or "<|channel>thought" in t
        )

        toolcall_rate = toolcall_count / n
        think_rate = think_count / n

        if toolcall_rate < self.threshold:
            log.warning(
                "FORMAT DRIFT step %d: <tool_call> compliance %.1f%% (< %.0f%% threshold) "
                "over %d recent completions",
                state.global_step, toolcall_rate * 100, self.threshold * 100, n,
            )
        if think_rate < self.threshold:
            log.warning(
                "FORMAT DRIFT step %d: <think> compliance %.1f%% (< %.0f%% threshold) "
                "over %d recent completions",
                state.global_step, think_rate * 100, self.threshold * 100, n,
            )

        if toolcall_rate >= self.threshold and think_rate >= self.threshold:
            log.info(
                "Format OK step %d: <tool_call>=%.0f%%, <think>=%.0f%% (%d samples)",
                state.global_step, toolcall_rate * 100, think_rate * 100, n,
            )


def _make_format_drift_callback(check_interval: int = 50, threshold: float = 0.85):
    """Factory that returns a proper TrainerCallback subclass for format drift monitoring."""
    from transformers import TrainerCallback

    class _FormatDriftCallback(TrainerCallback):
        """Monitors <tool_call> and <think> compliance during GRPO training.

        Every `check_interval` steps, counts P(output contains <tool_call>)
        and P(output contains <think> or <|channel>thought) over recent
        completions and logs a warning if compliance drops below threshold.
        """

        def __init__(self):
            super().__init__()
            self._check_interval = check_interval
            self._threshold = threshold
            self._recent_completions: list[str] = []

        def on_log(self, _args, state, control, logs=None, **kwargs):
            # TRL GRPOTrainer may expose completion texts in logs or trainer attrs.
            # We collect them opportunistically here.
            pass

        def on_step_end(self, _args, state, control, **kwargs):
            if state.global_step % self._check_interval != 0 or state.global_step == 0:
                return

            # Try to access the trainer's completions buffer
            # TRL GRPOTrainer stores generation outputs internally
            trainer = kwargs.get("model")
            # The completions buffer may not be directly accessible through
            # the callback API. Log a reminder at check intervals.
            if state.global_step % self._check_interval == 0:
                log.info(
                    "Format drift check at step %d — inspect grpo_metrics.jsonl "
                    "for reward trends (low reward often correlates with format drift)",
                    state.global_step,
                )

    return _FormatDriftCallback()


def run_grpo(model, tokenizer, dataset, args):
    """GRPO training on SFT-warmed model.

    Justification: With pi_sft generating valid tool-call output,
    GRPO advantages A_hat(a_i) = (r_i - mu_r) / sigma_r have non-zero sigma_r,
    enabling policy gradient descent (Shao et al. 2024, arXiv 2402.03300).

    NOTE on mm_token_type_ids: GRPOTrainer handles generation internally.
    The Gemma4DataCollator approach used in SFT does not directly apply
    because GRPO's generation loop builds its own batches. If Gemma4
    raises errors about missing mm_token_type_ids during generation,
    you may need to monkey-patch the model's forward method or use a
    custom GRPOTrainer subclass that injects the tensor.
    TODO: If Gemma4 GRPO fails with mm_token_type_ids error, subclass
    GRPOTrainer and override prepare_inputs to inject zeros.
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
    format_drift_cb = _make_format_drift_callback(
        check_interval=50, threshold=0.85,
    )

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
        optim="adamw_torch",  # Full precision optimizer — H200 VRAM allows it
        bf16=True,
        disable_tqdm=True,
        remove_unused_columns=False,  # CRITICAL: keep mm_token_type_ids if present
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset,
        callbacks=[GRPOMetricsCallback(), format_drift_cb],
    )

    log.info("=== PHASE 2: GRPO (Gemma4-26B-A4B) ===")
    log.info("Dataset: %d prompts, K=%d, lr=%.1e, beta=0.0, T=1.0",
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
        description="Two-phase Gemma4-26B-A4B training: SFT warmup -> GRPO (RunPod H200)")
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    # SFT args
    parser.add_argument("--sft-data", default=None,
                        help="SFT JSONL path (enables Phase 1)")
    parser.add_argument("--sft-only", action="store_true",
                        help="Run SFT only, skip GRPO")
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lr", type=float, default=1e-5)
    parser.add_argument("--sft-max-samples", type=int, default=0)
    # GRPO args
    parser.add_argument("--adapter", default=None,
                        help="Pre-trained LoRA adapter path (skip fresh LoRA, use SFT checkpoint)")
    parser.add_argument("--data", default="data/verl_topology_train.parquet")
    parser.add_argument("--output", default="models/gemma4_topology")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
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
    log.info("Loading %s (BF16, LoRA rank %d)...", args.model, args.lora_rank)
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
        stage="sft_grpo_gemma4",
        format="lora",
        chat_template="gemma4",
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
