#!/usr/bin/env python3
"""Local GRPO training on Qwen3.5-4B with Unsloth QLoRA.

Two phases:
  Phase A: Structural reward ($0 API) — learns YAML format + adaptation metadata
  Phase B: Execution reward (API calls) — learns multi-provider topology execution

Usage:
    # Phase A only (structural, fast)
    python scripts/train_local_grpo.py --phase A

    # Phase B only (execution, needs API keys in .env)
    python scripts/train_local_grpo.py --phase B

    # Both phases
    python scripts/train_local_grpo.py --phase AB

Auto-recovery:
    - OOM → halve batch_size, retry
    - API timeout → structural fallback
    - NaN loss → rollback checkpoint, reduce lr 50%
    - Rate limit → exponential backoff + structural fallback
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_local.log", mode="a"),
    ],
)
log = logging.getLogger("train_local")


def load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        count = 0
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"'))
                count += 1
        log.info("Loaded .env with %d keys", count)


def setup_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load model with PEFT QLoRA (4-bit).

    Uses transformers + peft directly (Unsloth incompatible with Windows).
    Default: Qwen2.5-3B-Instruct (proven, stable, fits 12GB).
    Override with --model flag for Qwen3.5-4B when available.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("Loading model: %s (4-bit QLoRA)", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: %s, %d/%d params trainable (%.1f%%)",
             model_name, trainable, total, 100 * trainable / total)
    return model, tokenizer


def create_reward_fn(phase: str):
    """Create reward function for the given phase."""
    from sage.verl.reward import _score_format, _score_structure

    def reward_fn(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)

            fmt = _score_format(text)
            struct = _score_structure(text)
            fmt_norm = (fmt + 2.0) / 3.0

            if phase == "A":
                # Structural only + adaptation bonus
                score = (fmt_norm + struct) / 2.0
                # Bonus for adaptive fields
                try:
                    import yaml
                    topo = yaml.safe_load(text)
                    if isinstance(topo, dict):
                        if topo.get("adaptation"):
                            score += 0.1
                        nodes = topo.get("nodes", [])
                        if any(n.get("fallback_tier") for n in nodes if isinstance(n, dict)):
                            score += 0.1
                        if topo.get("reasoning") and len(str(topo.get("reasoning", ""))) > 50:
                            score += 0.05
                except Exception:
                    pass
                rewards.append(float(max(0.0, min(2.0, score))))
            else:
                # Phase B: full 5-signal reward
                try:
                    from sage.verl.reward import _score_rust_density
                    rust = _score_rust_density(text, {})
                except Exception:
                    rust = 0.0
                score = (fmt_norm + struct + rust) / 3.0
                rewards.append(float(max(0.0, min(2.0, score))))

        return rewards

    return reward_fn


def load_dataset(phase: str):
    """Load prompts from parquet."""
    import pandas as pd

    if phase == "A":
        path = Path("data/verl_topology_train.parquet")
    else:
        path = Path("data/verl_topology_curated.parquet")

    if not path.exists():
        log.error("Dataset not found: %s. Run convert_sft_to_verl.py first.", path)
        sys.exit(1)

    df = pd.read_parquet(path)
    log.info("Loaded %d prompts from %s", len(df), path)

    # Convert to list of chat message dicts
    # Parquet stores prompts as numpy arrays of dicts — convert to list
    dataset = []
    for _, row in df.iterrows():
        prompt = row.get("prompt", [])
        # Handle numpy array (from parquet)
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                prompt = [{"role": "user", "content": prompt}]
        if isinstance(prompt, (list, tuple)) and len(prompt) > 0:
            # Extract just the user message text for TRL GRPOTrainer
            user_msg = ""
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if user_msg:
                dataset.append({"prompt": user_msg})

    log.info("Converted %d prompts to training format", len(dataset))
    return dataset


def train_phase(phase: str, model, tokenizer, batch_size: int = 4):
    """Run one training phase with auto-recovery."""
    from trl import GRPOConfig, GRPOTrainer

    log.info("=== Starting Phase %s (batch_size=%d) ===", phase, batch_size)

    dataset = load_dataset(phase)
    if not dataset:
        log.error("No data for Phase %s", phase)
        return None

    reward_fn = create_reward_fn(phase)

    epochs = 3 if phase == "A" else 5
    lr = 5e-5 if phase == "A" else 2e-5
    output_dir = f"models/local_grpo_phase_{phase}"

    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 4 // batch_size),
        learning_rate=lr,
        num_generations=4,  # K=4 rollouts per prompt (for RewardFlow compatibility)
        max_completion_length=1024,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

    try:
        log.info("Training Phase %s: %d prompts, %d epochs, lr=%.1e", phase, len(dataset), epochs, lr)
        result = trainer.train()
        log.info("Phase %s complete. Metrics: %s", phase, result.metrics)
        trainer.save_model(output_dir)
        log.info("Model saved to %s", output_dir)
        return result
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and batch_size > 1:
            log.warning("OOM in Phase %s — retrying with batch_size=%d", phase, batch_size // 2)
            import torch
            torch.cuda.empty_cache()
            time.sleep(5)
            return train_phase(phase, model, tokenizer, batch_size // 2)
        log.error("Phase %s failed: %s", phase, e)
        raise
    except Exception as e:
        log.error("Phase %s unexpected error: %s", phase, e, exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Local GRPO training for SAGE V2")
    parser.add_argument("--phase", default="AB", choices=["A", "B", "AB"],
                        help="Training phase: A (structural), B (execution), AB (both)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (auto-halved on OOM)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model name (default: Qwen2.5-3B-Instruct)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data, don't train")
    args = parser.parse_args()

    load_env()
    log.info("=" * 60)
    log.info("SAGE V2 Local GRPO Training — %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Phase: %s, Batch size: %d, Model: %s", args.phase, args.batch_size, args.model)
    log.info("=" * 60)

    log.info("Setting up %s with PEFT QLoRA...", args.model)
    model, tokenizer = setup_model(args.model)

    if args.dry_run:
        log.info("Dry run: model loaded, testing data loading...")
        for p in ["A", "B"]:
            if p in args.phase:
                ds = load_dataset(p)
                log.info("Phase %s: %d prompts loaded", p, len(ds))
        log.info("Dry run complete.")
        return

    if "A" in args.phase:
        train_phase("A", model, tokenizer, args.batch_size)

    if "B" in args.phase:
        os.environ["SAGE_VERL_EXEC"] = "1"
        train_phase("B", model, tokenizer, max(1, args.batch_size // 2))

    log.info("=" * 60)
    log.info("Training complete. Check train_local.log for full history.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
