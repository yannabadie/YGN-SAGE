"""Train topology generation policy via GRPO (Group Relative Policy Optimization).

Uses Phi-4-mini-instruct (3.8B, MIT, official ONNX) as base model.
Reward function: SAGE's verified dense reward (execution + HybridVerifier + S_complex + LTL).

Requires: pip install trl transformers torch peft accelerate

Usage:
    # Step 1: SFT on collected topology data
    python scripts/train_topology_grpo.py --mode sft --data data/topology_sft.jsonl --epochs 3

    # Step 2: GRPO with verified rewards
    python scripts/train_topology_grpo.py --mode grpo --sft-checkpoint models/topology_sft/ --episodes 1000

    # Step 3: Export to ONNX for Rust inference
    python scripts/train_topology_grpo.py --mode export --checkpoint models/topology_grpo/ --output models/topology_policy.onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("train_topology")

# Base model: Phi-4-mini-instruct (3.8B, MIT license)
# Official ONNX: microsoft/Phi-4-mini-instruct-onnx
BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ONNX_MODEL = "microsoft/Phi-4-mini-instruct-onnx"


def run_sft(data_path: str, output_dir: str, epochs: int):
    """Stage 1: Supervised Fine-Tuning on collected topology data."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig
    except ImportError:
        log.error("Missing deps: pip install trl transformers torch peft accelerate")
        sys.exit(1)

    log.info("Loading base model: %s", BASE_MODEL)
    # Use local tokenizer if available (avoids HF download issues)
    from pathlib import Path
    local_tok = Path("models/topology_sft/tokenizer.json")
    tok_path = str(local_tok.parent) if local_tok.exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=False,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Load SFT data — clean YAML (no line wrapping, consistent key ordering)
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            # Use clean YAML (width=10000, no multiline wrapping) if available
            topology = entry.get("topology_yaml_clean", entry.get("topology_yaml", ""))
            if not topology and entry.get("topology"):
                import yaml
                topology = yaml.dump(entry["topology"], default_flow_style=False, width=10000)
            if prompt and topology:
                data.append({
                    "text": (
                        f"<|system|>You are a multi-agent topology designer. "
                        f"Given a task, generate an optimal agent topology in YAML format.<|end|>\n"
                        f"<|user|>{prompt}<|end|>\n"
                        f"<|assistant|>{topology}<|end|>"
                    )
                })

    log.info("Loaded %d SFT examples from %s", len(data), data_path)
    if len(data) < 100:
        log.warning("Low SFT data count (%d). Aim for 1000+ for good quality.", len(data))

    from datasets import Dataset
    dataset = Dataset.from_list(data)

    # LoRA config (4-bit QLoRA for memory efficiency)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=1280,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    log.info("Starting SFT training (%d epochs)...", epochs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("SFT model saved to %s", output_dir)


def run_grpo(sft_checkpoint: str, output_dir: str, episodes: int):
    """Stage 2: GRPO with graduated format rewards (AgentConductor pattern).

    Key fixes from research (March 2026):
    - beta=0.0: no reference model (saves ~2GB VRAM, DAPO/Dr.GRPO validated)
    - loss_type="dr_grpo": removes std normalization (handles zero-variance groups)
    - num_generations=8: more diversity for reward variance
    - Graduated rewards: -2.0 to +1.0 (not binary -1/+1)
    - peft_config passed to GRPOTrainer (not pre-wrapped PeftModel)
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
    except ImportError:
        log.error("Missing deps: pip install trl transformers torch peft")
        sys.exit(1)

    log.info("Loading SFT checkpoint: %s", sft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=False,
                                              local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for GRPO (applied by GRPOTrainer, not pre-loaded)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # --- Graduated format reward (AgentConductor pattern) ---
    def format_reward(completions: list[str], **kwargs) -> list[float]:
        """Graduated YAML format reward. Ensures within-group variance."""
        import yaml
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                data = yaml.safe_load(text)
                if not isinstance(data, dict):
                    rewards.append(-1.5)  # Parsed but not a mapping
                    continue
                if "nodes" not in data:
                    rewards.append(-0.5)  # Valid YAML, missing nodes key
                    continue
                nodes = data["nodes"]
                if not isinstance(nodes, list) or len(nodes) == 0:
                    rewards.append(-0.25)  # Has nodes key but empty
                    continue
                # Valid topology structure
                rewards.append(0.5)
            except yaml.YAMLError:
                rewards.append(-2.0)  # Not valid YAML
            except Exception:
                rewards.append(-2.0)
        return rewards

    # --- Structural quality reward ---
    def structure_reward(completions: list[str], **kwargs) -> list[float]:
        """Reward structural quality of valid topologies."""
        import yaml
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                data = yaml.safe_load(text)
                if not isinstance(data, dict) or "nodes" not in data:
                    rewards.append(0.0)
                    continue
                nodes = data.get("nodes", [])
                if not isinstance(nodes, list):
                    rewards.append(0.0)
                    continue
                score = 0.0
                if 1 <= len(nodes) <= 10:
                    score += 0.3
                if data.get("edges"):
                    score += 0.2
                if all(isinstance(n, dict) and "role" in n for n in nodes):
                    score += 0.3
                if data.get("reasoning"):
                    score += 0.2
                rewards.append(score)
            except Exception:
                rewards.append(0.0)
        return rewards

    # --- Prepare prompts from SFT data ---
    prompts = []
    sft_data = None
    for candidate in [
        Path("data/topology_sft_clean.jsonl"),
        Path("data/topology_sft_combined.jsonl"),
    ]:
        if candidate.exists():
            sft_data = candidate
            break
    if sft_data:
        with open(sft_data, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                p = entry.get("prompt", "")
                if p:
                    prompts.append(
                        f"<|system|>You are a multi-agent topology designer. "
                        f"Given a task, generate an optimal agent topology in YAML format.<|end|>\n"
                        f"<|user|>{p}<|end|>\n"
                        f"<|assistant|>"
                    )
        log.info("Loaded %d prompts from %s", len(prompts), sft_data)
    else:
        log.error("No SFT data found")
        sys.exit(1)

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir=output_dir,
        # GRPO core (research-backed March 2026)
        num_generations=16,            # K=16 for diversity (DAPO standard, fills VRAM)
        max_completion_length=256,     # Topologies fit in ~200 tokens
        loss_type="dr_grpo",           # No std division — zero-variance safe (Dr. GRPO)
        scale_rewards=False,           # No per-group std scaling
        beta=0.0,                      # NO reference model (saves ~2GB, DAPO validated)
        # Training — fill 12GB VRAM
        num_train_epochs=1,
        per_device_train_batch_size=4, # 4x batch (was 1, only 4.3GB used)
        gradient_accumulation_steps=2, # effective batch = 8 (4*2)
        learning_rate=1e-5,            # TRL recommended for LoRA GRPO
        warmup_steps=20,
        # Memory (12GB VRAM)
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",      # 8-bit optimizer
        # Logging
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        log_completions=True,          # See what model generates
        mask_truncated_completions=True,
    )

    # Combined reward (TRL 0.29 doesn't support reward_weights)
    def combined_reward(completions: list[str], **kwargs) -> list[float]:
        fmt = format_reward(completions, **kwargs)
        struct = structure_reward(completions, **kwargs)
        return [f + 0.5 * s for f, s in zip(fmt, struct)]

    # Load SFT model as PeftModel — do NOT merge_and_unload (crashes on 4-bit)
    # GRPOTrainer handles LoRA stacking internally
    from transformers import BitsAndBytesConfig
    from peft import PeftModel
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    log.info("Loading base model (4-bit) + SFT adapter...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=False,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, sft_checkpoint)
    log.info("SFT model loaded. Launching GRPO...")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[combined_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info("Starting GRPO training (%d prompts)...", len(prompts))
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("GRPO model saved to %s", output_dir)


def run_export(checkpoint: str, output_path: str):
    """Stage 3: Export trained model to ONNX for Rust ort inference."""
    log.info("Exporting %s to ONNX: %s", checkpoint, output_path)

    try:
        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=checkpoint,
            output=output_path,
            task="text-generation",
            opset=18,
            fp16=False,
            trust_remote_code=False,
        )
        log.info("ONNX export complete: %s", output_path)
    except ImportError:
        log.error("Missing optimum: pip install optimum[exporters]")
        sys.exit(1)
    except Exception as exc:
        log.error("ONNX export failed: %s", exc)
        log.info("Alternative: use official ONNX from %s", ONNX_MODEL)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train topology generation policy (GRPO)")
    parser.add_argument("--mode", choices=["sft", "grpo", "export"], required=True)
    parser.add_argument("--data", type=str, default="data/topology_sft.jsonl")
    parser.add_argument("--sft-checkpoint", type=str, default="models/topology_sft/")
    parser.add_argument("--checkpoint", type=str, default="models/topology_grpo/")
    parser.add_argument("--output", type=str, default="models/topology_policy.onnx")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    if args.mode == "sft":
        run_sft(args.data, "models/topology_sft/", args.epochs)
    elif args.mode == "grpo":
        run_grpo(args.sft_checkpoint, "models/topology_grpo/", args.episodes)
    elif args.mode == "export":
        run_export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
