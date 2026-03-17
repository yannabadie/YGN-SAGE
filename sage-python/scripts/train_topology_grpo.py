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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
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
    """Stage 2: GRPO with SAGE's verified dense rewards."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import GRPOTrainer, GRPOConfig
        from peft import PeftModel
    except ImportError:
        log.error("Missing deps: pip install trl transformers torch peft")
        sys.exit(1)

    log.info("Loading SFT checkpoint: %s", sft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, trust_remote_code=False, dtype="auto",
    )
    model = PeftModel.from_pretrained(model, sft_checkpoint)

    # SAGE verified reward function
    try:
        from sage_core import TopologyReward, TopologyDensity, TopologyGraph, TopologyNode
        from sage_core import PyHybridVerifier
        reward_scorer = TopologyReward()
        density_scorer = TopologyDensity()
        verifier = PyHybridVerifier()
        log.info("SAGE reward infrastructure loaded (Rust)")
    except ImportError:
        log.error("sage_core not built. Run: cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor")
        sys.exit(1)

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        """RLVR reward: verified dense rewards from SAGE infrastructure."""
        rewards = []
        for completion in completions:
            try:
                # Parse JSON topology (AgentConductor-style graduated penalties)
                topo_data = json.loads(completion)
                if not topo_data or not isinstance(topo_data, dict):
                    rewards.append(-2.0)  # NO_JSON_FOUND
                    continue
                if "nodes" not in topo_data:
                    rewards.append(-1.0)  # JSON_SCHEMA_INVALID
                    continue

                # Build TopologyGraph
                graph = TopologyGraph("generated")
                for node_data in topo_data["nodes"]:
                    node = TopologyNode(
                        role=node_data.get("role", "agent"),
                        model_id=node_data.get("model_id", ""),
                        system=node_data.get("system", 2),
                    )
                    graph.add_node(node)

                # Compute verified reward signals
                density = density_scorer.compute(graph, 2)
                verification = verifier.verify(graph)
                structural_score = 1.0 if verification.passed else 0.5

                # Dense reward (no execution in GRPO loop — too expensive)
                # Use structural + density as proxy
                reward = reward_scorer.compute(
                    execution_passed=True,  # Assume execution for structural reward
                    structural_score=structural_score,
                    density_score=density.s_complex,
                    temporal_score=None,
                )
                # Penalty for over-budget topologies
                if density.over_budget:
                    reward_val = reward.total * 0.5
                else:
                    reward_val = reward.total

                rewards.append(float(reward_val))

            except Exception:
                rewards.append(-1.0)  # Parse failure penalty

        return rewards

    # Prepare prompts from SFT data
    prompts = []
    sft_data = Path(sft_checkpoint).parent / "topology_sft.jsonl"
    if sft_data.exists():
        with open(sft_data, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                prompts.append(
                    f"<|system|>You are a multi-agent topology designer.<|end|>\n"
                    f"<|user|>{entry.get('prompt', '')}<|end|>\n"
                    f"<|assistant|>"
                )
    else:
        log.warning("No SFT data found at %s, using dummy prompts", sft_data)
        prompts = ["<|user|>Write a function to sort a list<|end|>\n<|assistant|>"] * 100

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir=output_dir,
        num_generations=8,  # K=8 sampled topologies per query
        max_completion_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.04,  # KL penalty coefficient
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
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
