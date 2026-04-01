#!/usr/bin/env python3
"""V2 GRPO Training — TRL native with environment_factory.

Uses TRL's GRPOTrainer + environment_factory for multi-turn tool-calling.
The model calls create_topology() and adapt_topology() as tools.
TRL handles rollouts, KL, advantages, and gradients automatically.

This replaces train_v2_multiturn.py (custom loop, broken) and
train_v2_trl.py (wrong API usage).

Usage:
  # Smoke test (16 prompts, CPU-compatible)
  python scripts/train_v2_grpo.py --smoke

  # Full training (500 prompts, H200)
  SAGE_VERL_EXEC=1 python scripts/train_v2_grpo.py \
    --model /home/yann/qwen3_4b_base \
    --adapter /home/yann/v2_training/sft_checkpoint \
    --data data/v2_final.jsonl \
    --output /home/yann/v2_training/grpo_v2 \
    --max-samples 500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("train_v2_grpo")


# ═══════════════════════════════════════════════════════════════
# 1. TOPOLOGY ENVIRONMENT — TRL environment_factory pattern
# ═══════════════════════════════════════════════════════════════

class SageTopologyEnv:
    """Multi-turn topology environment for TRL GRPOTrainer.

    Exposes create_topology and adapt_topology as tools.
    TRL auto-discovers these methods and makes them available to the model.
    The model calls them as tool_calls; TRL handles the orchestration.

    Pattern: https://huggingface.co/docs/trl/openenv
    """

    def __init__(self):
        self.reward = 0.0
        self.topology = None
        self.turn = 0
        self.difficulty = "moderate"

    def reset(self, **kwargs) -> str | None:
        """Reset environment state. Called once per episode."""
        self.reward = 0.0
        self.topology = None
        self.turn = 0
        self.difficulty = "moderate"
        return None  # No prefix to add to user message

    def create_topology(
        self,
        reasoning: str,
        difficulty: str,
        nodes: list[dict],
        edges: list[dict] | None = None,
        checkpoints: list[int] | None = None,
        max_upgrades: int = 1,
        quality_threshold: float = 0.6,
    ) -> str:
        """Create a multi-agent topology for the given task.

        Args:
            reasoning: Why this topology design is appropriate for the task.
            difficulty: Task difficulty level (simple, moderate, complex).
            nodes: List of agent nodes. Each has role (str), model_tier (str: budget/fast/balanced/reasoner/codex), prompt (str), and optionally fallback_tier (str).
            edges: List of edges connecting nodes. Each has from_idx (int), to_idx (int), and flow_type (str: message/control/state).
            checkpoints: Node indices where quality is checked.
            max_upgrades: Maximum model tier upgrades allowed.
            quality_threshold: Minimum quality score to continue without upgrade.

        Returns:
            Confirmation message with structural score.
        """
        edges = edges or []
        checkpoints = checkpoints or []
        self.difficulty = difficulty
        self.turn += 1

        # Build the topology dict and score it
        topology = {
            "reasoning": reasoning,
            "difficulty": difficulty,
            "nodes": nodes,
            "edges": edges,
            "checkpoints": checkpoints,
            "max_upgrades": max_upgrades,
            "quality_threshold": quality_threshold,
        }
        self.topology = topology

        # Score using compute_score (structural mode)
        tc_str = f'<tool_call>{json.dumps({"name": "create_topology", "arguments": topology})}</tool_call>'
        try:
            from sage.verl.reward import compute_score
            self.reward = float(compute_score("sage_topology", tc_str, "", {}))
        except Exception as e:
            log.warning(f"Reward computation failed: {e}")
            self.reward = 0.0

        n_nodes = len(nodes)
        n_edges = len(edges)
        return (
            f"Topology created: {n_nodes} nodes, {n_edges} edges, difficulty={difficulty}. "
            f"Structural score: {self.reward:.2f}/1.0. "
            f"{'PASSED' if self.reward >= 0.5 else 'NEEDS IMPROVEMENT'}"
        )

    def adapt_topology(
        self,
        action: str,
        reason: str,
        node_idx: int | None = None,
        new_model_tier: str | None = None,
    ) -> str:
        """Adapt a running topology based on checkpoint feedback.

        Args:
            action: One of 'continue', 'upgrade', 'reroute', 'prune'.
            reason: Why this adaptation decision is appropriate.
            node_idx: Index of the node to adapt (for upgrade/prune).
            new_model_tier: New model tier for upgrade action.

        Returns:
            Result of the adaptation action.
        """
        self.turn += 1

        if action == "continue":
            # Small bonus for correct continue decision
            self.reward = min(self.reward + 0.05, 1.5)
            return f"Continuing execution. Decision: {reason}"
        elif action == "upgrade" and node_idx is not None:
            # Bonus for upgrade when quality was low
            self.reward = min(self.reward + 0.1, 1.5)
            tier = new_model_tier or "reasoner"
            return f"Upgraded node {node_idx} to {tier}. Reason: {reason}"
        elif action == "reroute":
            self.reward = min(self.reward + 0.08, 1.5)
            return f"Rerouting topology. Reason: {reason}"
        elif action == "prune" and node_idx is not None:
            self.reward = min(self.reward + 0.03, 1.5)
            return f"Pruned node {node_idx}. Reason: {reason}"
        else:
            return f"Unknown action: {action}. No changes made."


def topology_reward(environments, **kwargs) -> list[float]:
    """Extract rewards from environment instances.

    TRL calls this after each episode completes.
    The signature MUST accept `environments` as first arg.
    """
    return [env.reward for env in environments]


# ═══════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a topology orchestrator for the SAGE multi-agent system.
Your job: given a task description, call create_topology() to design the optimal multi-agent DAG.
If you receive checkpoint feedback, call adapt_topology() to make runtime decisions.

Design principles:
- Simple tasks: 1-2 nodes, budget tier, sequential
- Moderate tasks: 2-4 nodes, mixed tiers, reviewer pattern
- Complex tasks: 3-7 nodes, reasoner/codex tier, mesh/parallel patterns
- Every node needs: role, model_tier, prompt, fallback_tier
- Edges define data flow between nodes (message/control/state)"""


def load_dataset(jsonl_path: str, max_samples: int = 0):
    """Load v2_final.jsonl as TRL-compatible dataset.

    Extracts user prompts and formats as chat messages.
    """
    from datasets import Dataset

    prompts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Extract user prompt
            user_content = None
            if "turns" in entry and isinstance(entry["turns"], list):
                for turn in entry["turns"]:
                    if turn.get("role") == "user":
                        user_content = turn["content"]
                        break
            elif "prompt" in entry:
                user_content = entry["prompt"]

            if user_content:
                prompts.append(user_content)

    if max_samples > 0:
        prompts = prompts[:max_samples]

    # Format as chat messages for TRL
    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]
    })

    log.info(f"Loaded {len(dataset)} prompts")
    return dataset


# ═══════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/yann/qwen3_4b_base")
    parser.add_argument("--adapter", default="/home/yann/v2_training/sft_checkpoint")
    parser.add_argument("--data", default="data/v2_final.jsonl")
    parser.add_argument("--output", default="/home/yann/v2_training/grpo_v2")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 16
        args.output = "/tmp/smoke_grpo"

    os.makedirs(args.output, exist_ok=True)

    # ── Load dataset ──
    dataset = load_dataset(args.data, args.max_samples)

    # ── Load model with adapter ──
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig

    log.info(f"Loading base model: {args.model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Merge existing SFT adapter into base
    if args.adapter and os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        log.info(f"Merging SFT adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        log.info("Adapter merged.")

    # ── Training config ──
    from trl import GRPOConfig, GRPOTrainer

    smoke_overrides = {}
    if args.smoke:
        smoke_overrides = {
            "max_steps": 5,
            "num_generations": 2,
            "per_device_train_batch_size": 2,
            "max_completion_length": 512,
        }

    config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=smoke_overrides.get("per_device_train_batch_size", 4),
        num_generations=smoke_overrides.get("num_generations", 4),
        max_completion_length=smoke_overrides.get("max_completion_length", 2048),
        max_steps=smoke_overrides.get("max_steps", -1),
        logging_steps=1,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        report_to="none",
    )

    # New LoRA config for GRPO training (on top of merged SFT)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    log.info("=" * 60)
    log.info("V2 GRPO Training — TRL native + environment_factory")
    log.info(f"  Model: {args.model}")
    log.info(f"  Adapter: {args.adapter}")
    log.info(f"  Prompts: {len(dataset)}")
    log.info(f"  K (num_generations): {config.num_generations}")
    log.info(f"  LR: {config.learning_rate}")
    log.info(f"  Batch size: {config.per_device_train_batch_size}")
    log.info(f"  Max completion: {config.max_completion_length}")
    log.info(f"  Output: {args.output}")
    log.info(f"  Smoke: {args.smoke}")
    log.info("=" * 60)

    # ── Create trainer with environment ──
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=topology_reward,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
        environment_factory=SageTopologyEnv,
    )

    log.info("Trainer created. Starting training...")
    trainer.train()

    # ── Save ──
    trainer.save_model(args.output)
    log.info(f"Model saved to {args.output}")

    # ── Quick eval ──
    log.info("Training complete.")


if __name__ == "__main__":
    main()
