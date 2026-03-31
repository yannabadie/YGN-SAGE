#!/usr/bin/env python3
"""V2 Training via TRL GRPOTrainer with environment_factory.

Uses TRL's NATIVE multi-turn support instead of custom training loop.
The model calls create_topology and adapt_topology as tools,
TRL handles rollouts, gradients, and advantage computation.

Reference: https://github.com/huggingface/trl/blob/main/docs/source/openenv.md

Usage:
  # Full V2 on pod (H200)
  python scripts/train_v2_trl.py \
    --model /home/yann/qwen3_4b_base \
    --adapter /home/yann/v2_training/sft_checkpoint \
    --data data/v2_final.jsonl \
    --output /home/yann/v2_training/grpo_v2

  # Smoke test
  python scripts/train_v2_trl.py \
    --model /home/yann/qwen3_4b_base \
    --adapter /home/yann/v2_training/sft_checkpoint \
    --data data/v2_final.jsonl \
    --output /tmp/smoke --smoke
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import math
import os
import re
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("train_v2_trl")

# ── Reward weights (V2 spec) ──
W_STRUCTURAL = 0.20
W_EXECUTION = 0.35
W_RESILIENCE = 0.15
W_COST = 0.10
# W_REWARDFLOW = 0.20 computed at batch level

BUDGET_REF = {"simple": 0.01, "moderate": 0.05, "complex": 0.20}

TIER_UPGRADE = {"budget": "fast", "fast": "balanced", "balanced": "reasoner", "reasoner": "codex", "codex": "codex"}

TIER_PROVIDERS = {
    "budget": [("deepseek", "deepseek-chat"), ("kimi", "kimi-k2.5"), ("minimax", "minimax-m2.7")],
    "fast": [("google", "gemini-3.1-flash-lite-preview"), ("deepseek", "deepseek-chat")],
    "balanced": [("xai", "grok-4-1-fast-reasoning"), ("openrouter", "qwen/qwen3.5-plus-02-15")],
    "reasoner": [("openai", "gpt-5.4"), ("google", "gemini-3.1-pro-preview"), ("kimi", "kimi-k2.5")],
    "codex": [("openai", "gpt-5.4"), ("google", "gemini-3.1-pro-preview")],
}


def _resolve_provider(tier: str):
    """Resolve tier to (provider_name, model_id, base_url, api_key)."""
    from sage.providers.connector import PROVIDER_CONFIGS
    for prov_name, model_id in TIER_PROVIDERS.get(tier, TIER_PROVIDERS["budget"]):
        cfg = next((c for c in PROVIDER_CONFIGS if c["provider"] == prov_name), None)
        if cfg:
            api_key = os.environ.get(cfg["api_key_env"], "")
            if api_key:
                return prov_name, model_id, cfg["base_url"], api_key
    return None, None, None, None


def _call_provider(tier: str, prompt: str, system_prompt: str = "", timeout: float = 20.0) -> str:
    """Call a provider based on tier. Returns response text or empty string."""
    prov_name, model_id, base_url, api_key = _resolve_provider(tier)
    if not prov_name:
        return f"[no provider for tier={tier}]"
    try:
        import httpx
        r = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt or f"You are a {tier}-tier agent."},
                    {"role": "user", "content": prompt[:3000]},
                ],
                "max_tokens": 1024,
            },
            verify=False, timeout=timeout,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.debug(f"Provider {prov_name}/{model_id} failed: {e}")
    return f"[execution failed: {prov_name}/{model_id}]"


class SageTopologyToolEnv:
    """TRL-compatible tool environment for SAGE topology training.

    The model calls create_topology() and optionally adapt_topology().
    TRL handles the multi-turn orchestration automatically.
    """

    def __init__(self):
        self.reward = 0.0
        self._topology = None
        self._difficulty = "moderate"
        self._node_outputs = []
        self._adaptations = 0
        self._cost = 0.0
        self._exec_success = False

    def reset(self, **kwargs) -> str | None:
        """Reset env state for new episode."""
        self.reward = 0.0
        self._topology = None
        self._difficulty = "moderate"
        self._node_outputs = []
        self._adaptations = 0
        self._cost = 0.0
        self._exec_success = False
        return None  # No extra system message

    def create_topology(
        self,
        difficulty: str,
        reasoning: str,
        nodes: list[dict],
        edges: list[dict],
        checkpoints: list[int] | None = None,
        max_upgrades: int = 1,
        quality_threshold: float = 0.6,
    ) -> str:
        """Design a multi-agent DAG topology to solve a coding task.

        Args:
            difficulty: Task difficulty — simple, moderate, or complex
            reasoning: Why this topology is optimal for the task
            nodes: List of agent nodes, each with role, model_tier, prompt, fallback_tier
            edges: List of edges with from_idx, to_idx, flow_type
            checkpoints: Node indices where quality is checked
            max_upgrades: Maximum model tier upgrades allowed
            quality_threshold: Minimum quality score to continue

        Returns:
            Execution result summary with per-node outputs.
        """
        self._difficulty = difficulty
        self._topology = {
            "difficulty": difficulty,
            "reasoning": reasoning,
            "nodes": nodes,
            "edges": edges,
            "checkpoints": checkpoints or [],
            "max_upgrades": max_upgrades,
            "quality_threshold": quality_threshold,
        }

        # Structural reward
        r_struct = 0.0
        if 1 <= len(nodes) <= 10:
            r_struct += 0.3
        if edges:
            r_struct += 0.2
        if all(isinstance(n, dict) and "role" in n for n in nodes):
            r_struct += 0.3
        if reasoning and len(reasoning) > 20:
            r_struct += 0.2

        # Execute nodes via multi-provider
        exec_mode = os.environ.get("SAGE_VERL_EXEC", "0") == "1"
        results = []
        task_prompt = ""  # Will be filled from the conversation context

        for i, node in enumerate(nodes):
            role = node.get("role", f"node-{i}")
            tier = node.get("model_tier", "budget")
            node_prompt = node.get("prompt", f"You are {role}")

            if exec_mode:
                # Build context from predecessors
                pred_context = "\n".join(
                    f"[{self._node_outputs[j]['role']}]: {self._node_outputs[j]['output'][:300]}"
                    for j in range(i) if j < len(self._node_outputs)
                )
                full_prompt = f"{pred_context}\n\n{node_prompt}" if pred_context else node_prompt
                t0 = time.time()
                output = _call_provider(tier, full_prompt, node_prompt, timeout=20.0)
                latency = time.time() - t0
                self._cost += latency * 0.00001
            else:
                output = f"[structural mode] {role} output"
                latency = 0.0

            self._node_outputs.append({"role": role, "output": output, "tier": tier, "latency": latency})
            results.append(f"Node {i} ({role}, {tier}): {output[:200]}")

        # Check execution quality
        if exec_mode and self._node_outputs:
            last_output = self._node_outputs[-1]["output"]
            if "def " in last_output or "class " in last_output or "return " in last_output:
                self._exec_success = True

        # Compute reward
        r_exec = 0.5 if self._exec_success else 0.0
        r_cost = 1.0 - math.tanh(self._cost / BUDGET_REF.get(difficulty, 0.05))
        self.reward = W_STRUCTURAL * r_struct + W_EXECUTION * r_exec + W_COST * r_cost

        # Return summary for model context (enables multi-turn)
        summary = f"Topology executed: {len(nodes)} nodes, {len(edges)} edges.\n"
        for r in results:
            summary += f"  {r}\n"

        # If checkpoints exist, report quality for adaptation decisions
        if checkpoints:
            for cp_idx in checkpoints:
                if cp_idx < len(self._node_outputs):
                    node_out = self._node_outputs[cp_idx]
                    quality = 0.7 if len(node_out["output"]) > 100 else 0.3
                    summary += f"\nCheckpoint node {cp_idx} ({node_out['role']}): quality={quality:.2f}, threshold={quality_threshold}"
                    if quality < quality_threshold:
                        summary += f"\n  → Upgrade available: {node_out['tier']} → {TIER_UPGRADE.get(node_out['tier'], 'codex')}"

        return summary

    def adapt_topology(
        self,
        action: str,
        reason: str,
        node_idx: int | None = None,
        new_tier: str | None = None,
    ) -> str:
        """Runtime adaptation decision at a checkpoint.

        Args:
            action: One of continue, upgrade, reroute
            reason: Why this action was chosen
            node_idx: Which node to adapt (for upgrade/reroute)
            new_tier: New model tier (for upgrade)

        Returns:
            Result of the adaptation action.
        """
        self._adaptations += 1

        if action == "upgrade" and node_idx is not None and self._topology:
            nodes = self._topology.get("nodes", [])
            if node_idx < len(nodes):
                old_tier = nodes[node_idx].get("model_tier", "budget")
                actual_new_tier = new_tier or TIER_UPGRADE.get(old_tier, "reasoner")

                # Re-execute with upgraded tier
                exec_mode = os.environ.get("SAGE_VERL_EXEC", "0") == "1"
                if exec_mode:
                    node = nodes[node_idx]
                    output = _call_provider(actual_new_tier, node.get("prompt", ""), timeout=20.0)
                    if node_idx < len(self._node_outputs):
                        old_len = len(self._node_outputs[node_idx]["output"])
                        self._node_outputs[node_idx] = {
                            "role": node.get("role", "agent"),
                            "output": output,
                            "tier": actual_new_tier,
                            "latency": 0.0,
                        }
                        # Resilience bonus if output improved
                        if len(output) > old_len * 1.5:
                            self.reward += W_RESILIENCE * 0.3
                            self._exec_success = True

                return f"Upgraded node {node_idx} from {old_tier} to {actual_new_tier}. Reason: {reason}"

        elif action == "continue":
            return f"Continuing execution. Reason: {reason}"

        elif action == "reroute":
            return f"Rerouting node {node_idx}. Reason: {reason}"

        return f"Action {action} applied. Reason: {reason}"


def reward_func(environments, **kwargs):
    """Collect rewards from all environments."""
    return [env.reward for env in environments]


def load_dataset_for_trl(data_path: str, max_samples: int = 0):
    """Load v2_final.jsonl as TRL-compatible dataset."""
    from datasets import Dataset

    prompts = []
    seen = set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Extract user prompt
            if "turns" in entry and isinstance(entry["turns"], list):
                for turn in entry["turns"]:
                    if turn.get("role") == "user":
                        prompt_text = turn["content"]
                        break
                else:
                    continue
            elif "prompt" in entry:
                prompt_text = entry["prompt"]
            else:
                continue

            h = hashlib.md5(prompt_text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            prompts.append([
                {"role": "user", "content": prompt_text},
            ])

    if max_samples > 0:
        prompts = prompts[:max_samples]

    log.info(f"Loaded {len(prompts)} unique prompts from {data_path}")
    return Dataset.from_dict({"prompt": prompts})


def main():
    parser = argparse.ArgumentParser(description="V2 Training via TRL GRPOTrainer")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="/home/yann/v2_training/grpo_v2")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 16

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv("/workspace/YGN-SAGE/.env")
    except Exception:
        pass

    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings()

    # Load dataset
    dataset = load_dataset_for_trl(args.data, args.max_samples)

    # Setup TRL config
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig

    config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_generations=4,
        max_completion_length=2048,  # Multi-turn needs more tokens
        logging_steps=1,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    log.info("=" * 60)
    log.info("V2 Training via TRL GRPOTrainer + environment_factory")
    log.info(f"  Model: {args.model}")
    log.info(f"  Adapter: {args.adapter}")
    log.info(f"  Prompts: {len(dataset)}")
    log.info(f"  K (num_generations): {config.num_generations}")
    log.info(f"  Output: {args.output}")
    log.info(f"  Exec mode: {os.environ.get('SAGE_VERL_EXEC', '0')}")
    log.info("=" * 60)

    trainer = GRPOTrainer(
        model=args.adapter,  # Loads base + LoRA automatically
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
        environment_factory=SageTopologyToolEnv,
    )

    trainer.train()

    # Save
    trainer.save_model(args.output)
    log.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
