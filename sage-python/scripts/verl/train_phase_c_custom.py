#!/usr/bin/env python3
"""Phase C fallback: Custom multi-step training loop using SageTopologyEnv.

This script is the FALLBACK for Phase C when verl-agent multi-turn env
integration does not work. It implements GiGPO-style training directly
using PyTorch + PEFT, without verl's infrastructure.

How it works:
    1. Load model + LoRA from Phase A/B checkpoint
    2. Load curated training prompts
    3. For each epoch:
       a. For each batch of prompts, run K=4 rollouts per prompt:
          - reset env -> step(YAML) -> [step(decision) at checkpoints] -> terminal
       b. Collect StepRewardVector per rollout
       c. Compute GiGPO-style advantages:
          - Group steps by anchor key
          - Normalize within-group: A_i = (r_i - mean(group)) / (std(group) + eps)
       d. Apply REINFORCE-style gradient update:
          - loss = -sum(A_i * log_prob(action_i))
       e. Log metrics, save checkpoint

Compared to verl-agent:
    - Simpler: no Ray, no FSDP, no vLLM rollout engine
    - Slower: PyTorch generate() instead of vLLM
    - No token-level masking of observations (full sequence gets gradients)
    - But: works without verl-agent and exercises the full 4-state machine

Usage:
    python scripts/verl/train_phase_c_custom.py \\
        --model /workspace/patched_nemotron_orchestrator \\
        --checkpoint /workspace/topology_verl_output \\
        --data data/verl_topology_curated.parquet \\
        --output /workspace/topology_verl_phase_c_custom \\
        --epochs 3 --lr 5e-7 --k 4 --batch-size 8

Reference: GiGPO (arXiv 2505.10978) Section 3, Algorithm 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phase_c_custom")

# RewardFlow (PageRank per-node credit) + Graph-GRPO edge credit
from sage.verl.rewardflow import RewardFlowPropagator
from sage.verl.edge_credit import compute_edge_advantages, parse_edges_from_yaml

# HyEvo integration: reflect-then-generate + cascaded eval
from sage.verl.reflection import diagnose, format_reflection_prompt
from sage.verl.cascaded_eval import cascaded_evaluate
from sage.verl.topology_schema import TopologySchema


# ---------------------------------------------------------------------------
# GiGPO advantage computation
# ---------------------------------------------------------------------------

def compute_gigpo_advantages(
    all_step_rewards: list[list[float]],
    all_anchor_keys: list[list[str]],
    gamma: float = 0.95,
    eps: float = 1e-6,
) -> list[list[float]]:
    """Compute GiGPO step-level advantages across K rollouts.

    Groups steps by anchor key, normalizes within-group.
    This is the core of GiGPO: same anchor state -> compare decisions.

    Args:
        all_step_rewards: [K rollouts][N steps] rewards
        all_anchor_keys: [K rollouts][N steps] anchor strings
        gamma: discount factor for future rewards
        eps: numerical stability

    Returns:
        [K rollouts][N steps] advantages
    """
    # Collect all (anchor, reward) pairs across all rollouts
    anchor_groups: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for k, (rewards, anchors) in enumerate(zip(all_step_rewards, all_anchor_keys)):
        for t, (r, a) in enumerate(zip(rewards, anchors)):
            anchor_groups[a].append((k, t, r))

    # Compute within-group normalized advantages
    advantages = [[0.0] * len(rewards) for rewards in all_step_rewards]

    for anchor, entries in anchor_groups.items():
        if len(entries) < 2:
            # Need at least 2 samples for meaningful normalization
            for k, t, r in entries:
                advantages[k][t] = r
            continue

        rewards_in_group = [r for _, _, r in entries]
        mean_r = sum(rewards_in_group) / len(rewards_in_group)
        var_r = sum((r - mean_r) ** 2 for r in rewards_in_group) / len(rewards_in_group)
        std_r = var_r ** 0.5

        for k, t, r in entries:
            advantages[k][t] = (r - mean_r) / (std_r + eps)

    return advantages


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env,
    model,
    tokenizer,
    prompt: str,
    task_id: str,
    max_new_tokens: int = 768,
    temperature: float = 0.7,
    device: str = "cuda",
) -> dict:
    """Run one multi-step episode through the 4-state machine.

    Returns:
        {
            "prompt": str,
            "task_id": str,
            "actions": [str, ...],        # model outputs at each action step
            "action_log_probs": [Tensor],  # log probs for each action
            "step_rewards": [float, ...],
            "anchor_keys": [str, ...],
            "total_reward": float,
            "n_steps": int,
            "status": str,
        }
    """
    obs = env.reset(prompt, task_id)

    actions = []
    action_log_probs = []
    done = False
    step_count = 0
    max_steps = 10  # safety limit

    while not done and step_count < max_steps:
        # Build input for the model
        obs_text = obs.get("text", "")
        messages = [
            {"role": "system", "content": "You are a topology generator for a multi-agent system. "
             "Generate YAML topologies or make decisions (continue/upgrade/reroute) at checkpoints."},
            {"role": "user", "content": obs_text},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1280)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate with temperature sampling
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Extract generated tokens (exclude input)
        gen_ids = outputs.sequences[0, input_ids.shape[1]:]
        action_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Compute log probability of the generated sequence
        # Stack scores and compute log softmax
        if outputs.scores:
            log_probs_per_token = []
            for t_idx, score in enumerate(outputs.scores):
                if t_idx < len(gen_ids):
                    log_prob = torch.nn.functional.log_softmax(score[0], dim=-1)
                    token_log_prob = log_prob[gen_ids[t_idx]]
                    log_probs_per_token.append(token_log_prob)
            if log_probs_per_token:
                total_log_prob = torch.stack(log_probs_per_token).sum()
            else:
                total_log_prob = torch.tensor(0.0, device=device)
        else:
            total_log_prob = torch.tensor(0.0, device=device)

        actions.append(action_text)
        action_log_probs.append(total_log_prob)

        # Step the environment
        obs, reward, done, info = env.step(action_text)
        step_count += 1

    # Collect step rewards from the env
    srv = env.get_step_rewards()
    trace = env.get_trace()

    return {
        "prompt": prompt,
        "task_id": task_id,
        "actions": actions,
        "action_log_probs": action_log_probs,
        "step_rewards": srv.step_rewards,
        "anchor_keys": srv.anchor_keys,
        "total_reward": srv.episode_reward,
        "n_steps": srv.n_steps,
        "status": srv.status,
        # For batch-level RewardFlow + edge credit
        "node_traces_for_rewardflow": trace.node_traces_for_rewardflow,
        "topology_yaml": trace.topology_yaml,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_phase_c(
    model_path: str,
    checkpoint_path: str | None,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 5e-7,
    k_rollouts: int = 4,
    batch_size: int = 8,
    gamma: float = 0.95,
    max_new_tokens: int = 768,
    temperature: float = 0.7,
    memory_db: str = "",
    save_every: int = 50,
):
    """Phase C custom training loop.

    Loads model + optional LoRA, runs multi-step episodes via SageTopologyEnv,
    computes GiGPO advantages, updates model via REINFORCE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load model + LoRA ──────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Load existing LoRA from Phase A/B if available
    if checkpoint_path and Path(checkpoint_path).exists():
        log.info("Loading Phase A/B LoRA from %s", checkpoint_path)
        from peft import PeftModel
        try:
            # Try loading as PeftModel (LoRA checkpoint)
            model = PeftModel.from_pretrained(model, checkpoint_path)
            log.info("LoRA loaded from %s", checkpoint_path)
        except Exception as e:
            log.warning("Could not load LoRA from %s: %s", checkpoint_path, e)
            log.info("Applying fresh LoRA instead")
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=64, lora_alpha=32,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
    else:
        log.info("No checkpoint found, applying fresh LoRA")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=64, lora_alpha=32,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.train()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("Trainable: %d / %d (%.2f%%)", trainable_params, total_params,
             100 * trainable_params / total_params)

    # ── Optimizer ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # ── Load data ──────────────────────────────────────────────
    import pandas as pd
    df = pd.read_parquet(data_path)
    log.info("Training data: %d entries from %s", len(df), data_path)

    # Extract prompts - handle verl parquet format where "prompt" is an
    # array of message dicts [{role: system, content: ...}, {role: user, content: ...}]
    all_prompts = []
    if "prompt" in df.columns:
        for entry in df["prompt"]:
            if isinstance(entry, (list, np.ndarray)):
                # verl format: array of message dicts
                for m in entry:
                    if isinstance(m, dict) and m.get("role") == "user":
                        all_prompts.append(m["content"])
                        break
                else:
                    all_prompts.append(str(entry))
            elif isinstance(entry, str):
                all_prompts.append(entry)
            else:
                all_prompts.append(str(entry))
    elif "messages" in df.columns:
        for msgs in df["messages"]:
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            for m in msgs:
                if m.get("role") == "user":
                    all_prompts.append(m["content"])
                    break
            else:
                all_prompts.append(str(msgs))
    else:
        log.error("No 'prompt' or 'messages' column in data")
        sys.exit(1)

    # Extract task_ids from extra_info if present
    task_ids = []
    if "extra_info" in df.columns:
        for info in df["extra_info"]:
            if isinstance(info, dict):
                task_ids.append(info.get("task_id", f"t/{len(task_ids)}"))
            else:
                task_ids.append(f"t/{len(task_ids)}")
    elif "task_id" in df.columns:
        task_ids = df["task_id"].tolist()
    else:
        task_ids = [f"t/{i}" for i in range(len(df))]

    # ── Environment ────────────────────────────────────────────
    from sage.verl.topology_env import SageTopologyEnv
    env_config = {}
    if memory_db:
        env_config["memory_db"] = memory_db
    env = SageTopologyEnv(config=env_config)
    log.info("SageTopologyEnv initialized (memory_db=%s)", memory_db or "none")

    # ── Training metrics ───────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.jsonl")

    # ── Main training loop ─────────────────────────────────────
    global_step = 0

    for epoch in range(epochs):
        log.info("=== Epoch %d/%d ===", epoch + 1, epochs)

        # Shuffle prompts each epoch
        indices = np.random.permutation(len(all_prompts))

        epoch_rewards = []
        epoch_advantages = []
        epoch_decisions = {"continue": 0, "upgrade": 0, "reroute": 0}
        epoch_adaptations = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_prompts = [all_prompts[i] for i in batch_indices]
            batch_task_ids = [task_ids[i] for i in batch_indices]

            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_rewards_collected = []
            n_valid_rollouts = 0

            for prompt, task_id in zip(batch_prompts, batch_task_ids):
                # Run K rollouts for this prompt
                rollout_results = []

                for k in range(k_rollouts):
                    try:
                        result = run_episode(
                            env=env,
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            task_id=task_id,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            device=device,
                        )
                        rollout_results.append(result)
                    except Exception as exc:
                        log.warning("Rollout %d failed for %s: %s", k, task_id, exc)

                if len(rollout_results) < 2:
                    # Need at least 2 rollouts for GiGPO normalization
                    continue

                # Compute GiGPO advantages across K rollouts
                all_step_rewards = [list(r["step_rewards"]) for r in rollout_results]
                all_anchor_keys = [r["anchor_keys"] for r in rollout_results]

                # ── RewardFlow per-node credit (PageRank propagation) ──
                if len(rollout_results) >= 2:
                    try:
                        propagator = RewardFlowPropagator(damping=0.85, max_iters=20)
                        rollout_dicts = [
                            {
                                "node_traces": r["node_traces_for_rewardflow"],
                                "terminal_reward": r["total_reward"],
                            }
                            for r in rollout_results
                        ]
                        rewardflow_credits = propagator.compute(rollout_dicts)
                        # Add rewardflow credit to per-step rewards
                        for k_rf, credits in enumerate(rewardflow_credits):
                            for node_idx, rf_reward in credits.items():
                                # Find the step corresponding to this node
                                for step_idx, step in enumerate(rollout_results[k_rf].get("step_rewards", [])):
                                    # Steps include topology_generator (node_idx=-1) and
                                    # actual nodes (node_idx=0,1,...). Match by offset:
                                    # step 0 = topology, steps 1..N = nodes
                                    if step_idx - 1 == node_idx and step_idx < len(all_step_rewards[k_rf]):
                                        all_step_rewards[k_rf][step_idx] += 0.2 * rf_reward
                    except Exception as exc:
                        log.warning("RewardFlow failed: %s", exc)

                # ── Graph-GRPO edge credit (per-edge advantage) ──────
                if len(rollout_results) >= 2:
                    try:
                        edge_data = []
                        for r in rollout_results:
                            edges = parse_edges_from_yaml(r.get("topology_yaml", ""))
                            edge_data.append({"edges": edges, "reward": r["total_reward"]})

                        edge_advantages = compute_edge_advantages(edge_data)
                        # Adjust topology generation step (step 0) by edge advantage
                        for k_ec, ed in enumerate(edge_data):
                            edges = ed["edges"]
                            if edges and edge_advantages:
                                edge_bonus = sum(
                                    edge_advantages.get(tuple(e), 0.0) for e in edges
                                ) / max(len(edges), 1)
                                if all_step_rewards[k_ec]:
                                    all_step_rewards[k_ec][0] += 0.1 * edge_bonus
                    except Exception as exc:
                        log.warning("Edge credit failed: %s", exc)

                # ── HyEvo: Reflect on worst rollout vs best ────────
                try:
                    best_r = max(rollout_results, key=lambda r: r["total_reward"])
                    worst_r = min(rollout_results, key=lambda r: r["total_reward"])
                    if worst_r["total_reward"] < best_r["total_reward"] * 0.5:
                        diag = diagnose(
                            parent_yaml=worst_r.get("topology_yaml", ""),
                            parent_score=worst_r["total_reward"],
                            parent_traces=worst_r.get("node_traces_for_rewardflow", []),
                            top_yaml=best_r.get("topology_yaml", ""),
                            top_score=best_r["total_reward"],
                        )
                        if diag.recommendations:
                            log.info("Reflect: %s", diag.summary)
                except Exception as exc:
                    log.debug("Reflect failed: %s", exc)

                advantages = compute_gigpo_advantages(
                    all_step_rewards, all_anchor_keys, gamma=gamma,
                )

                # REINFORCE loss: -sum(advantage * log_prob)
                for k_idx, (result, advs) in enumerate(zip(rollout_results, advantages)):
                    log_probs = result["action_log_probs"]
                    if not log_probs:
                        continue

                    # Map step advantages to action-level advantages
                    # Each action corresponds to one or more env steps
                    # Simplification: use mean advantage per action
                    n_actions = len(log_probs)
                    n_steps = len(advs)

                    if n_steps == 0 or n_actions == 0:
                        continue

                    # Distribute step advantages across actions
                    steps_per_action = max(1, n_steps // n_actions)
                    action_advantages = []
                    for a_idx in range(n_actions):
                        start = a_idx * steps_per_action
                        end = min(start + steps_per_action, n_steps)
                        if start < n_steps:
                            action_adv = sum(advs[start:end]) / max(end - start, 1)
                        else:
                            action_adv = 0.0
                        action_advantages.append(action_adv)

                    # Compute policy gradient loss
                    for a_idx, (lp, adv) in enumerate(zip(log_probs, action_advantages)):
                        if abs(adv) > 1e-8:  # skip zero-advantage actions
                            batch_loss = batch_loss + (-adv * lp)
                            n_valid_rollouts += 1

                    batch_rewards_collected.append(result["total_reward"])
                    epoch_rewards.append(result["total_reward"])

                    # Track decisions
                    for action in result["actions"]:
                        a_lower = action.strip().lower()
                        if "upgrade" in a_lower:
                            epoch_decisions["upgrade"] += 1
                        elif "reroute" in a_lower:
                            epoch_decisions["reroute"] += 1
                        elif "continue" in a_lower:
                            epoch_decisions["continue"] += 1

                    if result["status"] == "PASSED" and any("upgrade" in a.lower() for a in result["actions"]):
                        epoch_adaptations += 1

            # Gradient update
            if n_valid_rollouts > 0:
                loss_val = batch_loss / n_valid_rollouts
                loss_val.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )

                optimizer.step()
                optimizer.zero_grad()

                mean_reward = np.mean(batch_rewards_collected) if batch_rewards_collected else 0.0
                log.info(
                    "Step %d | loss=%.4f | reward_mean=%.3f | rollouts=%d",
                    global_step, loss_val.item(), mean_reward, n_valid_rollouts,
                )

                # Log metrics
                with open(metrics_file, "a") as f:
                    f.write(json.dumps({
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss_val.item(),
                        "reward_mean": mean_reward,
                        "n_rollouts": n_valid_rollouts,
                        "decisions": dict(epoch_decisions),
                    }) + "\n")
            else:
                optimizer.zero_grad()
                log.warning("Step %d: no valid rollouts in batch", global_step)

            global_step += 1

            # Periodic save
            if global_step % save_every == 0:
                save_path = os.path.join(output_dir, f"checkpoint_step_{global_step}")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                log.info("Checkpoint saved to %s", save_path)

        # End of epoch summary
        if epoch_rewards:
            log.info(
                "Epoch %d summary: reward_mean=%.3f, reward_std=%.3f, "
                "decisions=%s, adaptations_succeeded=%d",
                epoch + 1,
                np.mean(epoch_rewards),
                np.std(epoch_rewards),
                epoch_decisions,
                epoch_adaptations,
            )
        else:
            log.warning("Epoch %d: no successful episodes", epoch + 1)

    # ── Final save ─────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    log.info("Final model saved to %s", final_path)
    log.info("Metrics log: %s", metrics_file)

    # Print summary
    log.info("")
    log.info("=== Phase C Custom Training Complete ===")
    log.info("Next: python scripts/verl/post_training_pipeline.py all")


def main():
    parser = argparse.ArgumentParser(
        description="Phase C fallback: custom multi-step GiGPO training"
    )
    parser.add_argument(
        "--model", default="/workspace/patched_nemotron_orchestrator",
        help="Base model path (Nemotron-Orchestrator-8B)",
    )
    parser.add_argument(
        "--checkpoint", default="/workspace/topology_verl_output",
        help="Phase A/B checkpoint dir (LoRA adapter)",
    )
    parser.add_argument(
        "--data", default="data/verl_topology_curated.parquet",
        help="Training data (curated parquet)",
    )
    parser.add_argument(
        "--output", default="/workspace/topology_verl_phase_c_custom",
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--k", type=int, default=4, help="Rollouts per prompt for GiGPO")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--memory-db", default="", help="SQLite episodic memory DB path")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")

    args = parser.parse_args()

    train_phase_c(
        model_path=args.model,
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        k_rollouts=args.k,
        batch_size=args.batch_size,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        memory_db=args.memory_db,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
