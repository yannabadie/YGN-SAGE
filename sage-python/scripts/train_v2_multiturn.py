#!/usr/bin/env python3
"""V2 Multi-Turn GRPO Training — The REAL V2 pipeline.

Unlike train_local_qwen3_4b.py (single-turn GRPO via TRL), this script
implements the full V2 spec from 2026-03-20-v2-adaptive-topology-design.md:

1. SageTopologyEnv multi-turn episodes (generate → checkpoint → adapt)
2. 5-signal reward (structural + execution + rewardflow + resilience + cost)
3. Episodic memory (SQLite, query at reset, store at terminal)
4. RewardFlow PageRank per-node credit
5. Custom GRPO loop (not TRL GRPOTrainer — it can't do multi-turn)

Flow per prompt:
  1. Query episodic memory → inject context
  2. Model generates topology (Turn 0)
  3. Execute nodes incrementally
  4. At checkpoints: model decides continue/upgrade (Turn 1..N)
  5. Terminal: sandbox test → 5-signal reward
  6. RewardFlow across K rollouts → per-node credit
  7. GRPO advantage = (reward - mean(K)) / std(K)
  8. Gradient update on all turns (topology + decisions)

Usage:
  # Full V2 on pod (H200)
  python scripts/train_v2_multiturn.py \
    --model /home/yann/qwen3_4b_base \
    --adapter /home/yann/v2_training/sft_checkpoint \
    --data sage-python/data/v2_final.jsonl \
    --output /home/yann/v2_training/grpo_v2 \
    --epochs 1 --k 4 --lr 5e-6

  # Smoke test
  python scripts/train_v2_multiturn.py \
    --model /home/yann/qwen3_4b_base \
    --adapter /home/yann/v2_training/sft_checkpoint \
    --data sage-python/data/v2_final.jsonl \
    --output /tmp/smoke --max-samples 4 --k 2 --smoke
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("train_v2")

# ── V2 Reward weights (from spec, subject to ablation) ──────────
W_STRUCTURAL = 0.20
W_EXECUTION = 0.35
W_REWARDFLOW = 0.20
W_RESILIENCE = 0.15
W_COST = 0.10

# Budget reference for cost efficiency (CARD-style)
BUDGET_REF = {"simple": 0.01, "moderate": 0.05, "complex": 0.20}


@dataclass
class Rollout:
    """One complete episode (multi-turn)."""
    prompt: str
    task_id: str
    difficulty: str
    turns: list[dict] = field(default_factory=list)  # [{role, content, log_probs, tokens}]
    reward_structural: float = 0.0
    reward_execution: float = 0.0
    reward_rewardflow: float = 0.0
    reward_resilience: float = 0.0
    reward_cost: float = 0.0
    reward_total: float = 0.0
    node_traces: list[dict] = field(default_factory=list)
    adaptations_triggered: int = 0
    cost_usd: float = 0.0
    outcome: str = ""


def load_prompts(data_path: str, max_samples: int = 0) -> list[dict]:
    """Load prompts from v2_final.jsonl."""
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

            # Deduplicate
            h = hashlib.md5(prompt_text.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)

            prompts.append({
                "prompt": prompt_text,
                "task_id": entry.get("task_id", h[:8]),
                "difficulty": entry.get("difficulty", "moderate"),
                "system_prompt": entry.get("system_prompt", ""),
            })

    if max_samples > 0:
        prompts = prompts[:max_samples]

    log.info(f"Loaded {len(prompts)} unique prompts from {data_path}")
    return prompts


def load_model(model_path: str, adapter_path: str, quantize_4bit: bool = False):
    """Load base model + LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True, local_files_only=True,
    )

    log.info(f"Loading base model from {model_path}...")
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb,
            trust_remote_code=True, device_map="auto",
            local_files_only=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto",
            local_files_only=True,
        )

    log.info(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.train()

    log.info(f"Model loaded: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    return model, tokenizer


def format_prompt(system_prompt: str, user_content: str, tokenizer) -> str:
    """Format as Qwen3 chat template."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_with_logprobs(model, tokenizer, prompt: str, max_new_tokens: int = 1024, temperature: float = 1.0):
    """Generate completion and return (text, log_probs tensor, token_ids)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract generated tokens (excluding input)
    gen_ids = outputs.sequences[0, input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Compute log probs for each generated token
    scores = outputs.scores  # tuple of (vocab_size,) tensors
    log_probs = []
    for i, score in enumerate(scores):
        if i >= len(gen_ids):
            break
        probs = torch.nn.functional.log_softmax(score[0], dim=-1)
        token_log_prob = probs[gen_ids[i]].item()
        log_probs.append(token_log_prob)

    return text, log_probs, gen_ids


def compute_logprobs_for_text(model, tokenizer, full_prompt: str, completion: str):
    """Compute log probs of completion given prompt (for policy gradient)."""
    full_text = full_prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    with torch.enable_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token log probs for the completion part only
        logits = outputs.logits[0, prompt_len - 1:-1]  # shift by 1
        target_ids = inputs["input_ids"][0, prompt_len:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_log_probs, outputs.loss


def run_episode(
    model, tokenizer, env, prompt_data: dict, system_prompt: str,
    max_new_tokens: int = 1024, temperature: float = 1.0,
) -> Rollout:
    """Run one multi-turn episode through SageTopologyEnv."""
    rollout = Rollout(
        prompt=prompt_data["prompt"],
        task_id=prompt_data["task_id"],
        difficulty=prompt_data["difficulty"],
    )

    # Reset env (queries episodic memory)
    obs = env.reset(prompt_data["prompt"], prompt_data["task_id"])
    obs_text = obs.get("text", prompt_data["prompt"])

    # Turn 0: Generate topology
    turn0_prompt = format_prompt(system_prompt, obs_text, tokenizer)
    topology_text, lp0, _ = generate_with_logprobs(
        model, tokenizer, turn0_prompt, max_new_tokens, temperature,
    )
    rollout.turns.append({
        "role": "assistant", "content": topology_text,
        "prompt": turn0_prompt, "log_probs": lp0,
    })

    # Step env with topology
    obs, reward, done, info = env.step(topology_text)
    # Note: reward here is the LAST step's reward if env auto-executes to terminal.
    # The real structural reward is in trace.steps[0].reward (topology_generator).
    # We'll extract it from the trace after the episode completes.

    # Turns 1..N: Execute + decide at checkpoints
    turn_idx = 1
    while not done and turn_idx < 10:  # safety limit
        if "[CHECKPOINT]" in obs.get("text", ""):
            # Model decides: continue or upgrade
            decision_prompt = format_prompt(system_prompt, obs["text"], tokenizer)
            decision_text, lp_d, _ = generate_with_logprobs(
                model, tokenizer, decision_prompt, 256, temperature,
            )
            rollout.turns.append({
                "role": "assistant", "content": decision_text,
                "prompt": decision_prompt, "log_probs": lp_d,
            })

            # Parse decision
            if "upgrade" in decision_text.lower():
                rollout.adaptations_triggered += 1

            obs, reward, done, info = env.step(decision_text)
        else:
            # Not a checkpoint — env auto-executes
            obs, reward, done, info = env.step("")

        turn_idx += 1

    # Collect traces (needed for GiGPO anchor-based advantages)
    trace = env._trace if hasattr(env, "_trace") else None
    if trace:
        rollout.node_traces = trace.node_traces_for_rewardflow
        rollout.outcome = trace.status
        rollout.cost_usd = sum(s.latency * 0.00001 for s in trace.steps)
        # Fix: use the TOTAL episode reward, not the last step's reward
        # The env's step() returns the last step's reward (terminal=0.0),
        # but the real structural reward is in the trace (topology_generator step)
        if trace.steps:
            rollout.reward_structural = trace.steps[0].reward  # topology_generator step
    rollout._trace = trace  # Keep full trace for GiGPO step-level advantages

    return rollout


def compute_5signal_reward(rollout: Rollout) -> float:
    """Compute the 5-signal V2 reward."""
    import math

    # R_structural (already computed by env)
    r_struct = max(0.0, min(1.0, rollout.reward_structural))

    # R_execution (from env terminal)
    exec_map = {"PASSED": 1.0, "WRONG_ANSWER": 0.5, "RUNTIME_ERROR": 0.3, "TIMEOUT": 0.2}
    r_exec = exec_map.get(rollout.outcome, 0.0)

    # R_resilience
    if rollout.adaptations_triggered > 0 and rollout.outcome == "PASSED":
        r_resil = 0.5
    elif rollout.adaptations_triggered > 0:
        r_resil = 0.3
    else:
        r_resil = 0.0

    # R_cost_efficiency
    budget = BUDGET_REF.get(rollout.difficulty, 0.05)
    r_cost = 1.0 - math.tanh(rollout.cost_usd / budget) if budget > 0 else 0.5

    # R_rewardflow is computed at batch level (see compute_rewardflow_batch)
    r_flow = rollout.reward_rewardflow  # set externally

    # Total
    total = (
        W_STRUCTURAL * r_struct
        + W_EXECUTION * r_exec
        + W_REWARDFLOW * r_flow
        + W_RESILIENCE * r_resil
        + W_COST * r_cost
    )

    rollout.reward_structural = r_struct
    rollout.reward_execution = r_exec
    rollout.reward_resilience = r_resil
    rollout.reward_cost = r_cost
    rollout.reward_total = total

    return total


def compute_rewardflow_batch(rollouts: list[Rollout]) -> None:
    """Compute RewardFlow per-node credit across K rollouts and assign back."""
    from sage.verl.rewardflow import RewardFlowPropagator

    prop = RewardFlowPropagator(damping=0.85, max_iters=20)

    # Build input for RewardFlow
    rf_input = []
    for r in rollouts:
        rf_input.append({
            "node_traces": r.node_traces,
            "terminal_reward": r.reward_execution,
        })

    per_node_rewards = prop.compute(rf_input)

    # Assign mean per-node reward as the RewardFlow signal
    for i, r in enumerate(rollouts):
        node_rewards = per_node_rewards[i] if i < len(per_node_rewards) else {}
        if node_rewards:
            r.reward_rewardflow = sum(node_rewards.values()) / len(node_rewards)
        else:
            r.reward_rewardflow = 0.0


def gigpo_step(
    model, tokenizer, optimizer,
    rollouts: list[Rollout],
    clip_eps: float = 0.2,
    max_grad_norm: float = 1.0,
) -> dict:
    """One GiGPO update step — step-level advantages grouped by anchor.

    Unlike GRPO (one advantage per episode), GiGPO computes advantages
    PER STEP, grouped by anchor state. The topology generation step is
    compared against other topology generations; the upgrade decision step
    is compared against other upgrade decisions.

    Reference: GiGPO (arXiv 2505.10978), Section 3.

    Algorithm:
    1. Collect (anchor, step_reward, turn) triples from all K rollouts
    2. Group by anchor key
    3. Within each group: advantage = (reward - group_mean) / group_std
    4. Apply per-turn advantage to the corresponding log_probs
    """
    from collections import defaultdict

    # 1. Collect all (anchor, reward, turn_data, rollout_idx) triples
    anchor_groups: dict[str, list[dict]] = defaultdict(list)

    for r_idx, rollout in enumerate(rollouts):
        trace = rollout._trace if hasattr(rollout, '_trace') else None
        steps = trace.steps if trace else []

        for t_idx, turn in enumerate(rollout.turns):
            # Match turn to step via index
            step_reward = 0.0
            anchor_key = f"turn_{t_idx}"

            if t_idx < len(steps):
                step_reward = steps[t_idx].reward
                anchor_key = steps[t_idx].anchor_key
            elif t_idx == 0:
                # Turn 0 is topology generation
                step_reward = rollout.reward_structural
                anchor_key = f"topology_generator:{rollout.difficulty}"
            else:
                # Adaptation turns get resilience credit
                step_reward = rollout.reward_resilience / max(1, len(rollout.turns) - 1)
                anchor_key = f"adaptation:{rollout.difficulty}"

            anchor_groups[anchor_key].append({
                "reward": step_reward,
                "turn": turn,
                "rollout_idx": r_idx,
                "turn_idx": t_idx,
            })

    # 2. Compute per-group advantages
    turn_advantages: list[tuple[dict, float]] = []  # (turn, advantage)

    for anchor_key, group in anchor_groups.items():
        rewards = [item["reward"] for item in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = max(1e-6, (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5)

        for item in group:
            adv = (item["reward"] - mean_r) / std_r
            turn_advantages.append((item["turn"], adv))

    # 3. Apply advantages to log_probs
    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    n_tokens = 0

    for turn, advantage in turn_advantages:
        if abs(advantage) < 1e-8:
            continue

        prompt_text = turn.get("prompt", "")
        completion = turn.get("content", "")
        if not prompt_text or not completion:
            continue

        token_log_probs, _ = compute_logprobs_for_text(
            model, tokenizer, prompt_text, completion,
        )

        # Clip advantage (DAPO-style)
        clipped_adv = max(-clip_eps, min(clip_eps, advantage))
        turn_loss = -clipped_adv * token_log_probs.sum()
        total_loss = total_loss + turn_loss
        n_tokens += len(token_log_probs)

    if n_tokens > 0:
        normalized_loss = total_loss / n_tokens
        normalized_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    # Metrics
    all_rewards = [r.reward_total for r in rollouts]
    mean_r = sum(all_rewards) / len(all_rewards)

    return {
        "loss": total_loss.item() / max(1, n_tokens),
        "reward_mean": mean_r,
        "reward_std": max(1e-6, (sum((r - mean_r) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5),
        "n_tokens": n_tokens,
        "n_turns": sum(len(r.turns) for r in rollouts),
        "n_anchor_groups": len(anchor_groups),
        "anchor_groups": {k: len(v) for k, v in anchor_groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="V2 Multi-Turn GRPO Training")
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--data", required=True, help="v2_final.jsonl path")
    parser.add_argument("--output", default="/home/yann/v2_training/grpo_v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--k", type=int, default=4, help="Rollouts per prompt")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--quantize", action="store_true", help="4-bit NF4 (for 12GB GPUs)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-db", default="", help="SQLite path for episodic memory")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 4
        args.k = 2
        args.epochs = 1
        args.max_new_tokens = 512

    # ── Load data ──
    prompts = load_prompts(args.data, args.max_samples)

    # ── Load model ──
    model, tokenizer = load_model(args.model, args.adapter, args.quantize)

    # ── Setup optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # ── Setup env ──
    from sage.verl.topology_env import SageTopologyEnv

    env_config = {}
    if args.memory_db:
        env_config["memory_db"] = args.memory_db

    # ── Setup episodic memory ──
    training_memory = None
    if args.memory_db:
        from sage.verl.training_memory import TrainingMemory
        training_memory = TrainingMemory(db_path=args.memory_db)
        log.info(f"Episodic memory: {training_memory.count()} existing episodes")

    # ── Setup system prompt ──
    # Use the first entry's system_prompt (they're all the same)
    system_prompt = prompts[0].get("system_prompt", "") if prompts else ""
    if not system_prompt:
        # Import from llm_caller where the prompt is defined
        from sage.topology.llm_caller import _TOOLCALL_SYSTEM_PROMPT
        system_prompt = _TOOLCALL_SYSTEM_PROMPT

    # ── Output dirs ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "grpo_v2_metrics.jsonl"

    # ── Training loop ──
    log.info("=" * 60)
    log.info(f"V2 Multi-Turn GRPO Training")
    log.info(f"  Prompts: {len(prompts)}")
    log.info(f"  K rollouts: {args.k}")
    log.info(f"  Epochs: {args.epochs}")
    log.info(f"  LR: {args.lr}")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Memory DB: {args.memory_db or 'disabled'}")
    log.info("=" * 60)

    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        log.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        epoch_rewards = []

        for i, prompt_data in enumerate(prompts):
            step_t0 = time.time()

            # Create fresh env per prompt (isolates state)
            env = SageTopologyEnv(config=env_config)

            # ── K rollouts ──
            rollouts = []
            for k in range(args.k):
                try:
                    rollout = run_episode(
                        model, tokenizer, env, prompt_data, system_prompt,
                        args.max_new_tokens, args.temperature,
                    )
                    rollouts.append(rollout)
                except Exception as e:
                    log.warning(f"  Rollout {k} failed: {e}")
                    # Append zero-reward rollout
                    rollouts.append(Rollout(
                        prompt=prompt_data["prompt"],
                        task_id=prompt_data["task_id"],
                        difficulty=prompt_data["difficulty"],
                    ))

            # ── RewardFlow across K rollouts ──
            try:
                compute_rewardflow_batch(rollouts)
            except Exception as e:
                log.debug(f"  RewardFlow failed: {e}")

            # ── 5-signal reward ──
            for rollout in rollouts:
                compute_5signal_reward(rollout)

            # ── GiGPO update (step-level advantages by anchor group) ──
            # Attach env trace to rollouts for anchor extraction
            for rollout in rollouts:
                if not hasattr(rollout, '_trace'):
                    rollout._trace = None
            metrics = gigpo_step(model, tokenizer, optimizer, rollouts)

            # ── Store to episodic memory ──
            if training_memory:
                best = max(rollouts, key=lambda r: r.reward_total)
                try:
                    from sage.verl.topology_env import _get_embedding
                    emb = _get_embedding(prompt_data["prompt"])
                    training_memory.store_episode(
                        task_id=prompt_data["task_id"],
                        prompt_hash=hashlib.md5(prompt_data["prompt"].encode()).hexdigest(),
                        domain="code",
                        topology_yaml=best.turns[0]["content"] if best.turns else "",
                        n_nodes=len(best.node_traces),
                        difficulty=prompt_data["difficulty"],
                        outcome=best.outcome,
                        total_reward=best.reward_total,
                        per_node_results=best.node_traces,
                        adaptations_triggered=best.adaptations_triggered,
                        embedding=emb,
                    )
                except Exception as e:
                    log.debug(f"  Memory store failed: {e}")

            # ── Logging ──
            global_step += 1
            epoch_rewards.append(metrics["reward_mean"])
            step_time = time.time() - step_t0

            # Detailed reward breakdown
            reward_breakdown = {
                "structural": sum(r.reward_structural for r in rollouts) / len(rollouts),
                "execution": sum(r.reward_execution for r in rollouts) / len(rollouts),
                "rewardflow": sum(r.reward_rewardflow for r in rollouts) / len(rollouts),
                "resilience": sum(r.reward_resilience for r in rollouts) / len(rollouts),
                "cost": sum(r.reward_cost for r in rollouts) / len(rollouts),
            }

            log_entry = {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": metrics["loss"],
                "reward_mean": metrics["reward_mean"],
                "reward_std": metrics["reward_std"],
                "n_turns": metrics["n_turns"],
                "n_tokens": metrics["n_tokens"],
                "adaptations": sum(r.adaptations_triggered for r in rollouts),
                "outcomes": [r.outcome for r in rollouts],
                "reward_breakdown": reward_breakdown,
                "step_time_s": round(step_time, 1),
                "difficulty": prompt_data["difficulty"],
                "n_anchor_groups": metrics.get("n_anchor_groups", 0),
            }

            with open(metrics_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if global_step % args.log_interval == 0:
                elapsed = time.time() - t_start
                avg_r = sum(epoch_rewards[-20:]) / min(20, len(epoch_rewards))
                log.info(
                    f"  [{global_step}] loss={metrics['loss']:.4f} "
                    f"reward={metrics['reward_mean']:.3f} "
                    f"(struct={reward_breakdown['structural']:.2f} "
                    f"exec={reward_breakdown['execution']:.2f} "
                    f"flow={reward_breakdown['rewardflow']:.2f} "
                    f"resil={reward_breakdown['resilience']:.2f} "
                    f"cost={reward_breakdown['cost']:.2f}) "
                    f"turns={metrics['n_turns']} "
                    f"time={step_time:.1f}s "
                    f"avg20={avg_r:.3f} "
                    f"elapsed={elapsed/60:.0f}min"
                )

            # ── Checkpoint ──
            if global_step % args.save_interval == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                log.info(f"  Saved checkpoint to {ckpt_dir}")

            # Free memory
            del rollouts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Final save ──
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"\nTraining complete. Final adapter saved to {output_dir}")

    # ── Summary ──
    total_time = time.time() - t_start
    all_rewards = epoch_rewards
    log.info(f"\n{'='*60}")
    log.info(f"V2 Multi-Turn GRPO Complete")
    log.info(f"  Steps: {global_step}")
    log.info(f"  Time: {total_time/3600:.1f}h")
    log.info(f"  Final avg reward: {sum(all_rewards[-20:])/min(20,len(all_rewards)):.3f}")
    if training_memory:
        log.info(f"  Episodic memory: {training_memory.count()} episodes")
        training_memory.close()
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
