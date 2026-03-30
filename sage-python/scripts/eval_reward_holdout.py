#!/usr/bin/env python3
"""N1 Evaluator: Score a model's topology generation on the holdout set.

Loads model, generates one topology per holdout prompt, scores with compute_score.
Returns: avg reward, max reward, P(reward > 0.3), clipped ratio.
No API cost. ~2 min on RTX 3500 Ada.

Usage:
    python scripts/eval_reward_holdout.py --adapter models/local_qwen3_4b_grpo/sft_checkpoint
    python scripts/eval_reward_holdout.py --adapter models/local_qwen3_4b_grpo/grpo_checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("eval_n1")

SYSTEM_PROMPT = (
    "You are a multi-agent topology designer for the YGN-SAGE framework. "
    "Given a coding task, design an optimal agent topology as a YAML DAG. "
    "Include: difficulty, reasoning, nodes (role + prompt + model_tier), "
    "edges (from_idx + to_idx + flow_type). The LAST node must be a "
    "synthesizer that returns the final answer."
)

HOLDOUT_PATH = "experiments/holdout_50.json"


def load_holdout(path: str = HOLDOUT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"]


def evaluate(model, tokenizer, prompts: list[dict], max_new_tokens: int = 512) -> dict:
    """Generate topologies and score them."""
    os.environ.setdefault("SAGE_VERL_EXEC", "0")
    os.environ.setdefault("SAGE_TRAINING_PHASE", "A")
    from sage.verl.reward import compute_score

    results = []
    for i, p in enumerate(prompts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt"]},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        clipped = len(output[0]) - inputs["input_ids"].shape[1] >= max_new_tokens

        reward = float(compute_score("sage_topology", generated, "", {}))
        results.append({
            "task_id": p.get("task_id", f"holdout_{i}"),
            "difficulty": p["difficulty"],
            "reward": reward,
            "clipped": clipped,
            "gen_length": len(generated),
        })

        if (i + 1) % 10 == 0:
            avg_so_far = sum(r["reward"] for r in results) / len(results)
            log.info("  %d/%d | avg reward=%.4f", i + 1, len(prompts), avg_so_far)

    rewards = [r["reward"] for r in results]
    clipped_count = sum(1 for r in results if r["clipped"])

    metrics = {
        "n1_reward_avg": sum(rewards) / len(rewards),
        "n1_reward_max": max(rewards),
        "n1_reward_min": min(rewards),
        "n1_above_03": sum(1 for r in rewards if r > 0.3) / len(rewards),
        "n1_clipped_ratio": clipped_count / len(results),
        "n1_count": len(results),
        "per_difficulty": {},
    }
    for diff in ["simple", "moderate", "complex"]:
        diff_rewards = [r["reward"] for r in results if r["difficulty"] == diff]
        if diff_rewards:
            metrics["per_difficulty"][diff] = {
                "avg": sum(diff_rewards) / len(diff_rewards),
                "count": len(diff_rewards),
            }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="N1 Evaluator: Reward on holdout")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--holdout", default=HOLDOUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default=None, help="Save metrics JSON to file")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    log.info("Loading model %s + adapter %s...", args.model, args.adapter)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto",
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

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

    prompts = load_holdout(args.holdout)
    log.info("Evaluating %d holdout prompts...", len(prompts))

    t0 = time.time()
    metrics = evaluate(model, tokenizer, prompts, args.max_new_tokens)
    elapsed = time.time() - t0

    metrics["eval_time_sec"] = elapsed
    metrics["adapter"] = args.adapter

    log.info("=== N1 Results (%.1fs) ===", elapsed)
    log.info("  avg reward: %.4f", metrics["n1_reward_avg"])
    log.info("  max reward: %.4f", metrics["n1_reward_max"])
    log.info("  P(r>0.3):   %.1f%%", metrics["n1_above_03"] * 100)
    log.info("  clipped:    %.1f%%", metrics["n1_clipped_ratio"] * 100)
    for diff, vals in metrics["per_difficulty"].items():
        log.info("  %s: avg=%.4f (n=%d)", diff, vals["avg"], vals["count"])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Saved to %s", args.output)

    return metrics


if __name__ == "__main__":
    main()
