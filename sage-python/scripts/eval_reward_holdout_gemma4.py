#!/usr/bin/env python3
"""N1 Evaluator (Gemma4-26B-A4B): Score topology generation on the holdout set.

Loads Gemma4-26B-A4B in BF16 (no NF4 quantization), applies LoRA adapter,
generates one topology per holdout prompt, scores with compute_score.
Returns: avg reward, max reward, P(reward > 0.3), clipped ratio.
No API cost. Requires ~26GB VRAM (BF16, MoE active params ~4B).

Usage:
    python scripts/eval_reward_holdout_gemma4.py --adapter models/gemma4_26b_a4b/sft_checkpoint
    python scripts/eval_reward_holdout_gemma4.py --adapter models/gemma4_26b_a4b/grpo_checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # Use cached models only

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("eval_n1_gemma4")

# System prompt with 7 SAGE tool definitions for tool-call format
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from sage_tool_schemas import TOOLCALL_SYSTEM_PROMPT as SYSTEM_PROMPT

HOLDOUT_PATH = "experiments/holdout_50_toolcall.json"


def load_holdout(path: str = HOLDOUT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prompts"]


def evaluate(model, tokenizer, prompts: list[dict], max_new_tokens: int = 512,
             batch_size: int = 4) -> dict:
    """Generate topologies in batches and score them.

    Batched generation uses left-padding for causal LMs.
    Batch size default is 4 (smaller than Qwen3's 8 because Gemma4 is larger).
    """
    os.environ.setdefault("SAGE_VERL_EXEC", "0")
    os.environ.setdefault("SAGE_TRAINING_PHASE", "A")
    from sage.verl.reward import compute_score

    # Left-pad for batched causal generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]

        # Tokenize batch
        texts = []
        for p in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["prompt"]},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))

        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        # Gemma4 requires mm_token_type_ids (all zeros for text-only input)
        inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each sample in batch
        for j, (p, output, plen) in enumerate(zip(batch, outputs, prompt_lengths)):
            generated = tokenizer.decode(output[plen:], skip_special_tokens=True)
            gen_len = len(output) - plen
            clipped = gen_len >= max_new_tokens

            reward = float(compute_score("sage_topology", generated, "", {}))
            idx = batch_start + j
            results.append({
                "task_id": p.get("task_id", f"holdout_{idx}"),
                "difficulty": p["difficulty"],
                "reward": reward,
                "clipped": bool(clipped),
                "gen_length": len(generated),
            })

        done = min(batch_start + batch_size, len(prompts))
        avg_so_far = sum(r["reward"] for r in results) / len(results)
        log.info("  %d/%d | avg reward=%.4f", done, len(prompts), avg_so_far)

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
    parser = argparse.ArgumentParser(description="N1 Evaluator (Gemma4): Reward on holdout")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--holdout", default=HOLDOUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default=None, help="Save metrics JSON to file")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation (default 4, smaller than Qwen3 due to model size)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info("Loading model %s (BF16) + adapter %s...", args.model, args.adapter)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    prompts = load_holdout(args.holdout)
    log.info("Evaluating %d holdout prompts...", len(prompts))

    t0 = time.time()
    metrics = evaluate(model, tokenizer, prompts, args.max_new_tokens, args.batch_size)
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
