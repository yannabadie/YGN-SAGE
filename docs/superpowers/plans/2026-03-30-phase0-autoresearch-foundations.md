# Phase 0: Autoresearch Foundations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the autoresearch infrastructure: holdout set, evaluators, config-driven training, experiment journal, and orchestrator loop.

**Architecture:** A fixed-budget experiment loop (inspired by karpathy/autoresearch) that reads a journal of past experiments, runs training with a JSON config, evaluates on 3 levels (reward/MASBENCH/BigCodeBench), and records results. Every experiment is reproducible from its config file.

**Tech Stack:** Python 3.12, TRL 0.29.1, PEFT 0.18.1, bitsandbytes, sage.verl.reward, sage.bench (MASBENCH + BigCodeBench)

---

### File Structure

```
sage-python/
  experiments/
    holdout_50.json              # 50 stratified prompts (never trained on)
    journal.jsonl                # Structured experiment log
    configs/
      baseline_sft_grpo.json     # First baseline config
  scripts/
    train_local_qwen3_4b.py      # MODIFY: accept --config JSON
    eval_reward_holdout.py       # NEW: N1 evaluator
    eval_masbench_local.py       # NEW: N2 evaluator wrapper
    eval_bigcodebench_local.py   # NEW: N3 evaluator wrapper
    autoresearch_loop.py         # NEW: orchestrator
```

---

### Task 1: Create Stratified Holdout Set

**Files:**
- Create: `sage-python/experiments/holdout_50.json`
- Create: `sage-python/scripts/create_holdout.py`

- [ ] **Step 1: Write the holdout creation script**

```python
#!/usr/bin/env python3
"""Create a stratified holdout set of 50 prompts from SFT data.

Split: 15 simple, 20 moderate, 15 complex.
These prompts are NEVER used in training — evaluation only.
"""
import json
import random
import sys

random.seed(42)  # Reproducible

TARGET = {"simple": 15, "moderate": 20, "complex": 15}
OUTPUT = "experiments/holdout_50.json"
SFT_DATA = "data/topology_sft_v2_combined.jsonl"


def main():
    by_difficulty: dict[str, list] = {"simple": [], "moderate": [], "complex": []}

    with open(SFT_DATA, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            diff = entry.get("difficulty", "simple")
            if diff not in by_difficulty:
                diff = "moderate"  # fallback
            by_difficulty[diff].append({
                "task_id": entry.get("task_id", ""),
                "prompt": entry["prompt"],
                "difficulty": diff,
                "reference_yaml": entry.get("topology_yaml", ""),
            })

    holdout = []
    for diff, count in TARGET.items():
        pool = by_difficulty[diff]
        if len(pool) < count:
            print(f"WARNING: only {len(pool)} {diff} prompts, need {count}", file=sys.stderr)
            count = len(pool)
        holdout.extend(random.sample(pool, count))

    random.shuffle(holdout)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "count": len(holdout), "prompts": holdout}, f, indent=2)

    print(f"Created {OUTPUT}: {len(holdout)} prompts")
    for diff in TARGET:
        n = sum(1 for h in holdout if h["difficulty"] == diff)
        print(f"  {diff}: {n}")


if __name__ == "__main__":
    main()
```

Save to `sage-python/scripts/create_holdout.py`.

- [ ] **Step 2: Run the script to create the holdout set**

Run: `cd sage-python && python scripts/create_holdout.py`

Expected output:
```
Created experiments/holdout_50.json: 50 prompts
  simple: 15
  moderate: 20
  complex: 15
```

- [ ] **Step 3: Verify the holdout set**

Run: `cd sage-python && python -c "import json; d=json.load(open('experiments/holdout_50.json')); print(d['count'], 'prompts'); print(set(p['difficulty'] for p in d['prompts']))"`

Expected: `50 prompts` and `{'simple', 'moderate', 'complex'}`.

- [ ] **Step 4: Commit**

```bash
git add sage-python/scripts/create_holdout.py sage-python/experiments/holdout_50.json
git commit -m "feat: create stratified holdout set (50 prompts, seed=42)"
```

---

### Task 2: N1 Evaluator — Reward on Holdout

**Files:**
- Create: `sage-python/scripts/eval_reward_holdout.py`
- Read: `sage-python/src/sage/verl/reward.py` (compute_score signature)

- [ ] **Step 1: Write the N1 evaluator**

```python
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

    # Set chat template (same as training)
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
```

Save to `sage-python/scripts/eval_reward_holdout.py`.

- [ ] **Step 2: Run N1 on the current SFT checkpoint to establish baseline**

Run: `cd sage-python && nvidia-smi -lgc 3105 && python scripts/eval_reward_holdout.py --adapter models/local_qwen3_4b_grpo/sft_checkpoint --output experiments/n1_baseline.json`

Expected: ~2-5 min, outputs avg/max reward, P(r>0.3), per-difficulty breakdown.

- [ ] **Step 3: Commit**

```bash
git add sage-python/scripts/eval_reward_holdout.py
git commit -m "feat: N1 evaluator — reward on 50-prompt holdout set"
```

---

### Task 3: N2/N3 Evaluator Wrappers

**Files:**
- Create: `sage-python/scripts/eval_masbench_local.py`
- Create: `sage-python/scripts/eval_bigcodebench_local.py`
- Read: `sage-python/src/sage/bench/masbench.py`, `sage-python/src/sage/bench/bigcodebench_bench.py`

- [ ] **Step 1: Write the N2 MASBENCH wrapper**

```python
#!/usr/bin/env python3
"""N2 Evaluator: MASBENCH depth with Path 6 local model.

Sets SAGE_ENABLE_PATH6=1 and runs MASBENCH depth benchmark.
Cost: ~$0.50 per run. Duration: ~10 min.

Usage:
    python scripts/eval_masbench_local.py --adapter models/local_qwen3_4b_grpo/sft_checkpoint --limit 20
"""
import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("eval_n2")


def main():
    parser = argparse.ArgumentParser(description="N2: MASBENCH depth evaluation")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    env["SAGE_ENABLE_PATH6"] = "1"
    env["SAGE_PATH6_ADAPTER"] = os.path.abspath(args.adapter)

    log.info("Running MASBENCH depth (limit=%d) with Path 6 from %s", args.limit, args.adapter)

    result = subprocess.run(
        [sys.executable, "-m", "sage.bench",
         "--type", "masbench", "--axis", "depth", "--limit", str(args.limit),
         "--output-json", args.output or "experiments/n2_latest.json"],
        env=env, capture_output=True, text=True, timeout=1200,
    )

    print(result.stdout)
    if result.returncode != 0:
        log.error("MASBENCH failed: %s", result.stderr[-500:] if result.stderr else "no stderr")
        return None

    if args.output and os.path.exists(args.output):
        with open(args.output) as f:
            metrics = json.load(f)
        log.info("N2 MASBENCH depth: %s", json.dumps(metrics, indent=2)[:500])
        return metrics

    return None


if __name__ == "__main__":
    main()
```

Save to `sage-python/scripts/eval_masbench_local.py`.

- [ ] **Step 2: Write the N3 BigCodeBench wrapper**

```python
#!/usr/bin/env python3
"""N3 Evaluator: BigCodeBench Hard Instruct with Path 6 local model.

Sets SAGE_ENABLE_PATH6=1 and runs BigCodeBench Hard benchmark.
Cost: ~$2.00 per run. Duration: ~30 min.

Usage:
    python scripts/eval_bigcodebench_local.py --adapter models/local_qwen3_4b_grpo/sft_checkpoint --limit 20
"""
import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("eval_n3")


def main():
    parser = argparse.ArgumentParser(description="N3: BigCodeBench Hard evaluation")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env = os.environ.copy()
    env["SAGE_ENABLE_PATH6"] = "1"
    env["SAGE_PATH6_ADAPTER"] = os.path.abspath(args.adapter)

    log.info("Running BigCodeBench Hard Instruct (limit=%d) with Path 6 from %s",
             args.limit, args.adapter)

    result = subprocess.run(
        [sys.executable, "-m", "sage.bench",
         "--type", "bigcodebench", "--subset", "hard", "--split", "instruct",
         "--limit", str(args.limit),
         "--output-json", args.output or "experiments/n3_latest.json"],
        env=env, capture_output=True, text=True, timeout=3600,
    )

    print(result.stdout)
    if result.returncode != 0:
        log.error("BigCodeBench failed: %s", result.stderr[-500:] if result.stderr else "no stderr")
        return None

    if args.output and os.path.exists(args.output):
        with open(args.output) as f:
            metrics = json.load(f)
        log.info("N3 BigCodeBench Hard: %s", json.dumps(metrics, indent=2)[:500])
        return metrics

    return None


if __name__ == "__main__":
    main()
```

Save to `sage-python/scripts/eval_bigcodebench_local.py`.

- [ ] **Step 3: Commit**

```bash
git add sage-python/scripts/eval_masbench_local.py sage-python/scripts/eval_bigcodebench_local.py
git commit -m "feat: N2 (MASBENCH) and N3 (BigCodeBench) evaluator wrappers"
```

---

### Task 4: Config-Driven Training

**Files:**
- Modify: `sage-python/scripts/train_local_qwen3_4b.py`
- Create: `sage-python/experiments/configs/baseline_sft_grpo.json`

- [ ] **Step 1: Add `--config` support to training script**

Add this function at the top of `train_local_qwen3_4b.py` (after the imports, before `SYSTEM_PROMPT`):

```python
def load_config(path: str) -> dict:
    """Load experiment config JSON. CLI args override config values."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

Then in `main()`, after `args = parser.parse_args()`, add:

```python
    # ── Load config if provided ────────────────────────────────
    if hasattr(args, 'config') and args.config:
        config = load_config(args.config)
        for key, value in config.items():
            key_attr = key.replace("-", "_")
            if not getattr(args, key_attr, None):  # CLI args take precedence
                setattr(args, key_attr, value)
        log.info("Loaded config from %s", args.config)
```

Add the CLI arg in the parser:

```python
    parser.add_argument("--config", default=None, help="JSON config file (CLI args override)")
```

- [ ] **Step 2: Create the baseline config**

```json
{
  "model": "Qwen/Qwen3-4B",
  "sft_data": "data/topology_sft_v2_combined.jsonl",
  "sft_epochs": 2,
  "sft_lr": 2e-5,
  "sft_max_samples": 0,
  "data": "data/verl_topology_train.parquet",
  "output": "models/local_qwen3_4b_grpo",
  "epochs": 1,
  "lr": 5e-6,
  "batch_size": 1,
  "lora_rank": 32,
  "num_generations": 2,
  "max_completion_length": 512,
  "grad_accum": 2,
  "max_samples": 40
}
```

Save to `sage-python/experiments/configs/baseline_sft_grpo.json`.

- [ ] **Step 3: Verify config loading works**

Run: `cd sage-python && python scripts/train_local_qwen3_4b.py --config experiments/configs/baseline_sft_grpo.json --sft-only --sft-max-samples 4 2>&1 | head -5`

Expected: Should show `Loaded config from experiments/configs/baseline_sft_grpo.json` and start SFT.
Kill after confirming it loads correctly.

- [ ] **Step 4: Commit**

```bash
git add sage-python/scripts/train_local_qwen3_4b.py sage-python/experiments/configs/baseline_sft_grpo.json
git commit -m "feat: config-driven training via --config JSON"
```

---

### Task 5: Experiment Journal

**Files:**
- Create: `sage-python/experiments/journal.jsonl`

- [ ] **Step 1: Write the baseline journal entry**

Using the results from the March 29-30 training session:

```json
{"id": "exp-001", "timestamp": "2026-03-30T00:30:00Z", "phase": "0-baseline", "hypothesis": "SFT warmup resolves GRPO cold-start (P(valid YAML | pi_base) ~= 0)", "config": "configs/baseline_sft_grpo.json", "base_checkpoint": null, "train_budget_min": 90, "train_steps_sft": 470, "train_steps_grpo": 20, "metrics": {"sft_loss_start": 2.736, "sft_loss_final": 0.976, "n1_reward_avg": 0.40, "n1_reward_max": 0.93, "n1_above_03": 0.45, "n1_clipped_ratio": 0.35, "n2_masbench_depth": null, "n3_bigcodebench_hard": null}, "conclusion": "SFT warmup validated: reward 0.134->0.400 (+198%), clipped 1.0->0.35. GRPO has gradient signal post-SFT.", "duration_min": 90}
```

Save to `sage-python/experiments/journal.jsonl` (one line, no trailing newline).

- [ ] **Step 2: Commit**

```bash
git add sage-python/experiments/journal.jsonl
git commit -m "feat: experiment journal with baseline entry (exp-001)"
```

---

### Task 6: Autoresearch Loop Orchestrator

**Files:**
- Create: `sage-python/scripts/autoresearch_loop.py`

- [ ] **Step 1: Write the orchestrator**

```python
#!/usr/bin/env python3
"""Autoresearch Loop: Autonomous experiment iteration.

Reads the experiment journal, runs training with a config,
evaluates on 3 levels, records results.

Inspired by karpathy/autoresearch: fixed-budget experiments,
structured journal, reproducible configs.

Usage:
    # Run one experiment with a config
    python scripts/autoresearch_loop.py --config experiments/configs/my_config.json --budget 10

    # Run one experiment, hypothesis provided
    python scripts/autoresearch_loop.py --config experiments/configs/my_config.json \
        --hypothesis "Increasing LoRA rank to 64 improves YAML structure" --budget 10

    # Evaluate only (no training, just N1/N2/N3 on existing adapter)
    python scripts/autoresearch_loop.py --eval-only --adapter models/local_qwen3_4b_grpo/sft_checkpoint
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("autoresearch")

JOURNAL_PATH = "experiments/journal.jsonl"
CONFIGS_DIR = "experiments/configs"


def read_journal() -> list[dict]:
    """Read all past experiments."""
    entries = []
    if os.path.exists(JOURNAL_PATH):
        with open(JOURNAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def next_experiment_id(journal: list[dict]) -> str:
    """Generate next experiment ID."""
    max_n = 0
    for entry in journal:
        eid = entry.get("id", "exp-000")
        try:
            n = int(eid.split("-")[1])
            max_n = max(max_n, n)
        except (IndexError, ValueError):
            pass
    return f"exp-{max_n + 1:03d}"


def get_best_n1(journal: list[dict]) -> float:
    """Get best N1 reward avg from journal."""
    best = 0.0
    for entry in journal:
        m = entry.get("metrics", {})
        val = m.get("n1_reward_avg", 0)
        if val > best:
            best = val
    return best


def run_training(config_path: str, budget_min: int, sft_only: bool = False) -> str | None:
    """Run training and return adapter path."""
    cmd = [sys.executable, "scripts/train_local_qwen3_4b.py",
           "--config", config_path]
    if sft_only:
        cmd.append("--sft-only")

    log.info("Training: %s (budget %d min)", " ".join(cmd), budget_min)
    timeout = budget_min * 60 + 120  # buffer

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        log.error("Training failed: %s", result.stderr[-500:] if result.stderr else "")
        return None

    # Find the latest checkpoint
    config = json.load(open(config_path))
    output_dir = config.get("output", "models/local_qwen3_4b_grpo")
    for sub in ["grpo_checkpoint", "sft_checkpoint"]:
        path = os.path.join(output_dir, sub)
        if os.path.exists(os.path.join(path, "adapter_config.json")):
            return path

    return output_dir


def run_n1(adapter_path: str) -> dict | None:
    """Run N1 evaluation (reward holdout)."""
    output = f"experiments/n1_{os.path.basename(adapter_path)}.json"
    cmd = [sys.executable, "scripts/eval_reward_holdout.py",
           "--adapter", adapter_path, "--output", output]

    log.info("N1 eval: %s", adapter_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        log.error("N1 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def run_n2(adapter_path: str, limit: int = 20) -> dict | None:
    """Run N2 evaluation (MASBENCH depth)."""
    output = "experiments/n2_latest.json"
    cmd = [sys.executable, "scripts/eval_masbench_local.py",
           "--adapter", adapter_path, "--limit", str(limit), "--output", output]

    log.info("N2 eval: MASBENCH depth (limit=%d)", limit)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        log.error("N2 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def run_n3(adapter_path: str, limit: int = 20) -> dict | None:
    """Run N3 evaluation (BigCodeBench Hard)."""
    output = "experiments/n3_latest.json"
    cmd = [sys.executable, "scripts/eval_bigcodebench_local.py",
           "--adapter", adapter_path, "--limit", str(limit), "--output", output]

    log.info("N3 eval: BigCodeBench Hard (limit=%d)", limit)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        log.error("N3 failed: %s", result.stderr[-300:] if result.stderr else "")
        return None

    if os.path.exists(output):
        with open(output) as f:
            return json.load(f)
    return None


def record_experiment(entry: dict):
    """Append experiment to journal."""
    with open(JOURNAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("Recorded %s -> %s", entry["id"], JOURNAL_PATH)


def run_experiment(args):
    """Execute a single experiment: train -> evaluate -> record."""
    journal = read_journal()
    exp_id = next_experiment_id(journal)
    best_n1 = get_best_n1(journal)
    t0 = time.time()

    log.info("=== Experiment %s ===", exp_id)
    log.info("Hypothesis: %s", args.hypothesis)
    log.info("Config: %s", args.config)
    log.info("Best N1 so far: %.4f", best_n1)

    entry = {
        "id": exp_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": args.phase,
        "hypothesis": args.hypothesis,
        "config": args.config,
        "base_checkpoint": args.adapter,
        "train_budget_min": args.budget,
        "metrics": {},
    }

    # ── Train ──────────────────────────────────────────────────
    adapter_path = args.adapter
    if not args.eval_only:
        adapter_path = run_training(args.config, args.budget, sft_only=args.sft_only)
        if not adapter_path:
            entry["conclusion"] = "FAILED: Training crashed."
            record_experiment(entry)
            return

    # ── N1: Reward Holdout ─────────────────────────────────────
    n1 = run_n1(adapter_path)
    if n1:
        entry["metrics"]["n1_reward_avg"] = n1["n1_reward_avg"]
        entry["metrics"]["n1_reward_max"] = n1["n1_reward_max"]
        entry["metrics"]["n1_above_03"] = n1["n1_above_03"]
        entry["metrics"]["n1_clipped_ratio"] = n1["n1_clipped_ratio"]
        log.info("N1: avg=%.4f max=%.4f P(>0.3)=%.0f%%",
                 n1["n1_reward_avg"], n1["n1_reward_max"], n1["n1_above_03"] * 100)

    # ── N2: MASBENCH (only if N1 improved) ─────────────────────
    if n1 and n1["n1_reward_avg"] > best_n1:
        log.info("N1 improved (%.4f > %.4f) -> running N2", n1["n1_reward_avg"], best_n1)
        n2 = run_n2(adapter_path)
        if n2:
            entry["metrics"]["n2_masbench_depth"] = n2
    else:
        log.info("N1 did not improve (%.4f <= %.4f) -> skipping N2/N3",
                 n1["n1_reward_avg"] if n1 else 0, best_n1)

    # ── N3: BigCodeBench (only if N2 improved) ─────────────────
    # TODO: compare N2 vs best N2 in journal — for now, run if N1 improved significantly
    if n1 and n1["n1_reward_avg"] > best_n1 * 1.1:  # 10% improvement threshold
        log.info("Significant N1 improvement -> running N3")
        n3 = run_n3(adapter_path)
        if n3:
            entry["metrics"]["n3_bigcodebench_hard"] = n3

    elapsed = (time.time() - t0) / 60
    entry["duration_min"] = round(elapsed, 1)

    if not args.eval_only:
        entry["conclusion"] = args.hypothesis  # User fills in after reviewing
    else:
        entry["conclusion"] = f"Eval-only on {adapter_path}"

    record_experiment(entry)

    log.info("=== Experiment %s complete (%.1f min) ===", exp_id, elapsed)
    log.info("Journal: %d entries total", len(journal) + 1)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch Loop")
    parser.add_argument("--config", default=None, help="Training config JSON")
    parser.add_argument("--hypothesis", default="Baseline evaluation",
                        help="What are we testing?")
    parser.add_argument("--phase", default="0-baseline", help="Roadmap phase")
    parser.add_argument("--budget", type=int, default=60, help="Training budget in minutes")
    parser.add_argument("--adapter", default=None, help="Existing adapter to evaluate")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--sft-only", action="store_true", help="SFT phase only")
    args = parser.parse_args()

    if not args.eval_only and not args.config:
        parser.error("--config required unless --eval-only")

    os.makedirs("experiments/configs", exist_ok=True)
    run_experiment(args)


if __name__ == "__main__":
    main()
```

Save to `sage-python/scripts/autoresearch_loop.py`.

- [ ] **Step 2: Commit**

```bash
git add sage-python/scripts/autoresearch_loop.py
git commit -m "feat: autoresearch loop orchestrator (train->eval->record)"
```

---

### Task 7: Integration Test

**Files:** None new — tests the full pipeline.

- [ ] **Step 1: Run the autoresearch loop in eval-only mode on existing SFT checkpoint**

Run:
```bash
cd sage-python
nvidia-smi -lgc 3105
python scripts/autoresearch_loop.py \
  --eval-only \
  --adapter models/local_qwen3_4b_grpo/sft_checkpoint \
  --hypothesis "Baseline N1 eval on SFT checkpoint" \
  --phase "0-baseline"
```

Expected: Journal entry appended with N1 metrics. ~2-5 min.

- [ ] **Step 2: Verify journal has new entry**

Run: `cd sage-python && tail -1 experiments/journal.jsonl | python -m json.tool`

Expected: JSON with `id: exp-002`, `n1_reward_avg`, `n1_reward_max`, `n1_above_03`.

- [ ] **Step 3: Run the autoresearch loop with a training config (quick smoke)**

Run:
```bash
cd sage-python
python scripts/autoresearch_loop.py \
  --config experiments/configs/baseline_sft_grpo.json \
  --hypothesis "Smoke test: full train+eval loop" \
  --phase "0-smoke" \
  --budget 10 \
  --sft-only
```

Expected: SFT trains for a few steps (budget 10 min), N1 eval runs, journal entry written.

- [ ] **Step 4: Commit all results and push**

```bash
git add sage-python/experiments/
git commit -m "feat: Phase 0 complete — autoresearch loop validated"
git push origin local
```

---

### Summary: Phase 0 Exit Criteria

| Criteria | How to verify |
|----------|---------------|
| Holdout set exists | `cat experiments/holdout_50.json \| python -c "import json,sys;print(json.load(sys.stdin)['count'])"` -> 50 |
| N1 evaluator works | `python scripts/eval_reward_holdout.py --adapter ... --output /tmp/test.json` succeeds |
| Config-driven training | `python scripts/train_local_qwen3_4b.py --config experiments/configs/baseline_sft_grpo.json --smoke` works |
| Journal has baseline | `wc -l experiments/journal.jsonl` >= 1 |
| Autoresearch loop runs | `python scripts/autoresearch_loop.py --eval-only --adapter ...` produces journal entry |
