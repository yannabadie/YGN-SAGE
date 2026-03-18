# GRPO Two-Phase Topology Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Phi-4-mini to generate optimal multi-agent topologies in 2 phases: Phase 1 learns YAML structure (fast, no API), Phase 2 learns execution quality (TopologyRunner + Gemini Flash).

**Architecture:** Phase 1 uses format+structure rewards only (~25 min, no LLM calls) to establish 80%+ YAML validity. Phase 2 loads Phase 1 checkpoint and adds execution reward via real TopologyRunner (Gemini Flash, ~3h). This prevents the YAML validity collapse observed at step 10-15 when all 3 rewards compete from step 0. AgentConductor validates this: they train structure only during GRPO, model assignment at inference.

**Tech Stack:** TRL 0.29 GRPOTrainer, Phi-4-mini-instruct (3.8B), sage-core Rust (TopologyReward, TopologyDensity, HybridVerifier, TopologyExecutor), TopologyRunner, Gemini 2.5 Flash, BigCodeBench test cases.

---

## Research Backing

| Insight | Source | Impact |
|---------|--------|--------|
| Structure-only GRPO, model assignment at inference | AgentConductor (2602.17100) | Validates Phase 1/2 split |
| 3B models converge in 20-60 GRPO steps | Verl Engineering Handbook (Qwen2.5-3B) | Phase 1: 50 steps sufficient |
| YAML validity collapse when execution gradient competes with format | Own logs (v5 steps 10-15: 0% valid) | Phase 1 establishes stability first |
| No paper jointly trains structure + model assignment | Literature survey (March 2026) | Phase 2 multi-model is novel |
| normalize_then_sum prevents reward scale mismatch | GDPO paper, AgentConductor | Keep in both phases |

## Files Map

| Action | File | Phase | Purpose |
|--------|------|-------|---------|
| Modify | `sage-python/scripts/train_topology_grpo.py` | 1+2 | Add `grpo-phase1` and `grpo-phase2` modes |
| Keep | `sage-python/src/sage/grpo/execution_reward.py` | 2 | Already correct (v5: TopologyRunner + Gemini) |
| Keep | `sage-python/tests/test_grpo_execution_reward.py` | - | 17 tests, all passing |

## Current State

- `execution_reward.py` (v5): TopologyRunner + Gemini Flash + BigCodeBench tests + edges + system from difficulty. **Done, 17 tests passing.**
- `train_topology_grpo.py`: has `grpo-v3` mode that uses all 3 rewards from step 0. **Needs splitting into Phase 1 and Phase 2.**
- format_reward: range [-2.0, +1.0]. **Done (v4 fix).**
- LR: 1e-6. **Done (v4 fix).**
- normalize_then_sum with [1.0, 1.0, 1.0]. **Done.**

---

### Task 1: Add grpo-phase1 mode (format + structure only, no API)

**Files:**
- Modify: `sage-python/scripts/train_topology_grpo.py`

Phase 1 trains with format_reward + structure_reward ONLY. No execution_reward = no API calls = ~15s/step. The model learns valid YAML structure in ~25 min.

- [ ] **Step 1: Add run_grpo_phase1 function**

Add after `run_grpo_v2` function (around line 512) in `train_topology_grpo.py`:

```python
def run_grpo_phase1(sft_checkpoint: str, output_dir: str):
    """Phase 1: Learn YAML structure (format + structure rewards only).

    No execution reward = no API calls = fast (~15s/step).
    Goal: 80%+ valid YAML before adding execution signal.
    """
    log.info("GRPO Phase 1 — structure learning (no API calls)")

    model, tokenizer, peft_config = _load_model_and_tokenizer(sft_checkpoint)

    # Reuse format_reward and structure_reward from run_grpo_v2
    # (they are defined inside run_grpo_v2 — extract them or redefine)
    def format_reward(completions, **kwargs):
        import yaml
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                data = yaml.safe_load(text)
                if not isinstance(data, dict):
                    rewards.append(-1.5)
                    continue
                if "nodes" not in data:
                    rewards.append(-0.5)
                    continue
                nodes = data["nodes"]
                if not isinstance(nodes, list) or len(nodes) == 0:
                    rewards.append(-0.25)
                    continue
                rewards.append(1.0)
            except yaml.YAMLError:
                rewards.append(-2.0)
            except Exception:
                rewards.append(-2.0)
        return rewards

    def structure_reward(completions, **kwargs):
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

    # Load prompts (code tasks only, no GSM8K)
    prompts = []
    sft_data = None
    for candidate in [Path("data/topology_sft_clean.jsonl"), Path("data/topology_sft_combined.jsonl")]:
        if candidate.exists():
            sft_data = candidate
            break
    if sft_data:
        with open(sft_data, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("task_id", "").startswith("GSM8K"):
                    continue
                p = entry.get("prompt", "")
                if p:
                    prompts.append(
                        f"<|system|>You are a multi-agent topology designer. "
                        f"Given a task, generate an optimal agent topology in YAML format.<|end|>\n"
                        f"<|user|>{p}<|end|>\n"
                        f"<|assistant|>"
                    )
        prompts = prompts[:200]
        log.info("Phase 1: %d prompts loaded", len(prompts))
    else:
        log.error("No SFT data found")
        sys.exit(1)

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})

    config = GRPOConfig(
        output_dir=output_dir,
        # Phase 1: structure only, no execution reward
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=512,
        temperature=0.4,
        mask_truncated_completions=False,
        loss_type="grpo",
        beta=0.04,
        multi_objective_aggregation="normalize_then_sum",
        reward_weights=[1.0, 1.0],  # format + structure (2 rewards, no execution)
        # Training — fast convergence
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        warmup_steps=20,
        seed=42,
        # Memory
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # Logging
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, structure_reward],  # NO execution reward
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info("Phase 1: starting (format + structure only, ~15s/step)...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Phase 1 complete. Checkpoint saved to %s", output_dir)
```

- [ ] **Step 2: Add run_grpo_phase2 function**

Add after `run_grpo_phase1`:

```python
def run_grpo_phase2(phase1_checkpoint: str, output_dir: str):
    """Phase 2: Learn execution quality (TopologyRunner + Gemini Flash).

    Loads Phase 1 checkpoint (stable YAML structure) and adds execution reward.
    Goal: model learns which topologies SOLVE problems better.
    """
    log.info("GRPO Phase 2 — execution learning (TopologyRunner + Gemini Flash)")

    model, tokenizer, peft_config = _load_model_and_tokenizer(phase1_checkpoint)

    # Same format + structure rewards as Phase 1
    def format_reward(completions, **kwargs):
        import yaml
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            try:
                data = yaml.safe_load(text)
                if not isinstance(data, dict):
                    rewards.append(-1.5)
                    continue
                if "nodes" not in data:
                    rewards.append(-0.5)
                    continue
                nodes = data["nodes"]
                if not isinstance(nodes, list) or len(nodes) == 0:
                    rewards.append(-0.25)
                    continue
                rewards.append(1.0)
            except yaml.YAMLError:
                rewards.append(-2.0)
            except Exception:
                rewards.append(-2.0)
        return rewards

    def structure_reward(completions, **kwargs):
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

    # Execution reward (v5: TopologyRunner + Gemini Flash + BigCodeBench tests)
    def execution_reward(completions, **kwargs):
        from sage.grpo.execution_reward import execution_reward_batch
        return execution_reward_batch(completions, **kwargs)

    # Load prompts with task_ids (needed for BigCodeBench test matching)
    prompts = []
    task_ids = []
    sft_data = None
    for candidate in [Path("data/topology_sft_clean.jsonl"), Path("data/topology_sft_combined.jsonl")]:
        if candidate.exists():
            sft_data = candidate
            break
    if sft_data:
        with open(sft_data, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                tid = entry.get("task_id", "")
                if tid.startswith("GSM8K"):
                    continue
                p = entry.get("prompt", "")
                if p:
                    prompts.append(
                        f"<|system|>You are a multi-agent topology designer. "
                        f"Given a task, generate an optimal agent topology in YAML format.<|end|>\n"
                        f"<|user|>{p}<|end|>\n"
                        f"<|assistant|>"
                    )
                    task_ids.append(tid)
        prompts = prompts[:200]
        task_ids = task_ids[:200]
        log.info("Phase 2: %d prompts loaded", len(prompts))
    else:
        log.error("No SFT data found")
        sys.exit(1)

    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts, "task_id": task_ids})

    config = GRPOConfig(
        output_dir=output_dir,
        # Phase 2: all 3 rewards (format + structure + execution)
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=512,
        temperature=0.4,
        mask_truncated_completions=False,
        loss_type="grpo",
        beta=0.04,
        multi_objective_aggregation="normalize_then_sum",
        reward_weights=[1.0, 1.0, 1.0],  # 3 rewards: format, structure, execution
        # Training — slower due to API calls
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        warmup_steps=20,
        seed=42,
        # Memory
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # Logging + checkpoints
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, structure_reward, execution_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    log.info("Phase 2: starting (format + structure + execution via TopologyRunner)...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Phase 2 complete. Final model saved to %s", output_dir)
```

- [ ] **Step 3: Add CLI modes for phase1 and phase2**

In the `main()` function, update the argparse choices and dispatch:

```python
parser.add_argument("--mode", choices=[
    "sft", "grpo", "grpo-v2", "grpo-v3",
    "grpo-phase1", "grpo-phase2",
    "export",
], required=True)
```

And add dispatch cases:

```python
    elif args.mode == "grpo-phase1":
        run_grpo_phase1(args.sft_checkpoint, "models/topology_grpo_phase1/")
    elif args.mode == "grpo-phase2":
        run_grpo_phase2("models/topology_grpo_phase1/", "models/topology_grpo_phase2/")
```

- [ ] **Step 4: Extract _load_model_and_tokenizer helper**

Both phase functions need to load model + tokenizer. Extract a shared helper at module level (before `run_grpo_v2`):

```python
def _load_model_and_tokenizer(checkpoint: str):
    """Load SFT/Phase1 checkpoint with LoRA config for GRPO training."""
    log.info("Loading checkpoint: %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    return checkpoint, tokenizer, peft_config
```

Note: GRPOTrainer takes the model path as string (not loaded model) when using peft_config.

- [ ] **Step 5: Verify syntax**

Run: `cd /c/Code/YGN-SAGE && python -c "import ast; ast.parse(open('sage-python/scripts/train_topology_grpo.py').read()); print('OK')"`

- [ ] **Step 6: Commit**

```bash
git add sage-python/scripts/train_topology_grpo.py
git commit -m "feat: 2-phase GRPO — Phase 1 structure-only, Phase 2 execution via TopologyRunner"
```

---

### Task 2: Run Phase 1 (structure learning, ~25 min)

- [ ] **Step 7: Launch Phase 1**

```bash
set -a && source .env && set +a && cd sage-python
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 nohup python -u \
    scripts/train_topology_grpo.py --mode grpo-phase1 \
    --sft-checkpoint models/topology_sft/ \
    > data/grpo_phase1.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 8: Monitor Phase 1 (check every 5 min)**

```bash
# After 5 min — should see steps progressing fast (~15s/step)
tail -20 data/grpo_phase1.log | grep -E "step|reward|loss|Parse"
```

Expected: ~15s/step, format_reward improving, structure_reward stable.

- [ ] **Step 9: Verify Phase 1 completion**

Phase 1 should complete in ~25 min. Check:
```bash
ls models/topology_grpo_phase1/adapter_model.safetensors
# Should exist if training completed
```

- [ ] **Step 10: Test Phase 1 output (YAML validity)**

```python
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-mini-instruct', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, 'models/topology_grpo_phase1/')
tokenizer = AutoTokenizer.from_pretrained('models/topology_grpo_phase1/')
model.eval()

import yaml
valid = 0
for i in range(10):
    prompt = '<|system|>You are a multi-agent topology designer. Given a task, generate an optimal agent topology in YAML format.<|end|>\n<|user|>Write a function to sort a list.<|end|>\n<|assistant|>'
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.4, do_sample=True)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict) and 'nodes' in data:
            valid += 1
    except:
        pass
print(f'Phase 1 validity: {valid}/10 ({valid*10}%)')
# GATE: must be >= 7/10 (70%) to proceed to Phase 2
# If < 70% → Phase 1 needs more epochs
"
```

**GATE:** Phase 1 validity must be >= 70% to proceed to Phase 2. If not, increase `num_train_epochs` to 3 and re-run.

- [ ] **Step 11: Commit Phase 1 results**

```bash
git add -f models/topology_grpo_phase1/adapter_config.json  # Don't add safetensors (too large)
git commit -m "checkpoint: GRPO Phase 1 complete — structure-only training"
```

---

### Task 3: Run Phase 2 (execution learning, ~3h)

- [ ] **Step 12: Kill any running processes**

```bash
kill $(ps -ef | grep train_topology | grep -v grep | awk '{print $2}') 2>/dev/null
```

- [ ] **Step 13: Launch Phase 2**

```bash
set -a && source .env && set +a && cd sage-python
PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 nohup python -u \
    scripts/train_topology_grpo.py --mode grpo-phase2 \
    > data/grpo_phase2.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 14: Verify Phase 2 starts correctly (5 min check)**

```bash
tail -30 data/grpo_phase2.log | grep -E "Agent provider|BigCodeBench|TopologyRunner|Topo:|Parse:|Exec:"
```

Expected:
- "Agent provider: Gemini 2.5 Flash"
- "Loaded 1140 BigCodeBench test cases"
- "Topo: RUNS_OK" or "Topo: PASSED" — real execution happening

- [ ] **Step 15: Leave running overnight. Check tomorrow.**

Pause/resume for PC move:
```bash
kill -STOP <PID>
kill -CONT <PID>
```

---

### Task 4 (FUTURE): Phase 2b — Multi-model execution

> This task is NOT for the current session. It documents the path forward for when Phase 2 converges.

**Goal:** Replace single Gemini Flash with per-node model assignment via ProviderPool.

**Prerequisites:**
- Phase 2 converges (execution reward improves over 100 steps)
- Phase 2 model generates topologies where structure matters (planner→coder→reviewer > coder alone)

**What changes:**

1. In `execution_reward.py`, replace single provider with ProviderPool:

```python
# Instead of:
provider, model = _get_agent_provider()
runner = TopologyRunner(graph=graph, executor=executor, llm_provider=provider, llm_config=config)

# Do:
pool = _get_provider_pool()  # ProviderPool with Google, DeepSeek, Kimi, MiniMax
runner = TopologyRunner(graph=graph, executor=executor, llm_provider=default_provider,
                        llm_config=default_config, provider_pool=pool)
```

2. TopologyRunner already resolves per-node model via `provider_pool.resolve(node.model_id)` (runner.py:122-126). No changes needed in TopologyRunner.

3. Before running, call `ModelAssigner.assign_models(graph, task_domain, budget)` to set `model_id` on each node based on `model_tier`.

4. The model_tier mapping (from SFT data):
   - `"reasoner"` → `gemini-3.1-pro-preview` (reasoning tasks)
   - `"smart"` → `gemini-2.5-flash` (code generation)
   - `"fast"` → `gemini-2.5-flash-lite` or `deepseek-chat` (simple formatting)

5. ProviderPool standalone creation (no boot needed):

```python
from sage.llm.provider_pool import ProviderPool
from sage.providers.openai_compat import OpenAICompatProvider

providers = {
    "google": OpenAICompatProvider(api_key=google_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", provider_name="google"),
    "deepseek": OpenAICompatProvider(api_key=ds_key, base_url="https://api.deepseek.com/v1", provider_name="deepseek"),
}
pool = ProviderPool(default_provider=providers["google"], registry=registry, providers=providers)
```

6. This creates a **3-phase training**:
   - Phase 1: Structure learning (format + structure rewards, no API)
   - Phase 2: Execution learning (single model, TopologyRunner)
   - Phase 2b: Model assignment learning (multi-model, ProviderPool)

**Estimated additional complexity:** ~50 lines in execution_reward.py. The infrastructure (ProviderPool, ModelAssigner, TopologyRunner per-node resolution) is already built and tested.

---

## Execution Summary

| Phase | Duration | API Cost | Reward Functions | What the Model Learns |
|-------|----------|----------|------------------|-----------------------|
| **Phase 1** | ~25 min | $0 | format + structure | Valid YAML, good structure (roles, edges, difficulty) |
| **Phase 2** | ~3h | ~$6 (Gemini) | format + structure + execution | Which topologies SOLVE problems (planner→coder→reviewer > coder alone) |
| **Phase 2b** (future) | ~3h | ~$10 (multi-model) | format + structure + execution (multi-model) | Which MODEL ASSIGNMENTS per node are optimal |

## Success Criteria

**Phase 1 gate:** >= 70% valid YAML on 10 test prompts.

**Phase 2 success (after 100 steps):**
1. Execution reward mean > 0.2 (some topologies produce working code)
2. Reward variance across K=8 generations > 0.1 (GRPO has signal)
3. Multi-node topologies score higher than single-node on average
4. format_reward stays above -0.5 (no YAML collapse)
