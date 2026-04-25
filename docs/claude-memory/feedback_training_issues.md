---
name: Training Issues Encountered (March-April 2026)
description: Debugging log for local training on Windows — tqdm crash, zero rewards, chat template, Unsloth incompatible, environment_factory failure
type: feedback
---

## Issue 1: tqdm crash on Windows nohup (FIXED)
tqdm writes to stdout which is invalid when running via `nohup`. Fix: `disable_tqdm=True` in GRPOConfig.
**Why:** Windows pipe handling differs from Unix. nohup redirects stdout but tqdm tries to write control characters.
**How to apply:** Always set `disable_tqdm=True` for background training on Windows.

## Issue 2: reward = 0.0 for all completions (FIXED)
Model generated 1024-token gibberish instead of YAML because prompts were passed as raw strings without chat template.
**Why:** Qwen-family models expect `<|im_start|>system\n...<|im_end|>` format. Raw text → model doesn't know to generate YAML.
**How to apply:** Pass prompts as list of chat message dicts (not strings). TRL applies `tokenizer.apply_chat_template()` automatically.

## Issue 3: Unsloth incompatible with Windows (KNOWN)
triton-windows has `AttrsDescriptor` import error. torchao also incompatible.
**Why:** triton-windows is a community port, not fully compatible with torch 2.6.
**How to apply:** Use TRL + PEFT directly. Skip Unsloth and torchao on Windows.

## Issue 4: Qwen3.5-4B vLLM bugs (KNOWN)
GDN (Gated DeltaNet) architecture causes illegal memory access on consumer GPUs.
**Why:** New architecture, vLLM support still being stabilized (March 2026).
**How to apply:** Use Qwen3-4B (not Qwen3.5-4B). Reserve newer architectures for pod H100.

## Issue 5: SSL certificate issues on corporate machine (KNOWN)
Google genai SDK ignores SSL verify override. DeepSeek and OpenAI work with `httpx.Client(verify=False)`.
**Why:** Corporate SSL inspection (GIE AD BRIVE) intercepts certificates. CLAUDE.md says "no proxy" but cert chain has "Missing Authority Key Identifier".
**How to apply:** Set `SAGE_SSL_VERIFY=false` in .env. For Google, use the OpenAI-compatible endpoint (not native genai SDK).
