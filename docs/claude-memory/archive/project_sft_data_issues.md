---
name: SFT Data Quality Issues
description: 4 issues found in topology_sft_combined.jsonl (2624 entries, GPT-5.4) — last-node prompt missing code return instruction, GSM8K bias, too few complex topologies
type: project
---

## SFT Data Analysis (2026-03-19)

Dataset: `sage-python/data/topology_sft_combined.jsonl` — 2624 entries, 100% GPT-5.4

### What's Good
- 100% YAML validity, good difficulty↔node_count correlation
- Substantive prompts (median 490 chars), diverse roles (5+), 3 edge types
- Model tiers varied: reasoner (1331), fast (2818), budget (868)

### 4 Issues

1. **CRITICAL: 4% of last nodes mention code return** → NO_CODE=92% in Phase 2. Fixed with Gemini followup (86% PASSED) but next SFT cycle must inject "Return ONLY the final Python code in a ```python block" in last node prompt.

2. **IMPORTANT: GSM8K = 50% of dataset, all "simple"** → biases model toward 1-node topologies. Phase 2 excludes GSM8K anyway. Next cycle: cap at 30% or remove.

3. **MODERATE: Only 2.2% complex (57/2624)** → model rarely generates 5+ node topologies. Need 200-300 complex entries from CodeContests distillation.

4. **MINOR: topology_sft.jsonl (98 entries) is noise** → empty prompts, no edges. Rename to _DEPRECATED.

**How to apply:** These are fixes for the NEXT SFT cycle, not the current GRPO run. Phase 2 is running with followup compensation.
