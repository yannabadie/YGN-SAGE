---
name: April 15 — Training Pipeline Parked
description: Commit b2f59ee deletes verl/, scripts/, data/, models/ (-4.3GB) from main. Training lives in separate branch; checkpoints on HuggingFace.
type: project
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
## Strategic Decision (2026-04-15)

**Commit `b2f59ee` (2026-04-15 18:48):** Training code deleted from main branch.

### What was removed
- `sage-python/src/sage/verl/` (13 files) — veRL integration module
- `sage-python/scripts/` (25+ files) — SFT/GRPO training scripts
- `sage-python/data/` (17 JSONL/Parquet files) — training datasets
- `sage-python/models/` — checkpoint metadata
- 5 training plan docs (grpo-two-phase, grpo-v4, verl-runpod, targeted-training, json-toolcall)
- 15 training-specific tests
- Training deps from `pyproject.toml` (torch, transformers, trl, peft, etc.)
- Path 6 inference from `pipeline.py` and `llm_caller.py`
- `PolicyVerifier` stub replaced with restored info-flow + fan-limit checks

### Why
Training pipeline was not in active use since V2 GRPO broke (2026-04-01, format destroyed by `environment_factory`). Main branch focus shifted to orchestration correctness (unified entry point Phase 1-3 complete) and benchmark proof (SWE-bench as next milestone). Training code kept in separate branch for future use.

### Preserved state
- Phase C checkpoint (40% MASBENCH, best trained model) on HuggingFace: `yannabadie/sage-topology-policy-local`
- Nemotron 8B veRL checkpoints on HuggingFace: `yannabadie/sage-topology-policy-v2`
- Training code lives in separate branch when revival needed
- Archived memories in `memory/archive/` (8 files: feedback_grpo_v2_lessons, project_local_training_status, project_phase_c_complete, project_v2_adaptive_design, project_sft_data_issues, project_codecontests_tests_fix, project_next_session, research_verl_gigpo_verified)

**Why:** Focus on shipping correct orchestration + SWE-bench proof before resuming training. Training convergence is no longer THE blocker — architecture correctness is.

**How to apply:** Do not attempt `train_*.py` scripts — deleted. For inference, Phase C checkpoint via HuggingFace. When user asks about training, point to archive/ and training branch. If reviving training, expect to restore verl/, scripts/, data/ and training deps.
