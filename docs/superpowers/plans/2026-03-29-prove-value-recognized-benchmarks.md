# Prove SAGE Value on Recognized Benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure SAGE's actual delta on GAIA (multi-step, tool-use) and SWE-bench (multi-file code), establish credible results on recognized 2026 benchmarks, and identify where topology actually helps.

**Architecture:** Run SAGE's existing pipeline (5-stage: CLASSIFY→DECOMPOSE→TOPOLOGY→ASSIGN→EXECUTE→LEARN) on GAIA Level 1+2 (validation split, ~165 tasks). Compare: (a) bare model (DeepSeek Chat direct), (b) SAGE with fixed sequential template, (c) SAGE with full TopologyEngine (6-path selection). This produces the first real ablation of topology value on a recognized multi-step benchmark.

**Tech Stack:** GAIA dataset (HuggingFace gated), SAGE pipeline, DeepSeek Chat API (primary), 7 providers for multi-provider mode, existing `gaia_bench.py` adapter.

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `sage-python/src/sage/bench/gaia_bench.py` | GAIA adapter — load, run, score | Exists, needs enhancement |
| `sage-python/src/sage/bench/gaia_ablation.py` | NEW: Ablation runner (bare vs template vs full engine) | Create |
| `sage-python/src/sage/bench/__main__.py` | CLI entry point — add `--type gaia` and `--type gaia_ablation` | Modify |
| `sage-python/src/sage/bench/runner.py` | BenchReport dataclass | Exists, no change |
| `docs/benchmarks/2026-03-29-gaia-results.json` | Raw results output | Generated |
| `docs/benchmarks/GAIA_ANALYSIS.md` | Analysis document | Create |

---

### Task 1: Fix GAIA bench adapter for real evaluation

**Files:**
- Modify: `sage-python/src/sage/bench/gaia_bench.py`
- Test: manual run on 5 GAIA tasks

The current adapter is minimal (54 lines). It needs:
1. Proper GAIA dataset access (gated, needs HF token)
2. Level filtering (Level 1, 2, 3)
3. File attachment handling (GAIA tasks include files)
4. Exact-match scoring (GAIA uses exact string match, not substring)
5. Cost tracking per task
6. JSONL output for leaderboard submission

- [ ] **Step 1: Access GAIA dataset**

```python
# Test dataset access
python3 -c "
from datasets import load_dataset
ds = load_dataset('gaia-benchmark/GAIA', '2023_all', split='validation',
                  token='$(grep HF_TOKEN /workspace/YGN-SAGE/.env | cut -d= -f2 | tr -d \")')
print(f'Loaded {len(ds)} tasks')
print(f'Columns: {ds.column_names}')
print(f'Level distribution: {dict(ds.to_pandas()[\"Level\"].value_counts())}')
print(f'Sample: {ds[0][\"Question\"][:200]}')
"
```

Expected: ~165 validation tasks across Levels 1-3.

- [ ] **Step 2: Rewrite gaia_bench.py with proper scoring and level filtering**

```python
# Enhanced GAIA adapter with:
# - Level filtering (--level 1, 2, 3, or all)
# - Exact match scoring (GAIA standard)
# - File attachment support
# - Cost tracking
# - JSONL output for leaderboard submission
```

- [ ] **Step 3: Test on 3 GAIA Level 1 tasks**

```bash
cd sage-python
python -m sage.bench --type gaia --level 1 --limit 3
```

Expected: 3 tasks completed, pass/fail for each, latency and cost reported.

- [ ] **Step 4: Commit**

```bash
git add src/sage/bench/gaia_bench.py
git commit -m "feat(bench): enhance GAIA adapter with level filtering, exact match, cost tracking"
```

---

### Task 2: Create GAIA ablation runner

**Files:**
- Create: `sage-python/src/sage/bench/gaia_ablation.py`

This is the core scientific contribution: measuring WHERE topology helps on GAIA.

Three conditions:
- **Bare model**: DeepSeek Chat direct (no SAGE pipeline)
- **SAGE-sequential**: SAGE pipeline with fixed sequential template (no topology selection)
- **SAGE-full**: SAGE pipeline with full TopologyEngine (6-path selection, kNN routing)

- [ ] **Step 1: Write ablation runner**

The runner executes the same GAIA tasks under all 3 conditions and produces a comparative report with statistical significance (Wilcoxon signed-rank, Cohen's d).

- [ ] **Step 2: Run ablation on GAIA Level 1 (5 tasks pilot)**

```bash
python -m sage.bench --type gaia_ablation --level 1 --limit 5
```

Expected: 3×5 = 15 runs, comparative table with pass rates per condition.

- [ ] **Step 3: Commit**

```bash
git add src/sage/bench/gaia_ablation.py
git commit -m "feat(bench): GAIA topology ablation (bare vs sequential vs full engine)"
```

---

### Task 3: Run full GAIA Level 1 evaluation

**Files:**
- Output: `docs/benchmarks/2026-03-29-gaia-level1.json`

- [ ] **Step 1: Run GAIA Level 1 full (all tasks)**

```bash
python -m sage.bench --type gaia --level 1
```

Expected: ~50 Level 1 tasks, 30-60 min, ~$5-10 API cost.

- [ ] **Step 2: Run ablation on Level 1 full**

```bash
python -m sage.bench --type gaia_ablation --level 1
```

Expected: 3×50 = 150 runs, clear statistical comparison.

- [ ] **Step 3: Analyze results and document**

Key questions to answer:
1. What is SAGE's GAIA Level 1 accuracy?
2. Does the full topology engine beat the sequential template?
3. Does the sequential template beat the bare model?
4. On which specific tasks does topology help/hurt?
5. What is the cost per task?

- [ ] **Step 4: Commit results**

```bash
git add docs/benchmarks/
git commit -m "results: GAIA Level 1 evaluation — SAGE vs bare model vs sequential"
git push origin main
```

---

### Task 4: Run GAIA Level 2 and prepare leaderboard submission

**Files:**
- Output: `docs/benchmarks/2026-03-29-gaia-level2.json`
- Output: `docs/benchmarks/gaia_submission.jsonl`

- [ ] **Step 1: Run GAIA Level 2**

```bash
python -m sage.bench --type gaia --level 2
```

- [ ] **Step 2: Generate leaderboard JSONL**

```python
# Format: {"task_id": "...", "model_answer": "..."}
```

- [ ] **Step 3: Submit to GAIA HuggingFace leaderboard**

Upload JSONL to https://huggingface.co/spaces/gaia-benchmark/leaderboard

- [ ] **Step 4: Document and publish results**

Write `docs/benchmarks/GAIA_ANALYSIS.md` with:
- Raw scores (Level 1, Level 2, combined)
- Ablation results (topology delta)
- Comparison with GAIA leaderboard (GPT-5 Mini 44.8%, OWL 69%)
- Honest assessment of where SAGE adds value

- [ ] **Step 5: Commit and push**

```bash
git add docs/benchmarks/
git commit -m "results: GAIA Level 1+2 with topology ablation and leaderboard submission"
git push origin main
```

---

### Task 5: Update HuggingFace model card and CLAUDE.md with results

**Files:**
- Modify: `CLAUDE.md`
- Modify: HF model card (via API)

- [ ] **Step 1: Update CLAUDE.md with GAIA results**

Add GAIA scores to the "Current State" section alongside BigCodeBench.

- [ ] **Step 2: Update HF model card**

If Path 6 checkpoint helped, document it. If not, document the framework value.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add GAIA benchmark results to project status"
git push origin main
```

---

## Success Criteria

| Criterion | Target | How to verify |
|-----------|--------|---------------|
| GAIA Level 1 runs to completion | 100% tasks attempted | No crashes, all tasks produce answers |
| Ablation has 3 conditions | bare, sequential, full | 3 separate result sets |
| Statistical significance measured | Wilcoxon + Cohen's d | p-values in report |
| Results on recognized benchmark | GAIA leaderboard | Submission accepted |
| Topology delta measured | Any (positive or negative) | Ablation comparison |
| Honest documentation | Results as-is, no cherry-picking | Analysis document |

## Cost Estimate

| Phase | Tasks | API calls/task | Cost/call | Total |
|-------|-------|---------------|-----------|-------|
| Level 1 full | ~50 | ~3 (pipeline) | $0.003 | ~$0.45 |
| Level 1 ablation | 3×50 | ~3 | $0.003 | ~$1.35 |
| Level 2 full | ~80 | ~5 | $0.005 | ~$2.00 |
| Level 2 ablation | 3×80 | ~5 | $0.005 | ~$6.00 |
| **Total** | | | | **~$10** |

## Notes

- GAIA is a gated dataset — requires HF token acceptance
- Exact match scoring — no partial credit, strict evaluation
- File attachments — some tasks include images/PDFs that the agent needs to process
- The combined RL training on GPU can continue running while we do the GAIA evaluation (CPU + API only)
- If GAIA shows topology helps → validates Path 6 direction
- If GAIA shows topology doesn't help → pivot to function-calling approach (MAS-Orchestra style)
