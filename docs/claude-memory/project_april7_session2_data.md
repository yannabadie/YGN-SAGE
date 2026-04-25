---
name: April 7 Session 2 — Data & Findings
description: 18 commits, BigCodeBench v1 37.2% + v2 45.2% (partial), MASBENCH stats, CORAL EvolutionMemory
type: project
---

## Benchmark Data (preserved)

### BigCodeBench Hard v1 (37.2%, 55/148, COMPLETE)
- Report: `docs/benchmarks/2026-04-07-bigcodebench-hard-instruct.json`
- Predictions: `sage-python/docs/benchmarks/2026-04-07-predictions-hard-instruct.jsonl`
- Full pipeline traces in JSONL (_trace per task)

### BigCodeBench Hard v2 (45.2%, 14/31, PARTIAL — stopped)
- Log: `docs/benchmarks/2026-04-07-benchv2-partial-log.txt`
- Changes: MiniMax pre-filter + stronger AVR repair (reasoner tier) + 1 retry
- Improvement: +8pp over v1 at same task count (45.2% vs 37.2%)

### MASBENCH Statistical Analysis (COMPLETE, N=50 per axis)
- Stats: `docs/benchmarks/2026-04-07-masbench-ablation-stats.json`
- **breadth: p=0.015**, d=+0.456 — ONLY significant axis
- depth/horizon/parallel/robustness: all p>0.05 — NOT significant
- "+27pp" claim is misleading — only breadth has statistical proof

## Key Findings

1. **Topology helps ONLY on breadth (omega>=3)** — AdaptOrch thresholds confirmed
2. **BigCodeBench Hard has omega~1.3** — topology is NOT the lever there
3. **MiniMax was causing 20% of failures** — pre-filter fixed it
4. **Stronger AVR repair (reasoner tier)** saves ~4 extra tasks per 30
5. **EvolutionMemory (CORAL Phase 1)** implemented + wired — lazy init, SQLite WAL
6. **Obsidian vault** tracked in git (48 files, 20 corrections)

## 18 Commits (23ab78b → c92df0b)
All pushed to origin/main.
