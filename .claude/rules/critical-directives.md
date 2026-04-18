---
paths:
  - "**/*.py"
  - "**/*.rs"
---

# Critical Directives — NEVER VIOLATE

## 1. Rust First, Python Tolerant
Performance-critical code MUST be in Rust (sage-core). Python is orchestration + fallback only.
- New modules: Rust first → PyO3 export → Python fallback optional
- Check sage-core/src/ for existing Rust module before writing Python
- Routing: Rust SystemRouter (88%) + Rust kNN (92%) are PRIMARY
- Quality: Rust QualityLabeler (Z3 formal) is PRIMARY
- Topology: Rust TopologyEngine (6-path) is PRIMARY
- Memory: Rust Arrow + S-MMU is PRIMARY

## 2. Minimal Heuristics
Prefer learned/verified decisions. When heuristics are necessary, follow this hierarchy:
- **Best**: Formally verified (Z3/OxiZ) or learned (kNN, bandit, ONNX)
- **Acceptable**: Research-backed initial values, documented as "subject to ablation" (e.g., THETA_GOOD=0.7)
- **Acceptable**: Engineering safety limits (MAX_RETRIES, cache bounds) — necessary for stability
- **Banned**: Arbitrary magic numbers without justification or calibration plan
- **Banned**: QUALITY_BASELINE, QUALITY_LENGTH_WEIGHT — these are dead heuristics

Current thresholds in TopologyController (THETA_GOOD=0.7, THETA_CRITICAL=0.3, etc.) and
engine.rs (similarity>0.7, quality>0.5) are calibrated initial values. A TopologyBench
ablation sweep is planned to optimize them. task_len>500/1000 heuristics should be replaced
by kNN embedding features when practical.

## 3. No Corporate Proxy
This machine has NO proxy. NEVER add:
- `verify=False` on HTTP clients
- SSL bypass workarounds
- `REQUESTS_CA_BUNDLE=""`
Direct HTTPS works. If SSL errors occur, the problem is elsewhere.
SSL bypass is controlled by `SAGE_SSL_VERIFY=false` env var (default: verify=True).

## 4. kNN Router is Primary
- KnnRouter (92% GT accuracy) — real router, arXiv 2505.12601
- Rust SystemRouter (88% GT) — domain scoring from cards.toml
- ComplexityRouter heuristic (34% GT) — DEAD CODE, emergency fallback only
- NEVER test/benchmark/optimize the heuristic router

## 5. Evidence Before Assertions
- Run `pytest tests/ -v` before claiming "tests pass"
- Run benchmarks before claiming scores
- Provide before/after metrics with statistical significance
- Date all benchmark results (pre-pipeline vs post-pipeline)

## 6. Don't Hardcode Model Quirks from Training Knowledge
Source of truth for which OpenAI/Gemini/Anthropic/xAI/DeepSeek models
exist in this repo is `sage-core/config/cards.toml`. Not the agent's
training snapshot. Before adding a model-name substring check (e.g.
`"o1" in model`, `"claude-3" in model`), verify the substring matches
at least one id in cards.toml. If it doesn't, delete the branch.

Before adding a NEW quirk (temperature clamp, token-param rename,
etc.), verify the restriction itself via Context7 `/berriai/litellm` or
the provider's own docs — not cached training data — and cite the
source in the code comment.

See `docs/patterns/knowledge-cutoff-checks.md` for the full audit
procedure and a log of known incidents.
