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

## 2. Zero Heuristics
NEVER hardcode thresholds, magic numbers, or weight tuning. Every decision must be:
- Formally verified (Z3/OxiZ SAT/UNSAT)
- Learned (trained model, ONNX)
- Research-backed (cite the paper)
- If none available: abstain (return None), don't guess

Violations: QUALITY_BASELINE, QUALITY_LENGTH_WEIGHT — these are BANNED constants.

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
