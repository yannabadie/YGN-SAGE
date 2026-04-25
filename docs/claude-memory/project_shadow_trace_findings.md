---
name: shadow_trace_divergence_analysis
description: Shadow trace evidence shows Rust router is better calibrated than Python — Phase 5 gate metric needs revision
type: project
---

## Shadow Trace Analysis (2026-03-12)

1090 traces collected. 49.6% divergence rate — Phase 5 gates FAIL.

**Root cause:** The divergence is NOT because Rust is wrong. It's because Python AdaptiveRouter without ONNX classifier is heavily S1-biased.

### Distribution comparison:
| System | Ground Truth | Rust SystemRouter | Python AdaptiveRouter |
|--------|-------------|-------------------|----------------------|
| S1 | 20% | 19.9% | 58.6% |
| S2 | 40% | 47.2% | 41.0% |
| S3 | 40% | 32.8% | 0.4% |

### Mismatch patterns:
- S3→S1 (Rust correct, Python wrong): 46.8% of mismatches
- S2→S1 (Rust correct, Python wrong): 32.3%
- S3→S2 (Rust closer): 19.4%

**Why:** Python AdaptiveRouter uses structural features only (no ONNX classifier). Structural features default to S1 for most text. Rust SystemRouter uses domain scoring from cards.toml which correctly identifies S2/S3 tasks.

### Direct accuracy measurement (2026-03-12):
| Router | Accuracy | Per-System |
|--------|----------|------------|
| kNN (Stage 0.5) | **92.0%** (46/50) | S1:70% S2:95% S3:100% |
| **Rust SystemRouter** | **88.0%** (44/50) | S1:80% S2:95% S3:85% |
| Heuristic baseline | 52.0% (26/50) | — |
| Python AdaptiveRouter | 44.0% (22/50) | S1:80% S2:50% S3:0% |

Rust misses: 2 S1→S2, 1 S2→S1, 3 S3→S2 (concurrent/verification tasks)

**How to apply:**
1. DONE: Measured Rust accuracy — 88% (4pp below kNN, 44pp above Python)
2. Action: Promote Rust SystemRouter as primary, Python is demonstrably worse
3. Consider wiring kNN into SystemRouter for S3 accuracy boost (S3 is 85% vs kNN's 100%)
4. Phase 5 gate should be replaced with accuracy-based evidence (done)
