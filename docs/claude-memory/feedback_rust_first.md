---
name: Rust First Philosophy
description: User insists on Rust-first development. Performance-critical code in Rust, Python as tolerant fallback only. No heuristics anywhere. SOTA minimum, AI breakthrough at least.
type: feedback
---

**Rust first, Python tolerant.** All performance-critical paths must be in Rust (sage-core). Python is only for orchestration and fallback.

**Why:** The user values speed and reliability. Rust provides both via type safety, zero-cost abstractions, and native SIMD/ONNX performance.

**How to apply:**
- New modules: implement in Rust first, expose via PyO3, Python fallback optional
- Quality estimation: Rust QualityLabeler + Rust ONNX inference
- Routing: Rust SystemRouter + Rust kNN already primary
- Topology: Rust TopologyEngine + TopologyExecutor already primary
- Memory: Rust Arrow + S-MMU already primary
- Never add heuristic code (hardcoded thresholds, magic numbers, weight tuning)
- SOTA minimum benchmark results, aim for AI breakthrough (self-adaptive, not just well-architected)
