# Active Plan: Phase 2 - ASI & Hardware-Aware Optimization

Based on the architectural audit by Gemini 3.1 Pro Preview, we are pivoting towards an ASI-ready, hardware-aware architecture. 

## 1. Hardware Auto-Discovery Module (`sage-core`)
- [x] Implement `sage-core/src/hardware.rs` to detect CPU topology (cores, threads).
- [x] Implement detection for advanced instruction sets (SIMD, AVX-2).
- [x] Expose `HardwareProfile` to Python via PyO3.

## 2. Refactor Memory Identifiers (ULID)
- [x] Replace `uuid::Uuid` with `ulid::Ulid` in `sage-core/src/memory.rs`.
  - *Status*: Memory events now use compact 128-bit ULIDs, eliminating heap fragmentation.

## 3. Arrow Integration (H7 Hypothesis)
- [x] Update `Cargo.toml` to include `arrow` and `arrow-array` crates.
- [x] Implement contiguous columnar builders in `WorkingMemory` (Rust).
- [x] Expose `to_arrow()` capability to Python SDK.
  - *Status*: Foundation for zero-copy memory traversal is READY.

## 4. Benchmarking Protocol (AIO Ratio)
- [ ] Implement `sage-discover/benchmark_engine.py` to calculate the **Agentic Infrastructure Overhead (AIO)**.
- [ ] Run YGN-SAGE on **SWE-Bench Pro** mock tasks to measure SIMD retrieval gains.
- [ ] Compare AIO Ratio between pure Python and Rust/Arrow implementations.
