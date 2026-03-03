# Active Plan: Phase 2 - ASI & Hardware-Aware Optimization

Based on the architectural audit by Gemini 3.1 Pro Preview, we are pivoting towards an ASI-ready, hardware-aware architecture. The first step is to enable `sage-core` (Rust) to dynamically evaluate the host machine's hardware capabilities.

## 1. Hardware Auto-Discovery Module (`sage-core`)
- [ ] Implement `sage-core/src/hardware.rs` to detect CPU topology (cores, threads).
- [ ] Implement detection for advanced instruction sets (SIMD, AVX-512, ARM NEON).
- [ ] Implement basic GPU/VRAM detection (if available).
- [ ] Expose `HardwareProfile` to Python via PyO3.

## 2. Refactor Memory Identifiers (ULID)
- [ ] Replace `uuid::Uuid` with `ulid::Ulid` in `sage-core/src/memory.rs`.
  - *Rationale*: ULIDs are lexicographically sortable by time and integer-based, drastically reducing heap fragmentation compared to String UUIDs.

## 3. Arrow Integration Prep
- [ ] Update `Cargo.toml` to include `arrow` and `ndarray` crates.
- [ ] Draft the new contiguous `WorkingMemory` struct using Arrow columnar formats.

## 4. Python SDK Integration
- [ ] Update `sage-python/src/sage/memory/working.py` to handle the new ULID formats.
- [ ] Expose the hardware profile to the `Agent` so it can dynamically decide whether to generate vectorized code (like the NumPy approach in H7).
