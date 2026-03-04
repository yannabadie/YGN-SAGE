# SOTA Alignment ASI Plan

Implementation of exact algorithmic mandates and hardware optimizations for YGN-SAGE.

## Phase 1: Algorithmic Precision (VAD-CFR & SHOR-PSRO)
- [x] Task 1: Update `VolatilityAdaptiveSolver` in `sage-python/src/sage/strategy/solvers.py`.
    - Set `alpha = 1.5`, `beta = -0.1` as constants.
    - Confirm boost factor `1.1` and negative regret cap `-20.0`.
    - Ensure EWMA volatility sensitivity is set to `0.5`.
- [x] Task 2: Update `SHORPSROSolver` in `sage-python/src/sage/strategy/solvers.py`.
    - Implement decoupled `SolverMode` (TRAINING vs EVALUATION).
    - Training: lambda `0.3 -> 0.05`, diversity bonus `0.05 -> 0.001`.
    - Evaluation: lambda `0.01` (strict), diversity bonus `0.0`.
    - Set annealing schedule to 75 iterations.
- [x] Task 3: Update `EvolutionConfig` and `EvolutionEngine` in `sage-python/src/sage/evolution/engine.py`.
    - Add `hard_warm_start_threshold: int = 500`.
    - Implement logic to delay permanent population updates until threshold is met.

## Phase 2: Hardware Optimization (Zero-Copy H96)
- [/] Task 4: Update `sage-core/Cargo.toml`.
    - **BLOCKER**: Dependency conflict between `pyo3` (0.23/0.24/0.25), `numpy` (0.23), and `pyo3-arrow`.
    - Goal: Align all to a version that supports Zero-copy Numpy AND Zero-copy Arrow.
- [ ] Task 5: Refactor `sage-core/src/simd_sort.rs`.
    - Implement zero-copy integration using `PyArray1`.
    - Replace current scalar/copy partitioner with true AVX-512 `vcompressps` intrinsics.
- [ ] Task 6: Recompile and Benchmark.
- [ ] Task 7: Update and run SOTA tests.
- [ ] Task 8: Final SOTA Audit.
