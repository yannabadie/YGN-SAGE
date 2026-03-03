# Active Plan: Memory Pillar & SOTA Integration

This plan focuses on finalizing the `Memory Pillar` (GraphRAG) and integrating all SOTA components into the core `Agent` runtime.

## 1. Finalize `MemoryCompressor` Agent
- [x] Implement `MemoryCompressor` logic in `sage-python/src/sage/memory/compressor.py`.
- [ ] Create concrete drivers for **Neo4j** (GraphDatabase protocol).
- [ ] Create concrete drivers for **Qdrant** (VectorDatabase protocol).
- [ ] Test the full compression cycle with a mocked LLM.

## 2. Integrate SOTA Components into `Agent` Loop
- [ ] Update `sage-python/src/sage/agent.py` to:
    - Automatically call `MemoryCompressor.step()` during the execution loop.
    - Support `StrategyEngine` with `vad_cfr` or `shor_psro` by default.
    - Use `SandboxManager` with snapshot-based warm-start for tools.

## 3. Implement MAP-Elites Mutator (Evolution Pillar)
- [ ] Implement code mutation logic using LLM.
- [ ] Implement evaluation scoring (fitness) using SOTA MARL solvers.
- [ ] Implement the population management for MAP-Elites.

## 4. Verification
- [ ] Run full system tests with the flagship `sage-discover` agent.
- [ ] Measure restore times for Docker snapshots (< 1s target).
- [ ] Verify GraphRAG retrieval accuracy.
