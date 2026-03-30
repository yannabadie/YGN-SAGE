# feat/rust-core — Rust Cognitive Engine

## Scope
All sage-core/ Rust: topology engine, MAP-Elites, CMA-ME, MCTS, bandit, routing, S-MMU memory, verification (OxiZ SMT, LTL), model assignment, embedder.

## Key Files
- `sage-core/src/topology/` — engine.rs, map_elites.rs, cma_me.rs, mcts.rs, mutations.rs, density.rs, reward.rs, templates.rs, executor.rs, verifier.rs
- `sage-core/src/routing/` — system_router.rs, bandit.rs, knn.rs, model_assigner.rs, model_registry.rs
- `sage-core/src/memory/` — smmu.rs, arrow_tier.rs, paging.rs, entity_graph.rs, embedder.rs
- `sage-core/src/verification/` — smt.rs, ltl.rs, quality_labeler.rs
- `sage-core/src/sandbox/` — validator.rs, wasm.rs (wasmtime v43), subprocess.rs, tool_executor.rs
- `sage-core/config/cards.toml` — 20 model cards

## Commands
```bash
cd sage-core && cargo test --no-default-features --lib  # 289 tests
cd sage-core && cargo clippy --no-default-features -- -D warnings
cd sage-core && maturin develop --features smt,onnx,cognitive,tool-executor
# Full build (Linux/RunPod):
maturin build --release --features smt,cognitive,tool-executor,sandbox,cranelift
```

## Recent Changes (March 2026)
- wasmtime v36 → v43 (March 29)
- should_evolve() + outcomes_since_last_evolve in engine.rs (memory-evolution)
- provider_hint +0.15 bonus in model_assigner.rs
- 18/18 Rust components operational, 54 PyO3 exports

## Out of Scope
Python SDK, training, UI.
