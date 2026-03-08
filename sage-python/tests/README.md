# Tests

Test suite for sage-python. 695 tests (694 passed, 1 skipped as of March 2026).

## Running Tests

```bash
cd sage-python
pip install -e ".[all,dev]"
python -m pytest tests/ -v
```

## Organization

Tests are organized by module, with each `test_*.py` file corresponding to a source module or feature area.

### Core Agent
- `test_agent.py`, `test_agent_loop.py`, `test_agent_pool.py` -- Agent lifecycle, loop phases, sub-agent pool
- `test_agent_factory.py` -- Agent creation and configuration
- `test_agents_composition.py` -- SequentialAgent, ParallelAgent, LoopAgent, Handoff

### Memory (Tiers 0-3)
- `test_memory.py`, `test_memory_v2.py` -- Working memory (Arrow buffer)
- `test_compressor_smmu.py`, `test_smmu_context.py`, `test_smmu_e2e.py`, `test_smmu_injection.py` -- S-MMU write/read paths
- `test_embedder.py`, `test_embedder_rust_fallback.py`, `test_boot_embedder.py` -- 3-tier embedder fallback
- `test_causal_memory.py` -- CausalMemory with bounded growth
- `test_semantic_persistence.py`, `test_semantic_wiring.py` -- SemanticMemory SQLite
- `test_memory_agent.py` -- Entity extraction in LEARN phase
- `test_write_gating.py` -- WriteGate confidence + dedup

### Contracts and Verification
- `test_task_node.py`, `test_task_dag.py` -- TaskNode IR, DAG construction
- `test_verification.py` -- Pre/post verification functions
- `test_z3_contracts.py`, `test_z3_validator.py`, `test_z3_topology.py` -- Z3 SAT verification
- `test_policy_verifier.py` -- Info-flow, budget, fan-in/fan-out policies
- `test_dag_executor.py` -- DAG execution with VF gates
- `test_planner.py` -- Plan-and-Act decomposition
- `test_repair.py` -- CEGAR repair loop
- `test_cost_enforcement.py` -- Budget cap enforcement
- `test_stress_contracts.py` -- Contract system stress tests

### Routing and Strategy
- `test_metacognition.py` -- ComplexityRouter S1/S2/S3 classification
- `test_router.py`, `test_dynamic_router.py` -- Model routing and selection
- `test_speculative_routing.py` -- Speculative zone behavior
- `test_ablation.py`, `test_ablation_routing.py` -- Component ablation studies

### LLM Providers
- `test_llm.py`, `test_llm_providers.py` -- Provider interface and conformance
- `test_llm_mutator.py` -- LLM-driven mutation
- `test_config_loader.py`, `test_model_registry.py` -- TOML config and model resolution
- `test_openai_compat.py` -- OpenAI compatibility layer

### Guardrails and Resilience
- `test_guardrails.py`, `test_guardrails_wiring.py` -- Guardrail pipeline and integration
- `test_resilience.py` -- CircuitBreaker per-subsystem tracking
- `test_sandbox.py`, `test_sandbox_safety.py` -- Sandbox isolation

### Benchmarks
- `test_bench.py`, `test_humaneval.py` -- Benchmark runner and HumanEval
- `test_baseline_bench.py` -- Baseline comparison mode
- `test_truth_pack.py` -- Truth pack JSONL traces

### Other
- `test_event_bus.py` -- EventBus pub/sub
- `test_tools.py` -- Tool registry and execution
- `test_evolution.py`, `test_evolution_sota.py` -- Evolution engine
- `test_drift_monitor.py` -- Performance drift detection
- `test_integration.py`, `test_integration_v2.py`, `test_integration_phase3.py` -- End-to-end integration
- `test_e2e_real.py` -- Real LLM integration (requires API keys)
