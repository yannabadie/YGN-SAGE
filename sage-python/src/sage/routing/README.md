# Routing

Dynamic model routing with capability constraints and performance feedback.

## Modules

### `dynamic.py` -- DynamicRouter

Capability-constrained model selection with runtime feedback. Given a task and its complexity tier (from ComplexityRouter), DynamicRouter selects the best available model based on:

- **Capability matrix** -- Each provider declares capabilities (coding, reasoning, formal verification). The router filters to models that satisfy the task's requirements.
- **Performance feedback** -- Historical success rates, latency, and cost are factored into scoring. Poor-performing models are deprioritized.
- **Budget constraints** -- Respects per-task and cumulative cost limits from CostTracker.

**Safety:** Handles edge cases where all models are filtered out (fixed IndexError on empty scored list, identified during audit). Falls back gracefully when no model matches constraints.

## Integration

The DynamicRouter sits between the ComplexityRouter (which determines S1/S2/S3 tier) and the LLM providers (which execute the call). The flow is:

```
Task -> ComplexityRouter (tier) -> DynamicRouter (model) -> LLM Provider (execution)
```

Feedback from task outcomes flows back to update the router's scoring model.

### `shadow.py` -- ShadowRouter

Dual Rust/Python routing with JSONL divergence traces. Runs both Rust SystemRouter and Python ComplexityRouter on every task, logs divergences as traces. 2-tier Phase 5 gate: soft (500 traces, <10% divergence) and hard (1000 traces, <5% divergence) before Python shadow can be retired.

## Rust Routing (sage_core)

The Rust core provides the performance-critical routing components:

- **SystemRouter** -- Cognitive system decision engine: hard constraints → structural scoring → telemetry-calibrated affinity → ContextualBandit model selection. `route_integrated()` is the end-to-end path.
- **ContextualBandit** -- Per-arm Beta/Gamma posteriors, Thompson sampling, Pareto front. Configurable decay_factor, warm_start_from_affinities.
- **ModelRegistry** -- TOML-loaded model catalog with telemetry calibration (blended card prior + observed quality).
- **AdaptiveRouter** -- 4-stage learned routing: structural features → BERT ONNX classifier → entropy probe → cascade (behind `onnx` feature).
