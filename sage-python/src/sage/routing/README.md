# Routing

Dynamic model routing with dual Rust/Python shadow tracing.

## Modules

### `shadow.py` -- ShadowRouter

Dual Rust/Python routing with JSONL divergence traces. Runs both Rust SystemRouter and Python ComplexityRouter on every task, logs divergences as traces. 2-tier Phase 5 gate: soft (500 traces, <10% divergence) and hard (1000 traces, <5% divergence) before Python shadow can be retired.

## Rust Routing (sage_core)

The Rust core provides the performance-critical routing components:

- **SystemRouter** -- **PRIMARY** (88% GT accuracy). Cognitive system decision engine: hard constraints → structural scoring → telemetry-calibrated affinity → ContextualBandit model selection. `route_integrated()` is the end-to-end path.
- **ContextualBandit** -- Per-arm Beta/Gamma posteriors, Thompson sampling, Pareto front. Configurable decay_factor, warm_start_from_affinities.
- **ModelRegistry** -- TOML-loaded model catalog with telemetry calibration (blended card prior + observed quality).
- **ModelAssigner** -- Per-node model assignment using ModelCard scoring (affinity + domain + cost). Filters by capabilities and budget.
