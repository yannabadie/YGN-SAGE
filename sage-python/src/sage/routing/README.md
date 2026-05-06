# Routing

Dynamic model routing with dual Rust/Python shadow tracing.

## Modules

### `shadow.py` -- ShadowRouter

Dual Rust/Python routing with JSONL divergence traces. Runs both Rust SystemRouter and Python ComplexityRouter on every task, logs divergences as traces. 2-tier Phase 5 gate: soft (500 traces, <10% divergence) and hard (1000 traces, <5% divergence) before Python shadow can be retired.

## Rust Routing (sage_core)

The Rust core provides the performance-critical routing components:

- **SystemRouter** -- **PRIMARY**. Accuracy claim `routing.system_router_88pct` is `evidence_pending` in `docs/CLAIMS.yaml`. Cognitive system decision engine: hard constraints → structural scoring → telemetry-calibrated affinity → ContextualBandit model selection. `route_integrated()` is the Stage-0 end-to-end path; successful learning must return through `record_outcome_checked()`.
- **ContextualBandit** -- Per-arm Beta/Gamma posteriors, Thompson sampling, Pareto front. Configurable decay_factor, warm_start_from_affinities.
- **ModelRegistry** -- TOML-loaded model catalog with telemetry calibration (blended card prior + observed quality).
- **ModelAssigner** -- Per-node model assignment using ModelCard scoring (affinity + domain + cost). Filters by capabilities and budget.

## Bandit Attribution Contract

Stage 0 owns bandit model selection through `SystemRouter.route_integrated()`. The Python pipeline records a bandit outcome only through the same Rust router via `record_outcome_checked(decision_id, executed_model_id, executed_template, ...)`, so the selected full arm must match the executed full arm before a posterior update is allowed.

Skipped or refused attribution is terminal: multi-node ambiguous runs, oracle abstains, recorder mismatches, fallback routing, constraint-failed bandit picks, and off-policy outcomes cancel the pending `decision_id` instead of leaving a replayable label. Legacy `record_outcome()` is telemetry-only and consumes any pending token it receives; it never updates ContextualBandit posteriors.
