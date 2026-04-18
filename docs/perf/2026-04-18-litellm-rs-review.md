# Review — `litellm-rs` (Rust port) as a berriai/litellm replacement

**Date**: 2026-04-18
**Source**: `majiayu000/litellm-rs` on crates.io (v0.4, 39 stars, Context7 `/majiayu000/litellm-rs`)

## Verdict: NOT YET — do not migrate

## Compatibility with our contract

Truth for providers we ship: `sage-core/config/cards.toml`.
We wire seven: google, openai, deepseek, **xai**, minimax, kimi, openrouter.

| Required | litellm-rs status |
|----------|-------------------|
| openai | ✅ |
| google (gemini) | ✅ |
| deepseek | ✅ |
| xai (grok) | **❌ missing from env-vars + docs** |
| minimax | ✅ |
| kimi | ✅ |
| openrouter | ✅ |
| Streaming | ✅ (`completion_stream`) |
| Tool/function calling | ✅ (DeepSeek example) |
| **Per-call `response_cost`** from provider | **❌** — only `calculate_cost(model, in_tokens, out_tokens)` token-estimate helper. Our P0.3 telemetry wiring (commit 5777864) specifically prefers LiteLLM's `response._hidden_params["response_cost"]` over estimates. Adopting litellm-rs regresses that fix. |
| PyO3 binding | ❌ (we'd author + maintain) |
| Circuit breaker / FrugalGPT cascade | not visible in docs |

## Maturity signals

- Pre-1.0 (v0.4), single maintainer, 39 stars, 730 commits.
- vs. berriai/litellm: ~22 k stars, team-maintained, daily PRs.
- Bus-factor risk high.

## Performance claim check

The stated motivation was "supprimer le saut GIL Python↔Rust au fan-out".
`tests/perf/profile_topology_runner_dispatch.py` shows Python dispatch
overhead is **0.4-1.2 %** on synthetic 1-3 s per-node latencies, and
**< 0.5 %** on real 2-30 s LLM calls. The GIL isn't the bottleneck
because asyncio releases it on every HTTP await — see
`docs/perf/2026-04-18-topology-dispatch-profile.md`.

Bottom line: the performance argument for swapping the LLM client to
Rust evaporates under measurement. LLM inference dominates wall-clock
by 2-3 orders of magnitude.

## Revisit triggers

Monitor the upstream repo for ALL of:

1. **v1.0 release** with stable semver.
2. **xAI / Grok provider** in the supported list (we need it).
3. **`response_cost` (or equivalent)** on `CompletionResponse` — i.e.
   the Rust port must faithfully pass through the provider's own cost
   figure, not a local token estimate.
4. **≥ 3 maintainers** OR corporate backing (bus-factor fix).

When 3 of 4 are satisfied, revisit with:
- A throwaway PyO3 prototype wrapping one provider (DeepSeek, our
  default).
- Same benchmark as `profile_topology_runner_dispatch.py` but with
  the actual provider — not a mock. Adoption threshold: **≥ 20 %
  wall-clock reduction on a 5-node fan-out** under real load.

Until then, stay on `berriai/litellm` + `sage.providers.litellm_provider`.

## What we'd have to do *today* if we migrated anyway

1. Fork + add xAI provider (~0.5-1 d, need their OpenAI-compat base URL).
2. Implement `response_cost` pass-through — either upstream PR or monkey
   patch the Rust crate. (~1-2 d, depends on their API surface.)
3. Write PyO3 binding that exposes the same `LLMProvider` protocol
   `sage-python/src/sage/llm/base.py` expects (generate, generate_stream,
   tools, usage). (~2-3 d with testing.)
4. Migration of every call site in `sage-python/src/sage/providers/`
   from `LiteLLMProvider` to the new wrapper. (~1 d.)
5. Re-run the P0.3 cost-telemetry test suite to confirm no regression.

**Cost: 5-7 engineering days for a < 1 % wall-clock win, with new
bus-factor + upstream-lag risk attached.** Don't do it.
