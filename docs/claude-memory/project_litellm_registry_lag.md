---
name: LiteLLM Registry Lag for GPT-5.4 + Kimi-K2.5
description: litellm.drop_params=True is insufficient for models where registry incorrectly lists unsupported params as supported — keep explicit hand-coded drop
type: project
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
LiteLLM's global `drop_params = True` flag silently filters any parameter
the target model's registry flags as unsupported. **BUT** it only helps
when the registry is correct. For some models the registry lags behind
the provider's actual API behaviour — the param appears under
`supported_openai_params` even though the live endpoint rejects it.

**Why:** Observed 2026-04-18, litellm v1.83.4 (SAGE's pinned version).
`litellm.get_model_info('gpt-5.4')['supported_openai_params']` lists
`'temperature'`. OpenAI's production API returns 400 "invalid
temperature: only 1 is allowed for this model". drop_params forwards
the param → OpenAI rejects. Same pattern for `openai/kimi-k2.5`
(custom-base Kimi via OpenAI-compat prefix bypasses litellm's native
moonshot handling).

`litellm.register_model(dict)` lets us override COST fields but NOT
`supported_openai_params` — that lookup lives in a separate hardcoded
table inside the library.

**How to apply:**

1. Keep `litellm.drop_params = True` as a safety net for params the
   registry IS correct about.
2. For models where it isn't (gpt-5, kimi-k2.5 today), keep explicit
   hand-coded drops in `sage-python/src/sage/providers/litellm_provider.py::generate()`
   with an inline comment citing this memory.
3. Before adding new hand-coded drops, first verify via
   `litellm.get_model_info('<model>')` whether the param is actually
   listed — if yes and the API still rejects, this is another
   registry-lag case.
4. Periodic refresh trigger: on each `pip install -U litellm`, re-run
   the intercept test
   `python -c "from sage.providers.litellm_provider import LiteLLMProvider; ..."`
   (shape in commit a56c37c) to check if the registry has caught up.
   When it has, delete the hand-coded drop for that model.

**Commits that illustrate the saga:**
- 9e8d2ca: initial GPT-5 temp clamp to 1.0 (didn't help — API rejects
  any value incl 1.0)
- 0695191: drop temp entirely for GPT-5 — correct
- 4eceb01: tried to delete in favor of drop_params — FAILED (this memory)
- a56c37c: restored explicit drop with registry-lag explanation

Do not remove the hand-coded drop without evidence the registry now
agrees with live API behaviour.
