---
name: No LiteLLM — Use PydanticAI Provider
description: LiteLLMProvider is deprecated. All new provider work must use sage.providers.pydantic_ai_provider.PydanticAIProvider
type: feedback
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
Do not import, extend, or add code paths through
`sage.providers.litellm_provider.LiteLLMProvider` or the `litellm` library.
The replacement is `sage.providers.pydantic_ai_provider.PydanticAIProvider`
and is the wired provider throughout the repo as of 2026-04-18
(`boot_pipeline.py:74` comment, `openai_compat.py` DeprecationWarning).

**Why:** User explicitly told me on 2026-04-20 "N'utilise pas litellm nous
l'avons remplacé par pydanticAI déjà." The migration is complete — litellm
references remaining in the tree are either deprecation shims, historical
comments, or legacy test files. Registry-lag hand-coded drops (gpt-5 temp
clamp etc.) that used to live in `litellm_provider.py` are not the active
code path any more.

**How to apply:**
- New provider integrations → `PydanticAIProvider.for_sage_provider(...)`
- When fixing a provider bug, check `pydantic_ai_provider.py` first, not
  `litellm_provider.py`.
- When referencing "the provider" in docs/commits/tests, default to
  PydanticAI.
- Do NOT add imports of `litellm` or `litellm_provider` in new files.
- The old memory `project_litellm_registry_lag.md` described a pattern
  that lived in the LiteLLM-era provider. Registry-lag fixes now live in
  PydanticAI-side model adapters or are mooted by pydantic-ai's
  per-provider model classes (OpenAIChatModel, GoogleModel, XaiModel...).
