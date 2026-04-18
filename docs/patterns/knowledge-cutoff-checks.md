# Pattern — Don't hardcode model quirks from training-era knowledge

**Status**: required for any PR touching provider code.
**Created**: 2026-04-18 after the agent shipping this repo repeatedly
added `"o1" / "o3" / "o4"` tags to OpenAI quirk tables even though none
of those models are wired in `cards.toml` — a training-cutoff leak.

## The trap

LLM-assisted development produces model-quirk tables like:

```python
# BAD — from the agent's training snapshot, not from reality
if provider == "openai" and any(t in model for t in ("gpt-5", "o1", "o3", "o4")):
    force_temperature_1(params)
```

Every tag in that tuple is a claim:
1. "This model exists today."
2. "This model has this specific quirk."
3. "The quirk still applies as of the moment this line runs."

An agent with a January 2026 cutoff will happily write all three for
models that were deprecated in Q1, merged into a new family, or never
shipped. The code then carries dead branches that:

- Confuse future readers who go looking for the "o3" branch and find no
  caller.
- Rot silently — when a real "o3" model returns later with different
  semantics, the old branch fires wrong.
- Hide the gap between "what the agent thought was true" and "what
  cards.toml actually wires."

## The rule

**Source of truth for what OpenAI/Gemini/Anthropic/xAI/DeepSeek models
exist in our world is `sage-core/config/cards.toml`.** Not the agent's
memory, not Context7's LiteLLM snapshot, not the model provider's
general docs. Ours.

Any quirk table must:

1. Name only model-id patterns that **match at least one entry** in
   `cards.toml`. If you can't grep the tag to a card, delete it.
2. Comment the file:line of the card that proves the pattern is live.
3. On a removal, delete the quirk branch the same commit. No "keeping
   it in case we re-enable" — we won't re-enable it at the same
   version.

## The escalation

Before hardcoding a NEW model-quirk restriction, verify by two sources:

1. **Primary**: `cards.toml` entry for the model — confirms we ship it.
2. **Secondary (restriction itself)**: Context7 query against
   `/berriai/litellm` OR the provider's own docs (not cached training
   data). Include the quoted restriction in the comment above the
   quirk branch.

Example (good):

```python
# Source: cards.toml lines 135-261 — all gpt-5.x variants are wired.
# Quirk verified 2026-04-18 via Context7 /berriai/litellm:
# "GPT-5 reasoning models reject any temperature != 1". Branch only
# fires for substring "gpt-5" so any future 5.x rev is covered without
# re-edit; do NOT expand to o-series — not in cards.toml.
if self.provider_name == "openai" and "gpt-5" in model:
    _apply_gpt5_quirks(params)
```

## How to audit existing code

```bash
# Grep for string-match quirk tables in provider code
grep -rnE 'in (self\.model|model_id|effective_model)' sage-python/src/sage/providers/

# For each hit, confirm each tag is in cards.toml:
grep -n "^id" sage-core/config/cards.toml
```

Any tag in a quirk tuple that isn't a substring of any cards.toml id is
a bug — remove it and test the removal produces no regression.

## Known incidents

- **2026-04-18** — openai_compat.py `_apply_quirks` carried
  `("gpt-5", "o1", "o3", "o4")`. o1/o3/o4 aren't in cards.toml.
  Trimmed to `"gpt-5"` in commit following this doc.
