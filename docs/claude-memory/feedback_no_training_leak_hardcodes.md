---
name: No Training-Leak Model Hardcodes
description: Never hardcode model-id substrings from training-era knowledge (o1/o3/o4/legacy claude); use cards.toml + Context7 as sources of truth
type: feedback
originSessionId: 703d3a88-64a4-4696-b4ea-a3bd735310c2
---
When writing provider-quirk logic (temperature clamps, param renames,
etc.) or model-id routing, **do not** hardcode substrings like `"o1"`,
`"o3"`, `"o4"`, `"claude-3"`, `"gpt-4"` based on what I remember from
training. My cutoff (January 2026) leaks dead model tags into
codebases that only wire newer variants.

**Why:** Shipped commits on 2026-04-18 multiple `("gpt-5", "o1", "o3",
"o4")` tuples for an OpenAI temperature-clamp quirk. `cards.toml` only
has `gpt-5.4`, `gpt-5.4-pro`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.2`
— o-series models are not wired in this repo. User caught it.

**How to apply:**
1. Before adding any `"<tag>" in model` or `model.startswith("<tag>")`,
   grep `sage-core/config/cards.toml` for that tag. If no card matches,
   the branch is dead — don't write it.
2. Before adding a NEW quirk (temperature, max_tokens→max_completion_tokens,
   reasoning_effort, etc.), verify via Context7
   `/berriai/litellm` OR the provider's own docs. Cite the source in
   the code comment (e.g. `# Verified 2026-04-18 via Context7 /berriai/litellm`).
3. Reference `docs/patterns/knowledge-cutoff-checks.md` — the audit
   procedure for existing files.
4. When in doubt, the broad prefix (`"gpt-"`) is safer than a
   specific list — it covers future revs without re-editing.

**Audit commands (run for every provider PR):**
```bash
grep -rnE 'in (self\.model|model_id|effective_model|model)' sage-python/src/sage/providers/
grep -n "^id" sage-core/config/cards.toml
```
Any tag in a quirk tuple that isn't a substring of a cards.toml id = bug.

CLAUDE.md has this rule as directive #7.
