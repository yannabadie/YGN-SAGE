---
name: Named-but-invisible tool anti-pattern in prompts
description: If a tool is listed in an LLM prompt with anti-affordance phrasing ("almost never the right tool", "optional — only use if…"), usage tends to 0. Avoid negative framing; reframe as positive use case.
type: feedback
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
If a tool is listed in an LLM system prompt but prefaced with discouraging
language — "(optional)", "almost never the right tool for day-to-day X",
"reach for it only when …" — the model will essentially never call it, even
when it would be the correct choice. Measured on SWE-bench 2026-04-23:
`search_exocortex` was registered, visible to coder/planner nodes, named in
both templates, and produced 0 calls across 616 tool.call entries in 4 smoke
logs (0.0% rate).

**Why:** models copy tone from the prompt. A bullet that carries a negative
frame reads as "don't use this" regardless of the "optional" qualifier.

**How to apply:** when adding a tool to a prompt, mirror the framing of an
already-used tool in the same prompt (e.g., `lookup_library_docs` in
SWE-bench). Positive use case first, then any budget / frequency hint
("at most once or twice per task"). Never open a bullet with a
discouragement. If the tool really shouldn't be on the path, remove it
from the prompt entirely — the "listed but discouraged" state is the worst
of both worlds: token cost with zero usage.

Incident log: 2026-04-21 audit
(`docs/audits/2026-04-21-exocortex-swebench-usage.md`) identified the gap
and applied R2 (add the bullet) but kept the anti-affordance phrasing.
2026-04-23 fix (`29987bc`): reframed to match `lookup_library_docs` tone.
