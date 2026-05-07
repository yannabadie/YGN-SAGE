---
name: Fix even pre-existing CI failures, find root causes, no sweeping under the rug
description: Yann's standing directive 2026-05-05: when CI is red, fix ALL surfaces — even pre-existing — at the root-cause level. Don't skip warnings, don't suppress errors, don't ratchet ceilings without per-item justification. Pair with cgpro for triage when uncertain.
type: feedback
originSessionId: 3b36b883-89b8-4bbd-a2fe-cc9d305be971
---
**Rule**: When CI is red, address every failing surface at the root cause, regardless of whether the regression is mine or pre-existing. Warnings count. "It was already broken" is not a reason to skip.

**Why**: Yann said explicitly 2026-05-05 17:38 (after CI debug round 1 partial success):
> "Fixes les erreurs, meme prééxistantes, trouve leurs causes racines, ne met pas la poussière sous le tapis"

And later, when CI was already 13/13 GREEN but `kimi/ERREURCI.md` showed an `unused_variable: sys_prefix` warning on Linux:
> "J'ai vu un petit warning [...] Corrige le."

Even a single non-blocking warning gets fixed. The standard is: clean output, not just green CI.

**How to apply**:

- **Use cgpro for triage**, not for licence to skip. cgpro identified which CI failures were "fake signals" (e.g., `cargo fmt --check` blocking before clippy reached the visible `node_count` assert) and which were real. Use that expertise to PRIORITIZE, not to defer.
- **Root cause means root cause**. cycle-11 CI debug examples:
  - `MagicMock.__int__` returns 1 by default → Stage 0 silently set ctx.system=1; fix is `SimpleNamespace` + correct method stub, NOT `# noqa` on the assert.
  - Test API drift (`MutationResult::unwrap()` removed cycle-10 P1, integration tests not updated) → fix is `try_into_graph().unwrap()`, NOT `#[allow(deprecated)]`.
  - Process-global static `LAST_NEW_USED_CACHE` raced sibling tests → fix is per-call `Arc<AtomicBool>` probe via `CacheOverrides.probe`, NOT `--test-threads=1`.
- **Justify per-item when ratcheting ceilings**. cycle-11 mypy_count 45→48: 3 new ignores documented inline (Windows ctypes, ulid fallback) before bump.
- **Never `--no-verify` / `# type: ignore` / `--allow-dirty` to ship over a red CI**. The discipline is fix → verify → push, not push → "we'll fix later".
- **Self-inflicted regressions count too**. cycle-11 cycle: I pushed a `cargo fmt`-clean commit locally (`673c27b7`), but CI Linux fmt was stricter and flagged 2 of MY changes — fixed in `1b67e9ce` as a separate commit, not amended.

**When to relax this rule**: never (within a session). The corollary: don't push something you wouldn't want investigated.

**Anti-pattern to avoid**: "CI was already red before my push so my push is fine." Wrong framing. The right framing: "the project's truth surface has dirt; my push doesn't add dirt; the dirt still gets cleaned in this session." See cgpro 2026-05-05 verdict on the cycle-11 CI debug — *propre, pas de poussière sous le tapis* — that's the bar.

**Reference incident**: 2026-05-05 cycle-11 CI debug. Started with 6 jobs red on `259b2066`. Ended with 13/13 GREEN on `e2e57ebe` after addressing all 6 surfaces + 2 latent test-API-drift bugs caught while cleaning + 13-file clippy hygiene across 7 feature variants + 1 wasmtime CVE bump. Commit chain `259b2066..e2e57ebe`, 18 commits. cgpro VERIFY: clean.
