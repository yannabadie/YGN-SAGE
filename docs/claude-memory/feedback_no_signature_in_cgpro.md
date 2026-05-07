---
name: No Co-Authored-By signature inside cgpro prompts
description: Yann directive 2026-05-07 — don't include the Co-Authored-By footer when composing cgpro VERIFY/post-push prompts. Keep it for git commits only.
type: feedback
originSessionId: ae15e41f-58ed-438b-8f62-6e3feb79131b
---
Don't include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (or equivalent) inside cgpro VERIFY / DESIGN / post-push prompts.

**Why:** Yann directive 2026-05-07 — the signature is a git commit attribution convention; cgpro doesn't need it and it's noise that wastes tokens + can confuse the reviewer about whether it's part of the commit message draft or my own sign-off.

**How to apply:**
- Commit-message DRAFTS shown to cgpro for review can still END at the body — no need to append the trailer; cgpro understands the convention without it.
- The actual `git commit -m` invocation still gets the Co-Authored-By footer per the global system instructions for git commit creation.
- Apply to all cgpro communication (VERIFY pre-commit, post-push reports, NEXT_BLOCK requests, DESIGN_LOCK prompts).
