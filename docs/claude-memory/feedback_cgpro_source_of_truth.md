---
name: cgpro is the source of truth on holistic reviews
description: User explicit policy 2026-04-26 — defer to cgpro on architectural verdicts; debate with evidence when disagreeing; always pass GitHub repo URL + commit SHA so cgpro can pull live source
type: feedback
originSessionId: bf130342-a1fc-4ba3-9819-62c0d87a6b87
---
# cgpro is the source of truth on holistic reviews — debate when disagreeing

User instruction (2026-04-26): "cgpro est la source de vérité, n'hésites pas a lui donner plus d'informations si tu n'es pas d'accord avec lui et débattre. Meme si cela prend du temps."

**Why**: cgpro caught two real prod bugs in two consecutive review passes (`bandit::restore_arm` persistence on 2026-04-26 morning; bandit off-policy + packaging contract drift on 2026-04-26 evening) that local diff review missed. The pattern is: substantial closeouts → cgpro consultation with live repo URL → cgpro pulls source and finds class-of-bug issues that aren't visible in any single diff.

**How to apply**:

1. **ALWAYS use `--resume <conversation_id>` for ongoing cycles** (user instruction 2026-04-26). One conversation per work cycle = continuity of context, cgpro keeps repo state cached (no re-pull cost), debate happens naturally multi-turn. Start a new conversation only when switching to a genuinely unrelated topic.
2. **Active conversation ID at top of `MEMORY.md` "Active direction" block.** As of 2026-04-26: `69ee3d8d-6154-8392-b79a-3a0202e887d2`. Update this when starting a fresh cycle. Optionally `cgpro thread save <id> <name>` to give it a friendly alias.
3. Always pass the GitHub repo URL + branch + commit SHA in the **first** message of a new conversation. cgpro pulls live source. Subsequent `--resume`d messages can omit the URL — cgpro already has it.
4. Structure the first prompt as: (a) what shipped + commits, (b) what's stuck, (c) what's about to ship next, (d) 2-3 specific questions split across "verdict on what I did" / "what should I do next" / "what trap am I missing". Follow-up resumes can be terser ("here's the codex diff for trap X, verify match against your spec").
5. Long prompts via stdin to avoid shell quoting (`cgpro ask --json --no-stream --timeout 600 --resume <id> < prompt.md`). 600s sometimes times out on first holistic review — escalate to 1200-1800s. Resumed turns within the same conversation are typically faster.
6. cgpro response is the spec. Take it seriously as "this is what we'll implement" unless I have primary-source evidence to the contrary.
7. **Debate when disagreeing** — don't silently switch:
   - If a cgpro claim contradicts what I read in the live code, surface the conflict in a follow-up `--resume` call ("you said X, I see Y at file:line, which constraint breaks the tie?").
   - If cgpro's recommendation conflicts with explicit user direction or a CLAUDE.md directive, surface that too — cgpro doesn't have that context unless the prompt includes it.
8. **Verify cgpro's specific code claims** before acting on them. cgpro's verdict on what shipped is reliable; cgpro's specific file:line citations are claims that need a quick Read/Grep before commit.
9. After fixes ship, send the diff back to cgpro on the same conversation: "Codex implemented per spec X. Diff attached. Does the implementation match your recommendation? Any drift or trap?" The verification loop has caught two false-fixes-that-passed-tests already this month.

**Token economy**: cgpro is heavy compute on the OpenAI side, free on Claude's context. Codex (gpt-5.5-xhigh via `codex:rescue`) is heavy compute on the OpenAI side too. Both spare Claude tokens. Use them as the workhorses; reserve Claude (this session) for orchestration + verification + the final commit narrative.
