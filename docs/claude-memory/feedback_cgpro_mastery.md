---
name: cgpro plugin mastery
description: Operational rules for using the cgpro CLI plugin to consult ChatGPT 5.5 Pro — browser profile, project linking, streaming flags, response retrieval
type: feedback
originSessionId: b7b56b62-e6ea-4a71-965c-def15a6da3a2
---
User said 2026-04-28: "tu dois apprendre à maîtriser ce plugin il est extremement utile". cgpro is the canonical interface to GPT-5.5 Pro for design/verify steps in the cgpro→codex→claude cycle.

**Why:** cgpro IS the design oracle for substantial cycles — see [cgpro source of truth](feedback_cgpro_source_of_truth.md). Failures to drive it cleanly waste tokens, lose locked specs, or force re-deliberation. Master it.

**How to apply:**

1. **Browser profile lock** — cgpro drives Chrome via Playwright on the user-data-dir `C:\Users\yann.abadie\AppData\Local\cgpro\Data\profile`. The ChatGPT desktop app uses the same profile. Only ONE process can hold the lock.
   - Symptom: `browserType.launchPersistentContext: Target page, context or browser has been closed` + `Ouverture dans une session de navigateur existante`.
   - Fix: ask user to close the desktop app, OR start `cgpro daemon start` once and let all subsequent `cgpro ask` reuse the warm browser.
   - 2026-04-28 incident: original R1 DESIGN BG task hung at `started` event because desktop app held the lock. Cleaning up via `TaskStop` + asking user to close app worked.

2. **Project linking is per-cwd, manual.** New project on chatgpt.com does NOT auto-link. Run `cgpro project list` to find the project, then `cgpro project link "PROJECT_NAME"`. The link is stored locally under `~/.cgpro/projects/<gizmoId>/`. Conversations created AFTER the link auto-route there. Pre-existing conversations (like `cgpro_2026_04_26_review` UUID `69ee3d8d-...`) stay in their original location even after linking.
   - 2026-04-28: project YGN-SAGE = `g-p-69ed9637e63c8191b61c9741b50d1c01`. Linked to `github.com/yannabadie/ygn-sage`.

3. **Streaming flags choice — `--json` ALONE, NOT `--json --no-stream`.**
   - `--json` emits NDJSON (one JSON event per line). Partial captures survive disconnects.
   - `--no-stream` buffers until done THEN flushes. If connection severs mid-stream, NOTHING is captured.
   - `--no-stream --json` together = worst of both: no partial output, no stream events. Avoid unless network is rock-solid.
   - For long extended-thinking turns (>5 min), always use `--json` only + `--background` (browser off-screen).

4. **Resume vs new conversation.**
   - `cgpro ask` auto-resumes the most recent shell-session conversation for 30 minutes — fine for back-to-back follow-ups.
   - For an explicit named thread: `cgpro ask --resume <name|uuid>`. The persistent conversation survives across sessions.
   - The active YGN-SAGE thread is `cgpro_2026_04_26_review` (alias for UUID `69ee3d8d-6154-8392-b79a-3a0202e887d2`). All cycle 1 (R0-R4) consultations use `--resume cgpro_2026_04_26_review`.
   - For a fresh conversation: `--new-session` (auto-saves to a new UUID; save with `--save <name>` if you want to bookmark it).

5. **Response retrieval if stream got lost.**
   - `cgpro thread show <name> --json` only returns metadata (no messages). There's NO command to fetch existing messages from a thread directly.
   - Workaround: re-resume the thread with a follow-up like "your previous answer was lost; please re-emit verbatim". GPT sees the prior context and re-emits cheaply.
   - NEVER duplicate the original prompt verbatim — GPT will treat it as a new question. Frame it as recovery: "Case A — you already answered, re-emit. Case B — my prompt didn't land, here's the prompt".
   - **NEVER `TaskStop` a cgpro BG task that may be near completion.** 2026-04-28 R1 incident: original cgpro ask was hung at `started` event (browser lock). I called TaskStop. But the GPT-5.5 Pro response had ALREADY landed server-side; I only killed the local CLI that was *trying* to fetch it. Result: forced user to copy-paste the answer manually. If you suspect a cgpro BG is hung, check the response-file size every minute via Read; only TaskStop after 20+ min of zero growth.
   - **Easiest recovery if the CLI fails to read back:** ask the user to copy-paste from chatgpt.com into a local file (e.g. `A1.md`), then archive to `.tmp/cgpro_<item>_design_locked_spec.md`. Less elegant than re-resume, but always works.

6. **Daemon mode for speed.** Cold-starting Chromium adds 5-10s per call. `cgpro daemon start` keeps the browser warm. All subsequent `cgpro ask` auto-detect the daemon. Use for tight cgpro DESIGN→codex IMPLEMENT→cgpro VERIFY iteration loops.

7. **Pattern for cgpro DESIGN prompts (validated on 9/9 traps 2026-04-26 + R1 2026-04-28):**
   - "Same conversation. [What just shipped]. Moving to [next item]. Want your locked spec before I hand to codex."
   - Section "What I verified" with file:line references — proves you read the code.
   - Section "What I need from you" — 2-3 specific binary or trinary decisions, with your recommendation + reasoning.
   - Section "Non-questions (decisions I've already made)" — pre-empts cgpro's tendency to debate scope.
   - Closing: "Reply with locked spec: [X], [Y], [Z]. Or push back. I'll ship next."

8. **Output files live under `.tmp/`** (untracked). Naming convention: `cgpro_<item>_<phase>.md` (prompt) + `cgpro_<item>_<phase>_response.json` (or `.ndjson` for streaming) + `cgpro_<item>_<phase>_stderr.log`. The `_phase` is `design` (locked spec) or `verify` (post-implementation review).

9. **Don't try to invoke `cgpro` via the cgpro:cgpro Skill tool with `--prompt-file` arg.** That flag doesn't exist. Pipe via stdin or use bash command substitution: `cgpro ask "$(cat prompt.md)"`. Skill tool just loads the cgpro skill docs — actual invocation is via Bash.

10. **NEVER:** invoke cgpro from PR description, never expose API keys in prompts, never let cgpro drive bash commands directly (it's a thinker, not an executor — that's codex's role).

11. **Code-source insights (from `C:\Code\CGPro4Code` 2026-04-28).** The plugin is at `github.com/yannabadie/CGPro4Code`, also installed locally. Key implementation truths:
   - `--resume <id|name>` **disables project auto-routing** (`src/cli/commands/ask.ts` line ~80: "Auto-route into the ChatGPT Project... unless we're resuming a specific conversation"). Resumed convos stay in their original location regardless of cwd's project link. **Implication:** for cycle-spanning threads, you EITHER stay with `--resume <thread>` (and accept it stays out of the project) OR start a fresh conv per ticket with `--save <name>` (lands in project + bookmarked).
   - **Project memory auto-prepend** only fires for new convos with `gizmoId` resolved. Memory lives at `~/.cgpro/projects/<gizmoId>/memory.md` and is wrapped in `<!-- cgpro project memory ... --> ... <!-- end project memory -->` HTML comments around the user prompt.
   - **`assertNoDaemon` is enforced** for every command except `ask` (HTTP API) and `thread list` (cached). When daemon is up, `status`/`models`/`doctor`/`adopt`/`login`/`logout`/`chat`/`thread sync`/`project *` all refuse with "Stop the daemon first". Same applies to plain `cgpro ask` if no daemon — the profile lock conflicts with any other cgpro process AND with the desktop ChatGPT app.
   - **Hang root cause** (the 2026-04-28 R2 VERIFY incident): if `openSession()` (`src/browser/session.ts`) tries `launchPersistentContext` while the profile is locked by ChatGPT desktop OR a stale chromium process from a prior `project list`/`thread sync`, it hangs at "Target page, context or browser has been closed". The `started` event never fires, the BG runs at 0 bytes indefinitely. **Fix:** before any cgpro call, ensure no chromium/cgpro process is alive and the desktop app is closed.
   - **`CGPRO_DEBUG=1`** env var dumps every state transition from the orchestrator. Use this when `cgpro ask` hangs without obvious cause.
   - **`cgpro doctor`** audits selectors against live DOM. The first ✖ points at the broken selector. All DOM selectors live in `src/browser/selectors.ts` (single file). Patch + `npm run build` + retest.
   - **Daemon mode is the right thing for tight cycles** (cgpro DESIGN → codex → claude verify-local → cgpro VERIFY → ship, repeat). `cgpro daemon start` once, then all `cgpro ask` calls go through HTTP API at 127.0.0.1 with a 256-bit token from `~/.cgpro/daemon.json`. No cold-start, no profile conflicts. BUT mutex with `project show` / `thread sync` / etc — must `cgpro daemon stop` first if you need those.
   - **Fresh conv per ticket pattern (alternative to resume):** `cgpro ask --new-session --save <thread_name> ...` creates a brand-new conv, bookmarks it, and (if cwd is project-linked) auto-routes into the project with memory pre-pended. Subsequent calls in the same ticket use `--resume <thread_name>`. This pattern keeps the convo short (good for cgpro thinking budget) AND organized in the project sidebar. Used 2026-04-28 for R2 VERIFY after the cgpro_2026_04_26_review thread approach hit too many resume failures.
