# Slice 10C — synth-diff-fidelity-audit findings

**Source data**: 5-task slice 9 artefact at `docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/`.
**Audit script**: `sage-python/scripts/diff_fidelity_audit.py` (slice 10C delivery, commit TBD).
**Raw JSON**: `synth_diff_fidelity.json` next to this doc.
**cgpro VERIFY**: RF#B verdict was MODIFY — "audit fidelity, don't bypass synth". This delivers that audit.

## Headline finding

The synth-equivalent role's behavior **depends on topology**:

| Topology | Tasks | Verdict distribution |
|---|---|---|
| sequential 2-node (coder→mixer) | 1 (NodeBB) | 1× **preserved** |
| sequential 3-node (planner→coder→synthesizer) | 2 (teleport, tutanota 219bc) | 1× preserved + 1× cosmetic_drift (43 chars) |
| AVR 3-node (actor→verifier→judge/output) | 2 (webclients, tutanota db90) | **2× rewritten** — judge/output emits a diff for ENTIRELY DIFFERENT FILE PATHS than the actor |

**2 out of 5** tasks shipped a final patch whose target file path was **fabricated** by the AVR judge/output role. The actor identified the right file (presumably by reading the cloned repo); the judge/output regenerated a "plausible-looking" path that doesn't exist in the repo.

This explains the **0/5 Modal-grader resolved** result observed today: `git apply` cannot succeed when the patch's `--- a/<path>` doesn't exist in the checked-out tree.

## Per-task evidence

### webclients (AVR) — **rewritten**

| Stage | Role | Model | Files |
|---|---|---|---|
| pre  | actor | deepseek-v4-pro | `packages/react-components/containers/payments/SubscriptionModal.tsx` |
| post | judge | gemini-2.5-flash | `src/app/components/subscription/SubscriptionPromptModal.tsx` |

The actor's path is plausible for the protonmail/web-clients monorepo (it does ship a `packages/react-components/` workspace). The judge's path is a hallucinated alternative — directory layout that doesn't exist in this repo. The judge has effectively **substituted one file for another**.

### tutanota db90 (AVR) — **rewritten**

| Stage | Role | Model | Files |
|---|---|---|---|
| pre  | actor | deepseek-v4-pro | `src/api/main/LoginController.ts` |
| post | output | gemini-2.5-flash | `src/app/controllers/loginController.ts` |

Same pattern. The judge renamed the file path AND changed the case (`LoginController.ts` → `loginController.ts`).

### tutanota 219bc (sequential 3-node) — **cosmetic_drift**

| Stage | Role | Model | Files | Chars |
|---|---|---|---|---|
| pre  | coder | deepseek-v4-pro | `src/api/worker/EventBusClient.ts` | 3049 |
| post | synthesizer | gemini-2.5-flash | `src/api/worker/EventBusClient.ts` | 3006 |

Same file. 43-char delta (~1.4%) — probably trailing whitespace or a `+` line cosmetic difference. Verdict: synth respected the coder's diff.

### NodeBB (sequential 2-node) — **preserved**

| Stage | Role | Model | Files | Chars |
|---|---|---|---|---|
| pre  | coder | deepseek-v4-pro | (same single file as final) | 1195 |
| post | mixer | gemini-2.5-flash | (same) | 1195 |

Byte-identical diff. The mixer in sequential 2-node is a thin pass-through.

### teleport (sequential 3-node) — **preserved**

Same pattern as NodeBB: byte-identical diff from coder to synthesizer.

## Why this matters

cgpro warned us in VERIFY: "if synth alters diff → fix synth, don't bypass it". The audit reveals **AVR's judge/output role is doing more than just packaging the answer — it's regenerating the diff content from scratch**, and it doesn't have direct repo access to validate the file paths.

In **sequential** topology, the mixer/synthesizer is a wrap-up step that preserves the coder's emission. The coder is the canonical answer-writer.

In **AVR** topology, the judge is supposed to ADJUDICATE between actor + verifier, but in this run the verifier sentinel-ed (no critique), so the judge has only the actor's output to work with. Yet the judge produces a STRUCTURALLY DIFFERENT diff. Likely root cause: the judge's prompt asks for a "final consolidated patch" and the LLM regenerates rather than passing through.

## Recommended next actions

Per cgpro VERIFY's decision tree:

> "if synth alters diff → fix synth or prompt synth, not bypass architecture"

1. **Inspect the AVR judge / output role prompt**. The current template likely asks the LLM to "produce the final unified diff" without explicit instruction "if the actor's diff is valid, return it unchanged". This invites regeneration with hallucinated paths.

2. **Verify the actor's diffs against the cloned repo**. If the actor's file paths exist in `base_commit`, the actor's emission would be salvageable. We could run a second N=5 with topology forced to sequential to confirm a higher git-apply rate.

3. **Do NOT bypass the synth** (cgpro VERIFY explicit). The architectural contract is "last node wins". If sequential preserves and AVR rewrites, the right fix is at the AVR judge ROLE level, not at the extraction layer.

## Verdict tally

```
n_tasks: 5
verdict_tally:
  preserved: 2       (both sequential synth/mixer pass-through)
  cosmetic_drift: 1  (sequential synth, 43 chars)
  rewritten: 2       (BOTH AVR judge/output — different files)

any_rewritten: True (slice 10C audit gate: FAIL — judge regression is real)
```

## Limitations

- 2 AVR observations is a small sample. A second run with the same instances might still pick AVR for some tasks (bandit), but we'd want N≥5 AVR runs to confirm `rewritten` is the dominant mode (not a 50/50).
- The actor's diff might ALSO have wrong paths (need to grep the cloned repo at `base_commit` to confirm). If the actor was wrong too, the judge's rewrite isn't the only source of `git apply` failures.
- Verifier sentinel'd in both AVR runs, so we don't have a "verifier worked + judge rewrote anyway" data point.

## Follow-up

- **DO NOT** change the canary's extraction logic. The right fix is upstream.
- **Next slice (10A reproducibility logging)** should capture the judge's prompt so we can verify the regeneration-vs-passthrough question forensically.
- **Future slice** (NOT slice 10A-D): adjust AVR judge prompt to "return actor's diff verbatim if valid, otherwise reconcile". Requires cgpro DESIGN before implementation.
