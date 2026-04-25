---
name: No TODO-looking strings in Rust source
description: Test-fixture strings that contain "TODO:" substrings must be named clearly so readers don't mistake them for real TODO comments; prefer simulated-LLM-output variable names
type: feedback
originSessionId: e6496ce0-f81e-4f1f-bc19-bd2fd75b67ef
---
Do not place bare string literals like `"TODO: implement X"` inline in
Rust test source, even when the string is a test input meant to exercise
a regex that searches LLM output for follow-up markers. Even if the
intent is correct, readers scanning the source file see "TODO" and
assume unfinished work.

**Why:** On 2026-04-20, while porting Python
`TopologyController._detect_emergent_subtask` to Rust in
`sage-core/src/topology/controller.rs`, I wrote tests using
`check_emergent_spawn("TODO: implement robust retry logic", 0)`.
The regex legitimately needs "TODO:" / "FIXME:" / "NOTE:" substrings
in the test input because that's what the runtime scanner looks for
in LLM output. But the user called it out:
*"Qu'est-ce que c'est que ça? 'TODO: implement robust retry logic'
Je ne veux pas de todo en plein milieu du rust...."*
— a TODO in production Rust source is a red flag even when it's just
test data.

**How to apply:**
1. Extract the LLM-output fixture into a module-level `const` with a
   name that makes the intent obvious:
   `const LLM_OUTPUT_WITH_TODO: &str = "... TODO: add unit tests ...";`
2. Add a comment block above the test-fixture constants explaining
   WHY the literal substring needs to appear in the test input.
3. Reference the constants from the test body; never inline raw
   test data that contains work-in-progress markers.
4. This rule applies to any source file a reviewer might grep for
   "TODO" / "FIXME" / "XXX" — not just Rust.

Commit pattern example: `sage-core/src/topology/controller.rs` after
the 2.5 refinement — see `LLM_OUTPUT_WITH_TODO`,
`LLM_OUTPUT_EMERGENT_BURST`, etc.
