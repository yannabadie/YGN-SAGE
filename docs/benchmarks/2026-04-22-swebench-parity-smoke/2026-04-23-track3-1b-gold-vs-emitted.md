# Track 3.1b — gold vs emitted patch diff (semantic-miss tracers)

**Date:** 2026-04-23
**Follow-up from:** Track 3.1 invalidation — both tracers DID read the test
files, so "read test files first" isn't the answer. Question shifts to
"what did the agent do with what it read?"
**Tracers:** astropy-14182 (Arm A + Arm B), django-10924 (Arm A) — the three
tasks from the 2026-04-22 parity smoke where the patch either applied
cleanly and failed tests, or was malformed but attempted a real edit.

## astropy__astropy-14182 — Arm B (typed-only, applied, tests failed)

**Gold:** refactor `class RST(FixedWidth)` in `astropy/io/ascii/rst.py`:
- `__init__(self, header_rows=None)` forwarded to `super().__init__(...)`
- new `read(self, table)` method
- `write()` uses `self.header.header_rows` to position delimiter lines
- remove `SimpleRSTData.start_line = 3`

**Emitted:** `+    _supported_write_kwargs = ("header_rows",)` plus a no-op
whitespace change in `ui.py`. Context header says `class RST(Table):`.

**Failure category — context hallucination.** The emitted diff's context
line is **`class RST(Table):`** but the base_commit has **`class RST(FixedWidth):`**.
This is from the `model_patch` field, not our SR extractor — unified diff
emitted directly by the coder. The model ran 37 tool calls including
`read_file: astropy/io/ascii/rst.py` *twice*, then emitted a diff whose
context line disagrees with file bytes. The patch applied only because
`class RST(` shows up as a prefix of the real line and the single-char
addition didn't collide with anything. Tests failed because none of the
real semantic work (ctor signature, read method, write logic) happened.

The hallucination survived tool use. That's the interesting bit.

## astropy__astropy-14182 — Arm A (bash, malformed)

**Emitted:** aggressive rewrite of `RST.write()` — removes the
`if self.header_rows: raise TypeError(...)` guard, reorders
`header_lines`/`data_lines` computation, **deletes the `_write_header`
method entirely**.

**Failure category — over-aggressive scope.** Right direction (remove the
TypeError gate, re-plumb `write()`), but too many simultaneous changes and
the context/line positions drifted enough that the diff is malformed.
`git apply --fuzz=5` rejected it; LLM repair also failed. Tool count 38,
latency 207s — not budget-bound, the agent had time to verify and didn't.

## django__django-10924 — Arm A (bash, applied, tests failed)

**Gold:** one-line surgical change inside `FilePathField.formfield`:
```python
return super().formfield(**{
    'path': self.path() if callable(self.path) else self.path,
    ...
```
So `self.path` stays as stored (callable or plain), and resolution happens
at formfield-construction time — which is what Django's migration
autodetector needs to serialize callables correctly.

**Emitted:** coerces in `__init__`:
```python
if callable(path):
    path = path()
self.path = path
```

**Failure category — right intent, wrong layer.** Agent understood "path
might be callable" (correct read of the test file), but placed the
coercion at construction, which eagerly resolves and so **loses the
callable reference** the migration autodetector depends on. Tool count 41,
latency 291s — tight against the 300s budget, possibly rushed.

## Summary by failure mode

| Tracer | Mode | What prompt change would fix it? |
|---|---|---|
| astropy-14182 Arm B | Context hallucination (wrong base class in diff) | a pre-emission verifier that checks the coder's diff context lines against the file |
| astropy-14182 Arm A | Over-aggressive scope, malformed diff | "smallest change that passes tests" norm (fragile; hard to quantify) |
| django-10924 Arm A | Right idea, wrong layer (missed migration serialization) | domain-specific knowledge — not addressable at prompt level |

**Three distinct failure modes across three tracers.** Track 3.2 ("Step 0:
read test files first") — already invalidated in 3.1 because both tracers
read the test files. The three failure modes above give a complementary
invalidation: even if the agent had read MORE of the test, the hallucination,
over-scope, and wrong-layer errors are orthogonal to what's in the test file.

## Implication for Track 3.5 (the N=50 smoke)

Track 3.5 was framed as "unified-vs-unified with all Track-3 prompt changes".
With 3.1 invalidated, 3.2 deleted, 3.4 showing max_steps isn't the lever, and
3.3 still pending, the plausible remaining prompt-level changes are:

- **F1 only** (soften the SR-template step-7 contradiction — only affects
  Arm B since that template ships the contradiction)
- **A verification pass** — too big a change to batch with F1; its own
  decision gate

If all we can batch is F1 before the N=50 smoke, the expected lift on unified
is zero (F1 doesn't touch the unified template) and the expected lift on SR
is bounded by the 3-vs-4 patch gap at N=10 (within noise even at N=50 —
per-task variance ~10 pp, combined arm-gap SE ~2 pp at N=50 → need ≥4 pp
to beat noise, and we don't expect that from a prompt-contradiction softening
alone).

**Honest bottom line:** Track 3.5 as originally scoped has no changes
worth measuring yet. Options:

1. **Hold 3.5 open.** Add a verification pass (new spec) or inject
   domain-specific knowledge (scope creep) as the ACTUAL Track 3 change,
   then smoke.
2. **Retarget 3.5 as a noise-floor calibration.** Run unified-vs-unified
   (same config, same slice) 3× at N=50 to measure combined arm-gap SE
   empirically. Useful input for every future gate. No lift measured; just
   noise.
3. **Defer 3.5.** Accept Track 3 as-is: investigation found that the simple
   prompt-level levers are either wrong (3.1/3.2) or out of scope (3.4/3.1b
   failure modes). No smoke spend.

Recommendation pending advisor input.

## Artefacts

- Gold patches: `datasets` cache
  (`C:\Users\yann.abadie\.cache\huggingface\datasets\princeton-nlp___swe-bench_lite`)
- Emitted patches: `docs/benchmarks/2026-04-22-swebench-parity-smoke/2026-04-22-parity-{bash,typed}-meta.json`
