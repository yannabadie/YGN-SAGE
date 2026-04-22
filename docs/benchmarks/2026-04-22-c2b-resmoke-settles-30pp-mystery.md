# C2b re-smoke — the 30 pp lift was noise (2026-04-22)

## TL;DR

**The 30 pp gen-rate lift I attributed to C2c's prompt scaffolding does not exist.** A byte-identical re-run of the C2b prompt on the same 10-task SWE-bench Lite slice landed at **70 %** gen rate — comfortably inside C2c's 80 % band. Both the original C2b (50 %) and C2c (80 %) runs were within the topology-routing variance distribution. The "prompt priming" hypothesis is rejected.

The advisor's concern was correct: *"If C2b re-smoke lands at 70-80 %, then C2b was unlucky and C2c's apparent lift was routing variance."* — it landed at 70 %.

## What we ran

```
git show 19643b3:sage-python/src/sage/input/swebench.py > sage-python/src/sage/input/swebench.py
python -m sage.bench --type swebench --dataset lite --limit 10 --output docs/benchmarks/2026-04-21-swebench-c2b-resmoke.json
git checkout -- sage-python/src/sage/input/swebench.py   # restore HEAD
```

Identical slice to C2b (`19643b3`) and C2c (`68ef3fa` + `c6dbf41`) — HF dataset order deterministic, so all three smokes hit the same ten tasks (astropy-12907 → django-11019).

## Results

| Run | commit | PATCH | EMPTY | ERR | Gen rate |
|-----|--------|------:|------:|----:|---------:|
| C2b original | 19643b3 | 5 | 3 | 2 | **50 %** |
| C2c original | 68ef3fa + c6dbf41 | 8 | 2 | 0 | **80 %** |
| **C2b re-smoke (this run)** | 19643b3 prompt, HEAD code | **7** | **2** | **1** | **70 %** |

Per-task outcomes for the re-smoke:

| Task | Outcome |
|------|---------|
| astropy-12907 | EMPTY |
| astropy-14182 | PATCH (4020 chars) |
| astropy-14365 | PATCH (545 chars) |
| astropy-14995 | PATCH (592 chars) |
| astropy-6938 | PATCH (407 chars) |
| astropy-7746 | PATCH (761 chars) |
| django-10914 | EMPTY |
| django-10924 | PATCH (572 chars) |
| django-11001 | PATCH (2956 chars) |
| django-11019 | ERR (timeout 300 s) |

## Interpretation

Three runs, same slice, **50 / 70 / 80 %**. Spread = 30 pp. On N = 10 with ±10 pp per-task flip variance, this is exactly what topology-routing + provider-routing randomness produces. There is **no evidence that C2c's prompt scaffolding improved gen rate.** The `2026-04-21-c2c-smoke-results.md` document's "30 pp lift consistent with the hypothesis that the expanded prompt primes the model" is **retracted**.

## Implications for downstream steps

1. **`docs/benchmarks/2026-04-21-c2c-smoke-results.md` needs a correction note** pointing to this file and withdrawing the prompt-scaffolding claim.
2. **`docs/superpowers/plans/2026-04-21-universal-input-adapter-design.md` §C2c** — no change, the Context7 rewire itself still stands (tool is correctly registered, correctly ignored on intra-repo tasks, correctly invoked when a task actually needs docs — per the Step 3 probe below).
3. **`crystalline-crafting-shore.md` (the current plan) Step 4 validation criterion is insufficient.** A 5 pp delta on N = 20 cannot distinguish signal from the noise we just measured. The plan must add variance-controlled validation (paired ON/OFF runs, larger N, negative control) **before** any Step 4 code is written. Blocked until the criterion is revised.

## Decision

Step 4 implementation is **paused** pending a revised, variance-controlled validation methodology. Step 3's last task (BigCodeBench/129) will complete as a descriptive data point, not a decision gate.
