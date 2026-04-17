# Sprint 6 — Decision Gate

**Status:** framework defined, execution gated on Sprint 5 ablation results.
**Date:** 2026-04-17
**Predecessor:** Sprint 5 ablation (docs/benchmarks/SWEBENCH_ABLATION_PROTOCOL.md)

After the SWE-bench Pro ablation lands, pick one of two branches based on
the `full` config pass rate. The criteria are deliberately sharp so the
decision does not drift into "run another experiment."

## Gate A — `full` pass rate ≥ 35% → v1.0 Release Candidate

If `full` crosses 35% on SWE-bench Pro (50+ tasks, McNemar p<0.05 vs bare)
and Python + Rust test suites are clean, ship a release candidate.

### v1.0 RC checklist

- [ ] Tag `v0.2.0-rc1` on `main` at the ablation SHA.
- [ ] Regenerate `docs/benchmarks/results.md` with the 4-config ablation
      table, CIs from 1000-bootstrap, and the environmental metadata
      (commit SHA, dataset version, model, date).
- [ ] Update `README.md` Benchmark Results table with SWE-bench Pro
      line (link to predictions JSONL for reproducibility).
- [ ] Publish the wheel to TestPyPI first, validate `pip install` from a
      clean env, then PyPI proper.
- [ ] Draft the arXiv paper: 4-page short paper framing the contribution
      as "first open-source system combining learned routing + runtime
      topology synthesis + autonomous tool synthesis + recursive self-
      invocation with formal verification." Keep it factual; MASBENCH
      breadth +22pp (p=0.015) and SWE-bench Pro delta are the headline
      numbers.
- [ ] Open a GitHub Discussion + tweet with the numbers. No hype — just
      commits, tests, and a link to the predictions file.

### Anti-patterns to avoid

- Cherry-picking tasks after the fact to push the number up.
- Training on SWE-bench Pro tasks (the dataset is held-out for
  evaluation only).
- Shipping before reproducing the ablation from a fresh clone on a
  second machine. Repro is a ship blocker.

## Gate B — `full` pass rate < 20% → Training revival

Architecture is not the bottleneck; the policy is. Revive the training
branch and run V2.1 GRPO from the Phase C checkpoint.

### Revival checklist

- [ ] `git checkout <training-branch>` (the code deleted on 2026-04-15
      in commit `b2f59ee`). If the branch name has shifted, grep the
      remote for `verl` or `sage-topology-policy` to find it.
- [ ] Restore the training extras in `pyproject.toml`
      (torch, transformers, trl, peft, datasets, bitsandbytes).
- [ ] Download `yannabadie/sage-topology-policy-local` Phase C adapter
      (best known checkpoint, 40% MASBENCH depth).
- [ ] Rebalance V2 data: 50%+ `create_topology`, 30% `adapt_topology`,
      20% multi-turn. Current 22/60 split caused the V2 SFT regression
      (memory: project_april15_training_parked.md → archive/
      project_local_training_status.md).
- [ ] **Do NOT use `environment_factory`** — it destroyed the
      `<tool_call>` format on 2026-04-01. Use plain `reward_funcs` with
      TRL `GRPOTrainer`. (memory: archive/feedback_grpo_v2_lessons.md).
- [ ] Wire Graph-GRPO per-edge credit (arXiv 2603.02701, +1.82%
      published) into `reward.py`.
- [ ] Wire Dr. MAS per-agent advantage normalization
      (arXiv 2602.08847) — fixes gradient instability when planner /
      coder / reviewer have divergent reward distributions.
- [ ] Re-run MASBENCH depth + SWE-bench Pro ablation with Path 6 enabled
      (`SAGE_ENABLE_PATH6=1`). Target: beat the `full` untrained
      baseline by ≥5pp on SWE-bench Pro.
- [ ] Re-evaluate at Gate A / B once the new policy is online.

## Gate C — 20% ≤ pass rate < 35% → Narrow improvements

Iterate on the highest-contribution component identified by the
ablation. Do *not* revive training yet and do *not* ship v1.0.

### Targeted work

- If `no_toolforge` causes the biggest drop: invest in richer
  `CreationTicket` context (predecessor node outputs, error traces) and
  broaden the AST/sandbox gates.
- If `no_sage_recurse` causes the biggest drop: raise
  `SAGE_RECURSION_MAX` to 4, add a "solved sub-task cache" keyed by
  sub_task hash so repeat recursions are instant.
- If `no_topology` causes the biggest drop: add the HyEvo code-node
  type (arXiv 2603.19639) as a first-class topology primitive.

Exit criterion: one of these pushes `full` over 35% → Gate A. Failure
after 3 targeted iterations → escalate to Gate B.

## Automated helper

`scripts/decide_next_phase.py` reads the ablation JSON and prints the
recommended gate. Use it after every Sprint 5 run so the decision is
traceable.

```bash
python scripts/decide_next_phase.py \
    docs/benchmarks/2026-04-17-swebench-pro-ablation.json
# -> Gate A / B / C + one-line rationale.
```
