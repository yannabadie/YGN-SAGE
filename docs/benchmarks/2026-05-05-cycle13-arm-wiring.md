# Cycle-13 4-arm wiring — concrete invocation contract

**Status**: scaffolding (cycle-12 closeout, 2026-05-05). Run is cycle-13 work.
**Companion**: `docs/benchmarks/2026-05-05-cli-baseline-plan.md` (the plan).
**Source-of-truth check**: this doc compiles the actual CLI signatures + env vars per arm; the plan describes the hypothesis.

---

## Why this doc exists

The cycle-13 plan says "4 arms × 50 tasks". This doc pins the EXACT invocation per arm so a benchmark runner can reproduce results without ambiguity — and so future maintainers can understand which behaviors were measured vs which were assumed.

Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505` verdict GO_TIER_1_PLUS_2), 6 traps must be addressed in the wire-up. Each is documented inline below.

---

## Arms

### Arm A — Claude Code direct

**Status**: not yet wired.
**Binary**: Claude Code CLI (assumed `claude` or `claude-code` on PATH).
**Invocation pattern** (cycle-13 work — exact CLI flags TBD):
```bash
claude --task "$problem_statement" \
       --workdir "$repo_clone" \
       --output-patch "$output_dir/$instance_id.patch"
```
**Cost metering**: Claude Code's own metering (read from its output).
**Telemetry**: Claude Code's own logs.
**Notes**:
- Claude Code chooses its own model. We do NOT force `claude-opus-4-7` for arm A — that's its decision.
- Per-task cap: 30 min wall-clock + $5 budget — implemented as outer-loop timeout + balance check.

### Arm B — pi-mono coding-agent direct

**Status**: not yet wired (cycle-13 starts here).
**Binary**: `pi` (from `@mariozechner/pi-coding-agent`, npm name verified 2026-05-05).
**Pin**: `@mariozechner/pi-coding-agent@0.73.0` exact (commit `dbcb473d6fdb96f60570b9ebe73e7aa6316fa8fb`).
**Invocation pattern** (cycle-13 work — verify against pi-mono README):
```bash
pi --task "$problem_statement" \
   --workdir "$repo_clone" \
   --model claude-opus-4-7 \
   --output-format json
```
**Env hygiene** (cgpro DESIGN E trap Q3):
```
PI_OFFLINE=1            # no network update checks during bench
PI_TELEMETRY=0          # no install telemetry beacons
PI_SKIP_VERSION_CHECK=1 # no startup version check
```
**Cost metering**: pi-mono emits cost in JSON output (per coding-agent README).

### Arm C — YGN-SAGE via pi-mono CLI

**Status**: SKIPPED in cycle-13 dry-run (cgpro DESIGN E Q2 verdict).
**Reason**: arm C requires a real `clients/pi-ygn-sage/` adapter (npm package wrapping `sage run --jsonl` as a pi-mono extension). That's substantial cycle-13 work and would block the dry-run.
**Cycle-13 plan**: validate arm A vs arm D first (the pure-orchestration delta). If A vs D shows ≥5pp lift, build arm C and re-run. If not, pivot before sinking adapter dev time.

### Arm D — YGN-SAGE direct via `sage run --jsonl`

**Status**: ready (cycle-12 prelude shipped `sage run --jsonl` in commit `d09bed4d`).
**Binary**: `python -m sage.cli run --jsonl` OR `sage run --jsonl` if console script registered.
**Invocation pattern**:
```bash
SAGE_LLM_TIER=reasoner \
SAGE_DIFF_VERIFIER_MODE=observe \
SAGE_OTEL_EXPORTER=none \
python -m sage.cli run --jsonl < <(echo '{"command":"prompt","payload":{"task":"$problem_statement"}}') \
    > "$output_dir/$instance_id.events.jsonl"
```
**Patch extraction** (cgpro DESIGN E trap Q5 / Pro grader format mismatch):
- The CLI emits a `final_result` event whose payload contains the agent's output.
- For SWE-bench Pro, we MUST extract a unified diff and re-format as:
  ```json
  {"instance_id": "<task_id>", "patch": "<unified-diff>", "prefix": "ygn-sage-arm-d-smoke"}
  ```
- **DO NOT use the SWE-bench Lite shape** (`{instance_id, model_name_or_path, model_patch}`). Pro's `swe_bench_pro_eval.py` expects the new shape.
**Cost metering**: `cli_complete` event payload `total_cost` field (cycle-12 prelude SAGE_CLI_PROTOCOL.md).
**Telemetry**: full RuntimeEventLog v0 stream on stdout (16 unit tests in `test_sage_cli_jsonl.py`).

---

## Acceptance gates per arm

Per cgpro DESIGN E trap Q5 (current `sage run --jsonl` is protocol-v0 partial):

### Arm D telemetry — what MUST be present at HEAD

- `cli_started` first event (pre-condition for protocol-v0 spec).
- `cli_complete` last event.
- `task_started` + `task_ended` per task.
- All emitted RuntimeEventLog events schema-valid (cycle-7 R6.1c versioning enforced).

### Arm D telemetry — what is NYI at HEAD (not a blocker)

- `cli_progress` heartbeat (spec'd, NOT YET EMITTED).
- `set_budget` mid-run inbound command (spec'd, NOT YET HANDLED).
- `cancel` mid-run cancellation token (spec'd, NOT YET ROUTED through pipeline).
- `cli_complete.payload.final_seq` (per spec equals `run_frame_summary.seq`; current impl emits `trace_dir` without `final_seq`).

These NYI items become FINDINGS in the dry-run report, not test failures. Cycle-13 phase 2 closes the gaps.

---

## SWE-bench Pro grader call

**Source**: `scaleapi/SWE-bench_Pro-os` repo, MIT license.
**Setup**: `pip install -r requirements.txt` from that repo + Modal account OR local Docker.
**Patch input format** (cgpro DESIGN E trap Q5):
```json
[
  {"instance_id": "<task_id>", "patch": "<unified-diff>", "prefix": "<run_label>"},
  ...
]
```
**Grader invocation** (per repo README + helper_code/gather_patches.py):
```bash
python helper_code/gather_patches.py \
    --directory <pred_files_dir> \
    --prefix ygn-sage-arm-d-smoke \
    --output predictions.json

python swe_bench_pro_eval.py \
    --raw_sample_path <huggingface_metadata.json> \
    --patch_path predictions.json \
    --scripts_dir <repo_clone_dir>
```
**Cycle-13 dry-run cutoff** (cgpro DESIGN E trap Q4 — Docker is the long pole):
- If Docker image pull or grader eval exceeds 15 min for the 1st task: stop.
- If API spend exceeds $5 across all attempted tasks: stop.

---

## JSONL framing discipline (both pi-mono RPC and SAGE_CLI_PROTOCOL)

Per cgpro DESIGN E trap Q6 + SAGE_CLI_PROTOCOL.md framing rule:
- **LF-only delimiter** (NOT CRLF, NOT Unicode line separators).
- **Strict UTF-8 encoding**, no BOM.
- One JSON object per line, each line is independent (no streaming JSON across lines).
- Reader MUST split on `\n` (single byte 0x0A), NOT use Node `readline.createInterface()` or Python `for line in f:` if the source is binary.

Tests for arm D capture this via `tests/test_sage_cli_jsonl.py:test_jsonl_only_lf_delimited` (cycle-12 prelude). Same constraint applies to pi-mono RPC mode per pi-mono `coding-agent/docs/rpc.md`.

---

## Stratification (per fetch script)

Per cgpro DESIGN E trap Q5 — driven by metadata, NOT hardcoded language list:

```
group by (task_size, language)
fallback: round-robin across buckets, prefer unseen repos
exclude: any task_id in _KNOWN_BUG_INSTANCES (currently empty,
         tracks SWE-bench_Pro-os open issues)
```

Implementation: `sage-python/scripts/swebench_pro_fetch.py`.
Output: `data/swebench_pro/n10/instances.json` + per_task/<id>.json + manifest.json.
Idempotent re-runs via `--seed 42` produce byte-identical output.

---

## Status changes

- 2026-05-05 (cycle-12 closeout): Tier 1 scaffolding (this doc + fetch script + dataset map already wired).
- TBD (cycle-13 dry-run gating): Tier 2.0 grader-shape canary (no API).
- TBD (cycle-13 dry-run): Tier 2.1 arm D smoke on 1-2 tasks (~$5 API).
- TBD (cycle-13 main): arms A + B + D × N=50 stratified (~$240-460).
- TBD (cycle-13+): arm C wired if A vs D delta justifies it.

## References

- Cycle-13 plan: `docs/benchmarks/2026-05-05-cli-baseline-plan.md`.
- SAGE_CLI_PROTOCOL v0: `docs/contracts/SAGE_CLI_PROTOCOL.md`.
- Runtime integrity ledger (9 invariants): `docs/contracts/runtime-integrity-ledger.md`.
- SWE-bench Pro repo: `github.com/scaleapi/SWE-bench_Pro-os`.
- HuggingFace dataset: `huggingface.co/datasets/ScaleAI/SWE-bench_Pro`.
- pi-mono v0.73.0 source: `github.com/badlogic/pi-mono` commit `dbcb473`.
- pi-mono npm packages: `@mariozechner/pi-coding-agent@0.73.0`, `@mariozechner/pi-ai@0.73.0`.
- cgpro DESIGN E verdict: `cgpro_pi_mono_pivot_20260505` 2026-05-05 GO_TIER_1_PLUS_2.
