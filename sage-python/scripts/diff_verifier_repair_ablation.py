#!/usr/bin/env python3
"""Slice 10B — diff verifier observe-vs-repair ablation.

cgpro VERIFY 2026-05-11 on conv ``cgpro_ygn_sage_global_analysis_20260510``:
Fix #5 verdict was MODIFY — "oui comme bras d'ablation, non comme nouveau
default headline. Le bon prochain slice court est: repair=observe baseline
N=5 déjà obtenu, puis repair=repair N=5 sur les mêmes instances/patches,
avec coût, nombre de hunks réparés, git apply success, et resolved rate.
Ne pas prétendre que repair améliore la compréhension du bug: il peut
corriger format/contexte, pas la logique."

This script consumes a canary run directory and, per task:
1. Loads the predicted patch from ``predictions.json``.
2. Clones the target repo at ``base_commit`` (using
   ``_setup_repo_for_canary`` from ``run_dryrun_arm_d.py``).
3. Runs ``verify_diff_context_with_reasons`` to count hunk mismatches.
4. **mode=observe**: records the mismatch count + outcome verdict
   and stops.
5. **mode=repair**: additionally calls ``repair_with_verifier_feedback``
   to produce a corrected diff, records cost / new patch / mismatches
   AFTER repair, and writes the repaired predictions next to the input.

Usage::

    # Observe baseline — no LLM cost
    python sage-python/scripts/diff_verifier_repair_ablation.py \\
        --run-dir docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/run \\
        --instances-json docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json \\
        --mode observe \\
        --output docs/benchmarks/2026-05-12-diff-verifier-observe.json

    # Repair ablation — LLM cost per mismatched hunk
    python sage-python/scripts/diff_verifier_repair_ablation.py \\
        --run-dir docs/benchmarks/2026-05-11-canary-patch-focused-prompt-profile/run \\
        --instances-json docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json \\
        --mode repair \\
        --llm-tier budget \\
        --output docs/benchmarks/2026-05-12-diff-verifier-repair.json \\
        --repaired-predictions docs/benchmarks/2026-05-12-diff-verifier-repair/predictions.json

Cgpro NON_GOALS:
- Do NOT flip ``SAGE_DIFF_VERIFIER_MODE=repair`` as default.
- Do NOT claim repair improves logic; it only fixes hunk format/context.
- Do NOT re-run the canary generation; reuse existing predictions.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Import the canary helpers we need (per-task repo clone). This avoids
# duplicating ~150 LOC of git plumbing in the ablation script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARM_D_PATH = _REPO_ROOT / "sage-python" / "scripts" / "run_dryrun_arm_d.py"


def _load_arm_d() -> Any:
    spec = importlib.util.spec_from_file_location("run_dryrun_arm_d", _ARM_D_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


arm_d = _load_arm_d()

from sage.bench.swebench_diff_verifier import (  # noqa: E402
    repair_with_verifier_feedback,
    verify_diff_context_with_reasons,
)


log = logging.getLogger("diff_verifier_repair_ablation")


def _load_instances(instances_json: Path) -> dict[str, dict[str, Any]]:
    """Read instances.json (list of dicts) and index by instance_id."""
    raw = json.loads(instances_json.read_text(encoding="utf-8"))
    return {inst["instance_id"]: inst for inst in raw}


def _load_predictions(run_dir: Path) -> list[dict[str, Any]]:
    pred_path = run_dir / "predictions.json"
    return json.loads(pred_path.read_text(encoding="utf-8"))


def _verify_one(
    diff: str,
    repo_dir: Path,
) -> dict[str, Any]:
    """Run verify_diff_context_with_reasons on a diff against a repo.

    Returns a dict with mismatch count + outcome string + per-hunk info.
    """
    result = verify_diff_context_with_reasons(diff, repo_dir)
    return {
        "mismatch_count": len(result.mismatches),
        "outcome": result.outcome,
        "mismatches": [
            {
                "file": m.file,
                "hunk_index": m.hunk_index,
                "old_start": m.old_start,
                "old_count": m.old_count,
                "kind": m.kind,
                "match_ratio": m.match_ratio,
            }
            for m in result.mismatches
        ],
        "reason_events": [
            {"reason": ev.reason, "scope": getattr(ev, "scope", None), "file": getattr(ev, "file", None)}
            for ev in result.reason_events
        ],
    }


async def _audit_task(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    mode: str,
    llm: Any | None,
    repair_budget_usd: float,
) -> dict[str, Any]:
    """One task ablation pass.

    Returns a dict with:
    - instance_id, repo, base_commit
    - observe stats (mismatches, outcome)
    - repair stats (new_patch_chars, repair_stage, post_repair_mismatches) if mode=repair
    """
    instance_id = instance["instance_id"]
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "mode": mode,
    }

    patch = prediction.get("patch", "")
    record["original_patch_chars"] = len(patch)

    if not patch:
        record["verdict"] = "empty_patch_skipped"
        return record

    # Clone the repo (reuse the canary's helper)
    repo_context = arm_d._setup_repo_for_canary(instance)
    record["repo_context_status"] = repo_context["repo_context_status"]
    tmp_root = repo_context.get("tmp_root")
    repo_dir = repo_context.get("repo_dir")

    try:
        if repo_context["repo_context_status"] != "ready":
            record["verdict"] = "repo_clone_failed"
            record["failure_reason"] = repo_context.get("failure_reason")
            return record

        # OBSERVE PASS
        observe = _verify_one(patch, Path(repo_dir))
        record["observe"] = observe
        record["mismatch_count_before_repair"] = observe["mismatch_count"]

        if mode == "observe":
            record["verdict"] = "observe_only"
            return record

        # REPAIR PASS (mode=repair)
        if observe["mismatch_count"] == 0:
            record["verdict"] = "no_repair_needed"
            record["repair"] = {"stage": "skipped_no_mismatches"}
            return record

        if llm is None:
            record["verdict"] = "repair_skipped_no_llm"
            record["repair"] = {"stage": "skipped_no_llm"}
            return record

        problem_statement = instance.get("problem_statement", "")
        # Build a HunkMismatch list to feed repair_with_verifier_feedback
        # — but observe returned dicts. Re-run the verifier to get
        # the typed list.
        verifier_result = verify_diff_context_with_reasons(patch, Path(repo_dir))

        new_patch, stage = await repair_with_verifier_feedback(
            llm=llm,
            problem_statement=problem_statement,
            broken_patch=patch,
            mismatches=verifier_result.mismatches,
            instance_id=instance_id,
            timeout=60.0,
            repair_budget_usd=repair_budget_usd,
        )

        record["repair"] = {
            "stage": stage,
            "new_patch_chars": len(new_patch),
            "patch_unchanged": new_patch == patch,
        }

        if new_patch != patch and stage in {"verifier_repair", "verifier_repair_empty"}:
            # Re-run verifier on the repaired patch
            post_observe = _verify_one(new_patch, Path(repo_dir))
            record["post_repair"] = post_observe
            record["mismatch_count_after_repair"] = post_observe["mismatch_count"]
            record["repaired_patch"] = new_patch
            record["verdict"] = (
                "repaired"
                if post_observe["mismatch_count"] < observe["mismatch_count"]
                else "repair_did_not_reduce_mismatches"
            )
        else:
            record["verdict"] = stage
        return record
    finally:
        # Clean up the tempdir
        if tmp_root or repo_dir:
            arm_d._cleanup_repo_dir(repo_dir, tmp_root=tmp_root)


async def _run_ablation(
    run_dir: Path,
    instances_json: Path,
    mode: str,
    llm_tier: str,
    repair_budget_usd: float,
    repaired_predictions: Path | None,
) -> dict[str, Any]:
    """Top-level ablation runner."""
    instances = _load_instances(instances_json)
    predictions = _load_predictions(run_dir)

    llm = None
    if mode == "repair":
        # Load .env so init_llm_provider sees the API keys. The canary
        # runs the sage CLI in a subprocess with pre-loaded env (slice
        # 8 _load_ygn_dotenv_into), but THIS script runs in the same
        # Python process as init_llm_provider — we have to pre-load.
        try:
            from dotenv import load_dotenv
            for parent in [Path.cwd()] + list(Path.cwd().parents):
                env_file = parent / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
        except ImportError:
            pass

        # Initialize a budget-tier LLM provider for repair calls.
        # `init_llm_provider` returns (provider, llm_config); we only
        # need provider.
        from sage.boot_providers import init_llm_provider
        provider, _llm_config = init_llm_provider(use_mock_llm=False, llm_tier=llm_tier)
        llm = provider

    per_task_records: list[dict[str, Any]] = []
    repaired_preds: list[dict[str, Any]] = []
    for pred in predictions:
        iid = pred.get("instance_id")
        instance = instances.get(iid) if isinstance(iid, str) else None
        if instance is None:
            per_task_records.append({
                "instance_id": iid,
                "verdict": "instance_metadata_missing",
            })
            continue
        record = await _audit_task(instance, pred, mode, llm, repair_budget_usd)
        per_task_records.append(record)

        # Build repaired predictions output (mode=repair only)
        if mode == "repair":
            new_pred = dict(pred)
            if record.get("repaired_patch"):
                new_pred["patch"] = record["repaired_patch"]
                new_pred["_diff_verifier_repaired"] = True
            else:
                new_pred["_diff_verifier_repaired"] = False
            repaired_preds.append(new_pred)

    # Aggregate stats
    tally: dict[str, int] = {}
    for r in per_task_records:
        v = r.get("verdict", "?")
        tally[v] = tally.get(v, 0) + 1

    total_mismatches_before = sum(r.get("mismatch_count_before_repair", 0) for r in per_task_records)
    total_mismatches_after = sum(r.get("mismatch_count_after_repair", 0) for r in per_task_records if "mismatch_count_after_repair" in r)
    tasks_with_post_repair = sum(1 for r in per_task_records if "mismatch_count_after_repair" in r)

    summary = {
        "schema_version": "diff_verifier_repair_ablation_v1",
        "run_dir": str(run_dir),
        "mode": mode,
        "llm_tier": llm_tier if mode == "repair" else None,
        "n_tasks": len(per_task_records),
        "verdict_tally": tally,
        "total_mismatches_before": total_mismatches_before,
        "total_mismatches_after_repair": total_mismatches_after if mode == "repair" else None,
        "tasks_with_post_repair_count": tasks_with_post_repair,
        "per_task": per_task_records,
    }

    # Write repaired predictions if requested
    if mode == "repair" and repaired_predictions is not None and repaired_preds:
        repaired_predictions.parent.mkdir(parents=True, exist_ok=True)
        repaired_predictions.write_text(
            json.dumps(repaired_preds, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Canary run dir containing predictions.json",
    )
    parser.add_argument(
        "--instances-json",
        required=True,
        type=Path,
        help="The instances.json that fed the canary (need repo + base_commit per id)",
    )
    parser.add_argument(
        "--mode",
        choices=("observe", "repair"),
        default="observe",
        help="observe: measure mismatches only; repair: also call repair LLM",
    )
    parser.add_argument(
        "--llm-tier",
        default="budget",
        help="LLM tier for repair mode (only used when --mode=repair)",
    )
    parser.add_argument(
        "--repair-budget-usd",
        type=float,
        default=0.5,
        help="Max USD per repair call (passed to repair_with_verifier_feedback)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the ablation summary JSON",
    )
    parser.add_argument(
        "--repaired-predictions",
        type=Path,
        default=None,
        help="(repair mode only) Where to write the repaired predictions.json",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    summary = asyncio.run(
        _run_ablation(
            run_dir=args.run_dir,
            instances_json=args.instances_json,
            mode=args.mode,
            llm_tier=args.llm_tier,
            repair_budget_usd=args.repair_budget_usd,
            repaired_predictions=args.repaired_predictions,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    if args.verbose:
        for r in summary["per_task"]:
            log.info(
                "  %s verdict=%s mismatches_before=%s mismatches_after=%s",
                (r.get("instance_id") or "?")[:50],
                r.get("verdict"),
                r.get("mismatch_count_before_repair"),
                r.get("mismatch_count_after_repair"),
            )

    print(
        f"Ablation complete: mode={args.mode} n_tasks={summary['n_tasks']} "
        f"verdict_tally={summary['verdict_tally']} -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
