"""V16 repair-only re-run: apply try_repair_patch to v13 predictions, then eval.

Isolates the repair-pipeline effect from generation variance
(minimax 529 storm). For each v13 prediction, clones the instance's
repo at base_commit, runs ``try_repair_patch`` against a live LLM, and
writes a NEW predictions.jsonl. Then runs swebench Docker eval.

Usage:
    python sage-python/scripts/swebench_repair_and_eval_v16.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Resource stub before swebench import.
from sage.bench.swebench_bench import SWEBenchBench, _build_task_prompt, load_swebench_dataset  # noqa: E402
from sage.bench.swebench_patch_repair import try_repair_patch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("v16")

V13_PREDICTIONS = Path(
    "C:/Users/yann.abadie/AppData/Local/Temp/sage_swebench_kfhxbz7i/predictions.jsonl"
)


def _setup_repo(instance: dict) -> str | None:
    """Clone repo at base_commit, return path. Mirrors SWEBenchBench._setup_repo."""
    repo = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    if not repo or not base_commit:
        return None
    repo_url = f"https://github.com/{repo}.git"
    tmp = tempfile.mkdtemp(prefix="sage_swe_v16_")
    repo_dir = os.path.join(tmp, repo.split("/")[-1])
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, repo_dir],
            capture_output=True, timeout=180, check=True,
        )
        result = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_dir, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            # Shallow clone didn't include commit; deepen and retry.
            subprocess.run(
                ["git", "fetch", "--unshallow"],
                cwd=repo_dir, capture_output=True, timeout=180,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_dir, capture_output=True, timeout=30, check=True,
            )
        return repo_dir
    except Exception as exc:
        log.warning("clone/checkout failed for %s@%s: %s", repo, base_commit, exc)
        shutil.rmtree(tmp, ignore_errors=True)
        return None


async def _build_llm():
    """Build a fresh PydanticAIProvider for the repair calls."""
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    # Match the v13 smoke: gemini-3.1-flash-lite-preview.
    model_id = "gemini-3.1-flash-lite-preview"
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY not set — LLM repair stage will no-op")
        return None
    return PydanticAIProvider(
        provider_name="google",
        model_id=model_id,
        api_key=api_key,
    )


async def repair_all(predictions: list[dict], instances_by_id: dict) -> list[dict]:
    llm = await _build_llm()
    out: list[dict] = []
    for pred in predictions:
        iid = pred["instance_id"]
        patch = pred.get("model_patch", "") or ""
        instance = instances_by_id.get(iid)
        if not patch or not instance:
            out.append(pred)
            continue

        repo_dir = _setup_repo(instance)
        if not repo_dir:
            log.warning("[%s] could not clone repo — skipping repair", iid)
            out.append(pred)
            continue

        try:
            repaired, stage = await try_repair_patch(
                patch=patch,
                repo_dir=repo_dir,
                llm=llm,
                problem_statement=instance.get("problem_statement", ""),
                instance_id=iid,
                llm_timeout=60.0,
            )
            if stage and stage != "unchanged":
                log.info("[%s] repair stage=%s (delta=%d chars)",
                         iid, stage, len(repaired) - len(patch))
            new_pred = dict(pred)
            new_pred["model_patch"] = repaired
            new_pred["_repair_stage"] = stage
            out.append(new_pred)
        finally:
            shutil.rmtree(Path(repo_dir).parent, ignore_errors=True)
    return out


async def main() -> int:
    if not V13_PREDICTIONS.is_file():
        log.error("v13 predictions not at %s", V13_PREDICTIONS)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"sage-v16-repair-{timestamp}"
    log.info("v16 repair-only run: run_id=%s", run_id)

    # Load v13 predictions
    predictions = [json.loads(ln) for ln in V13_PREDICTIONS.read_text().splitlines() if ln.strip()]
    log.info("Loaded %d v13 predictions", len(predictions))

    # Load swebench dataset to get repo / base_commit for each instance
    instances = load_swebench_dataset("lite")
    instances_by_id = {inst["instance_id"]: inst for inst in instances}

    # Apply repair pipeline
    repaired_preds = await repair_all(predictions, instances_by_id)

    # Log stage distribution
    from collections import Counter
    stage_counts = Counter(p.get("_repair_stage", "") for p in repaired_preds)
    log.info("Repair stage distribution: %s", dict(stage_counts))

    # Write new predictions file
    out_dir = Path(tempfile.mkdtemp(prefix="sage_swebench_v16_"))
    preds_path = out_dir / "predictions.jsonl"
    preds_path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in repaired_preds) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote repaired predictions to %s", preds_path)

    # Run eval
    bench = SWEBenchBench(
        system=None, dataset="lite",
        eval_timeout=600, max_workers=4, run_id=run_id,
    )
    results = bench.evaluate_with_harness(preds_path)

    print("\n" + "=" * 70)
    print(f"V16 repair-and-eval (run_id={run_id})")
    print("=" * 70)
    print(f"  Total: {results.get('total', 0)}")
    print(f"  Resolved: {results.get('resolved', 0)}")
    print(f"  Resolved rate: {results.get('resolved_rate', 0):.1%}")
    print(f"  Completed IDs: {results.get('completed_ids', [])}")
    print(f"  Resolved IDs: {results.get('resolved_ids', [])}")
    print(f"  Error IDs: {results.get('error_ids', [])}")
    print(f"  Repair stages: {dict(stage_counts)}")

    bench_dir = Path(__file__).resolve().parents[2] / "docs" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / f"{datetime.now().strftime('%Y-%m-%d')}-swebench-v16-repair-eval-report.json"
    out_path.write_text(json.dumps({**results, "repair_stages": dict(stage_counts)}, indent=2))
    print(f"  Saved to: {out_path}")
    print("=" * 70)

    return 0 if results.get("resolved", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
