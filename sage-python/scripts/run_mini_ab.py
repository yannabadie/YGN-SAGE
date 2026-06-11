"""MINI_2B_UNGRADED_A_VS_D — paired applicability & failure-class bench.

cgpro NEXT_BLOCK_ID = MINI_2B_UNGRADED_A_VS_D_APPLICABILITY_AND_FAILURE_CLASS
(2026-06-11): before spending $15-20 on a graded arm-A-vs-arm-D N=10, run
the ~$2-3 UNGRADED paired comparison and decide:

    Arm D <= Arm A on non-empty + applyability + verifier-clean
        -> pivot to product diagnosis (patch generation quality);
    Arm D >  Arm A clearly
        -> GO graded 2.b.

Arms (same pinned instances, run PAIRED per instance):

- **Arm A** — controlled single-call baseline: ONE reasoner-tier LLM call
  with the IDENTICAL ``patch_focused`` prompt, worktree, and
  verifier/repair chain as arm D. Orchestration is the only variable
  (the cycle-13 wiring doc's "pure-orchestration delta"). NOTE: this is
  deliberately NOT the wiring doc's Claude-Code arm A (a
  product-competitiveness question, out of scope for the mini).
- **Arm D** — the full SAGE pipeline via the existing canary runner
  (``run_dryrun_arm_d._run_one_task``), unchanged.

Per-instance metrics (cgpro contract): ``patch_non_empty``,
``git apply --check`` applicability against a fresh worktree,
verifier outcome pre/post repair, provider/cost audit, and the failure
class: EMPTY_PATCH / NOT_UNIFIED_DIFF / COUNT_MISMATCH /
CONTENT_MISMATCH / APPLY_FAILED / PLAUSIBLE_PATCH.

No Modal, no official grading, no learning-evidence claims.

Usage:
    python sage-python/scripts/run_mini_ab.py \
        --instances-json docs/benchmarks/2026-05-11-canary-n5-graded/instances/instances.json \
        --limit 10 \
        --output-dir docs/benchmarks/<date>-mini-2b-ab \
        --tier reasoner --arm-a-tier reasoner \
        --budget-usd 2.0 --global-budget-usd 8.0 \
        --provider-allowlist google,deepseek --provider-denylist openai
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_dryrun_arm_d import (  # noqa: E402
    _annotate_diff_verifier,
    _build_repair_llm,
    _cleanup_repo_dir,
    _repair_patch_with_feedback,
    _setup_repo_for_canary,
    _swebench_extract_patch,
)
from run_dryrun_arm_d import _run_one_task as _arm_d_run_one_task  # noqa: E402

log = logging.getLogger("sage.bench.run_mini_ab")


# ── Metrics helpers ─────────────────────────────────────────────────────────


def _git_apply_check(patch: str, repo_dir: str) -> tuple[bool, str]:
    """Cheap decisive applicability: ``git apply --check`` in the worktree.

    Returns ``(ok, detail)``; ``empty_patch`` short-circuits.
    """
    if not patch.strip():
        return False, "empty_patch"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".patch", delete=False, encoding="utf-8", newline="\n"
    ) as fh:
        fh.write(patch if patch.endswith("\n") else patch + "\n")
        patch_path = fh.name
    try:
        proc = subprocess.run(  # noqa: S603 — git trusted, paths ours
            ["git", "-C", repo_dir, "apply", "--check", "--verbose", patch_path],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0:
            return True, "applies"
        stderr_tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        return False, f"git_apply_exit={proc.returncode} {stderr_tail.strip()}"
    except subprocess.TimeoutExpired:
        return False, "git_apply_timeout"
    finally:
        try:
            Path(patch_path).unlink(missing_ok=True)
        except OSError:
            pass


_COUNT_CLASS = {"hunk_body_count_mismatch", "malformed_hunk_header"}
_CONTENT_CLASS = {"content_mismatch", "fuzzy_below_threshold", "file_missing"}


def _failure_class(
    *, patch: str, verifier_outcome: str | None, apply_ok: bool
) -> str:
    """cgpro contract taxonomy. ``git apply`` is the decisive cheap signal
    when the verifier abstains; a clean/abstained verdict that still fails
    to apply is APPLY_FAILED."""
    if not patch:
        return "EMPTY_PATCH"
    if apply_ok:
        return "PLAUSIBLE_PATCH"
    outcome = verifier_outcome or ""
    if outcome == "not_unified_diff":
        return "NOT_UNIFIED_DIFF"
    if outcome in _COUNT_CLASS:
        return "COUNT_MISMATCH"
    if outcome in _CONTENT_CLASS:
        return "CONTENT_MISMATCH"
    return "APPLY_FAILED"


def _repo_file_tree(repo_dir: str, *, max_files: int = 400) -> str:
    """``git ls-files`` tree for the localization call, capped."""
    try:
        proc = subprocess.run(  # noqa: S603
            ["git", "-C", repo_dir, "ls-files"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        files = (proc.stdout or b"").decode(
            "utf-8", errors="replace"
        ).splitlines()
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if len(files) > max_files:
        files = files[:max_files] + [
            f"... (+{len(files) - max_files} more files)"
        ]
    return "\n".join(files)


def _parse_file_list(
    reply: str, repo_dir: str, *, max_files: int = 6
) -> list[str]:
    """Extract existing repo-relative paths from the localization reply."""
    import re

    candidates: list[str] = []
    for raw in (reply or "").splitlines():
        line = raw.strip().strip("`*-\u2022 ").strip()
        if not line:
            continue
        if " " in line:
            found = re.findall(r"[\w./\\-]+\.[A-Za-z0-9_]+", line)
            line = found[0] if found else ""
        line = line.strip("`'\"")
        if not line:
            continue
        rel = line.replace("\\", "/").lstrip("./")
        if (Path(repo_dir) / rel).is_file() and rel not in candidates:
            candidates.append(rel)
        if len(candidates) >= max_files:
            break
    return candidates


def _files_block(
    repo_dir: str, rel_paths: list[str], *, max_chars_total: int = 60000
) -> str:
    """Concatenate selected file contents with headers, capped."""
    blocks: list[str] = []
    budget = max_chars_total
    for rel in rel_paths:
        try:
            text = (Path(repo_dir) / rel).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        chunk = (
            f"### FILE: {rel}\n```\n{text[: max(0, budget - 40)]}\n```\n"
        )
        blocks.append(chunk)
        budget -= len(chunk)
        if budget <= 1000:
            break
    return "\n".join(blocks)


_LOCALIZE_PROMPT = """You are a senior engineer. Given a bug report and the \
repository file listing, name the files (max {max_files}) most likely \
needing changes to fix the bug.
Reply with ONE repo-relative path per line, nothing else.

## Bug report
{problem}

## Repository files
{tree}
"""

_PATCH_PROMPT = """You are a senior engineer fixing a bug. Produce a \
unified diff patch.

STRICT OUTPUT CONTRACT:
- Output ONLY a unified diff (```diff fenced or raw), nothing else.
- Use a/ and b/ path prefixes; context lines MUST match the file content
  shown below byte-for-byte; correct @@ hunk headers.
- Minimal change that fixes the bug.

## Bug report
{problem}

## Relevant files (current content)
{files_block}
"""


# ── Arm A: Agentless-lite two-call baseline ──────────────────────────────────


def run_arm_a_task(
    instance: dict[str, Any],
    *,
    llm_factory: Any = None,
    arm_a_tier: str = "reasoner",
    verifier_mode: str = "repair",
    repair_budget_usd: float = 0.50,
    repair_timeout_s: float = 180.0,
    prompt_profile: str = "patch_focused",
    call_timeout_s: float = 600.0,
    provider_allowlist: tuple[str, ...] = (),
    provider_denylist: tuple[str, ...] = (),
) -> dict[str, Any]:
    """ONE direct LLM call with the arm-D prompt + the arm-D verifier and
    repair chain. Returns the per-instance metric record."""
    instance_id = instance["instance_id"]
    started = time.monotonic()
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "arm": "A",
        "patch_non_empty": False,
        "apply_ok": False,
        "apply_detail": None,
        "failure_class": "EMPTY_PATCH",
        "provider": None,
        "model": None,
        "usage": None,
        "_diff_verifier_outcome": None,
        "_diff_verifier_outcome_pre_repair": None,
        "_verifier_repair_stage": None,
        "repo_context_status": None,
        "elapsed_s": 0.0,
        "error": None,
    }

    repo_context = _setup_repo_for_canary(instance)
    record["repo_context_status"] = repo_context.get("repo_context_status")
    repo_dir = repo_context.get("repo_dir")
    tmp_root = repo_context.get("tmp_root")
    try:
        if repo_context.get("repo_context_status") != "ready" or not repo_dir:
            record["error"] = "repo_unavailable"
            record["failure_class"] = "EMPTY_PATCH"
            return record

        try:
            llm, provider, model = (
                llm_factory or (lambda: _build_repair_llm(tier=arm_a_tier))
            )()
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"llm_unavailable: {exc}"
            return record
        record["provider"] = provider
        record["model"] = model
        if (provider in set(provider_denylist)) or (
            provider_allowlist and provider not in set(provider_allowlist)
        ):
            record["error"] = "provider_blocked_by_policy"
            return record

        from sage.llm.base import Message, Role

        async def _call(text: str) -> Any:
            return await asyncio.wait_for(
                llm.generate(
                    messages=[Message(role=Role.USER, content=text)]
                ),
                timeout=call_timeout_s,
            )

        problem = str(instance.get("problem_statement") or "")

        async def _flow() -> tuple[str, dict[str, Any]]:
            """ALL awaits on this llm share ONE loop: the provider's
            underlying client binds connections to the loop of its first
            call — a second asyncio.run() raised 'Event loop is closed'
            (arm-A v2 first attempt, 10/10)."""
            # Agentless-lite call 1: localization over the file tree (the
            # single-call v1 was trivially empty: the patch_focused
            # prompt presumes repo access a bare LLM does not have).
            tree = _repo_file_tree(repo_dir)
            loc_response = await _call(_LOCALIZE_PROMPT.format(
                max_files=6, problem=problem[:6000], tree=tree
            ))
            selected = _parse_file_list(
                getattr(loc_response, "content", None) or "", repo_dir
            )
            record["localized_files"] = selected
            usage1 = getattr(loc_response, "usage", None)

            # Call 2: strict-diff emission over the selected contents.
            files_block = _files_block(repo_dir, selected)
            response = await _call(_PATCH_PROMPT.format(
                problem=problem[:6000], files_block=files_block
            ))
            usage2 = getattr(response, "usage", None)
            merged: dict[str, Any] = {}
            for u in (usage1, usage2):
                if isinstance(u, dict):
                    for k, v in u.items():
                        if isinstance(v, (int, float)):
                            merged[k] = merged.get(k, 0) + v
            record["usage"] = merged or None

            content = getattr(response, "content", None) or ""
            inner_patch = _swebench_extract_patch(content)
            record["patch_non_empty"] = bool(inner_patch)

            inner_annotation = _annotate_diff_verifier(
                patch=inner_patch, repo_dir=repo_dir, mode=verifier_mode
            )
            if (
                verifier_mode == "repair"
                and inner_patch
                and inner_annotation["_diff_verifier_outcome"]
                not in {
                    "skipped_no_patch",
                    "skipped_no_repo_dir",
                    "skipped_mode_off",
                    "unsupported_no_opinion",
                }
            ):
                inner_patch, repair_meta, inner_annotation = (
                    await _repair_patch_with_feedback(
                        patch=inner_patch,
                        repo_dir=repo_dir,
                        problem_statement=problem,
                        instance_id=instance_id,
                        repair_budget_usd=repair_budget_usd,
                        llm_factory=lambda: (llm, provider, model),
                        provider_allowlist=provider_allowlist,
                        provider_denylist=provider_denylist,
                        repair_timeout_s=repair_timeout_s,
                    )
                )
                record["_verifier_repair_stage"] = repair_meta[
                    "_verifier_repair_stage"
                ]
                record["_diff_verifier_outcome_pre_repair"] = repair_meta[
                    "_diff_verifier_outcome_pre_repair"
                ]
            return inner_patch, inner_annotation

        try:
            patch, annotation = asyncio.run(_flow())
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"llm_call_failed: {type(exc).__name__}: {exc}"
            return record
        record["_diff_verifier_outcome"] = annotation["_diff_verifier_outcome"]

        apply_ok, apply_detail = _git_apply_check(patch, repo_dir)
        record["apply_ok"] = apply_ok
        record["apply_detail"] = apply_detail
        record["failure_class"] = _failure_class(
            patch=patch,
            verifier_outcome=annotation["_diff_verifier_outcome"],
            apply_ok=apply_ok,
        )
        record["patch"] = patch
        return record
    finally:
        record["elapsed_s"] = round(time.monotonic() - started, 1)
        if tmp_root is not None:
            _cleanup_repo_dir(repo_dir, tmp_root=tmp_root)


# ── Arm D wrapper: reuse the canary task, then uniform apply-check ──────────


async def run_arm_d_task(
    instance: dict[str, Any],
    output_dir: Path,
    *,
    budget_usd: float,
    tier: str,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
    prompt_profile: str,
    verifier_mode: str,
    repair_budget_usd: float,
    repair_tier: str,
    repair_timeout_s: float,
    fmt_module: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    result = await _arm_d_run_one_task(
        instance,
        output_dir,
        mock=False,
        budget_usd=budget_usd,
        tier=tier,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
        prefix="mini-2b-arm-d",
        fmt_module=fmt_module,
        claim_default_pipeline_learning_evidence=False,
        expect_default_pipeline_learn=False,
        swebench_prompt_profile=prompt_profile,
        verifier_mode=verifier_mode,
        repair_budget_usd=repair_budget_usd,
        repair_tier=repair_tier,
        repair_timeout_s=repair_timeout_s,
    )
    summary = result["summary"]
    patch = str(result["record"].get("patch") or "")

    # Uniform post-hoc applicability: fresh worktree (the canary cleaned
    # its own), git apply --check, cleanup.
    apply_ok, apply_detail = False, "repo_unavailable_for_check"
    repo_context = _setup_repo_for_canary(instance)
    repo_dir = repo_context.get("repo_dir")
    tmp_root = repo_context.get("tmp_root")
    try:
        if repo_context.get("repo_context_status") == "ready" and repo_dir:
            apply_ok, apply_detail = _git_apply_check(patch, repo_dir)
    finally:
        if tmp_root is not None:
            _cleanup_repo_dir(repo_dir, tmp_root=tmp_root)

    return {
        "instance_id": instance["instance_id"],
        "arm": "D",
        "patch_non_empty": bool(patch),
        "apply_ok": apply_ok,
        "apply_detail": apply_detail,
        "failure_class": _failure_class(
            patch=patch,
            verifier_outcome=summary.get("_diff_verifier_outcome"),
            apply_ok=apply_ok,
        ),
        "provider": summary.get("provider_final"),
        "model": summary.get("model_id_final"),
        "cost_usd": summary.get("total_cost_usd"),
        "cost_source": summary.get("_total_cost_usd_source"),
        "_diff_verifier_outcome": summary.get("_diff_verifier_outcome"),
        "_diff_verifier_outcome_pre_repair": summary.get(
            "_diff_verifier_outcome_pre_repair"
        ),
        "_verifier_repair_stage": summary.get("_verifier_repair_stage"),
        "repo_context_status": (summary.get("repo_context") or {}).get(
            "status"
        ),
        "elapsed_s": round(time.monotonic() - started, 1),
        "patch": patch,
    }


# ── Driver ──────────────────────────────────────────────────────────────────


def _arm_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    return {
        "n": n,
        "patch_non_empty": sum(1 for r in records if r.get("patch_non_empty")),
        "apply_ok": sum(1 for r in records if r.get("apply_ok")),
        "verifier_clean": sum(
            1 for r in records if r.get("_diff_verifier_outcome") == "clean"
        ),
        "plausible": sum(
            1 for r in records if r.get("failure_class") == "PLAUSIBLE_PATCH"
        ),
        "failure_classes": {
            cls: sum(1 for r in records if r.get("failure_class") == cls)
            for cls in (
                "EMPTY_PATCH",
                "NOT_UNIFIED_DIFF",
                "COUNT_MISMATCH",
                "CONTENT_MISMATCH",
                "APPLY_FAILED",
                "PLAUSIBLE_PATCH",
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier", default="reasoner",
                        help="Arm D pipeline tier")
    parser.add_argument("--arm-a-tier", default="reasoner",
                        help="Arm A single-call tier")
    parser.add_argument("--budget-usd", type=float, default=2.0,
                        help="Arm D per-task budget")
    parser.add_argument("--global-budget-usd", type=float, default=8.0)
    parser.add_argument("--task-timeout-s", type=float, default=900.0)
    parser.add_argument("--provider-allowlist", default="google,deepseek")
    parser.add_argument("--provider-denylist", default="openai")
    parser.add_argument("--swebench-prompt-profile", default="patch_focused")
    parser.add_argument("--verifier-mode", default="repair",
                        choices=["observe", "repair"])
    parser.add_argument("--repair-budget-usd", type=float, default=0.50)
    parser.add_argument("--repair-tier", default="reasoner")
    parser.add_argument("--repair-timeout-s", type=float, default=180.0)
    parser.add_argument("--arms", default="AD", choices=["A", "D", "AD"],
                        help="Arms to run; the other arm's records merge "
                             "from existing per_task pair files.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # 2026-06-11 invalid first run: arm A's IN-PROCESS LLM calls died with
    # SSL CERTIFICATE_VERIFY_FAILED on the corporate MITM CA (Python 3.13
    # strict). The SAGE subprocess is immune via boot.py's truststore
    # injection; launcher-side callers must inject too. No-op on clean
    # networks.
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001
        pass

    from run_dryrun_arm_d import _load_format_patch_module
    from sage.bench.keep_awake import prevent_os_sleep

    allowlist = tuple(
        s.strip() for s in args.provider_allowlist.split(",") if s.strip()
    )
    denylist = tuple(
        s.strip() for s in args.provider_denylist.split(",") if s.strip()
    )
    instances = json.loads(args.instances_json.read_text(encoding="utf-8"))
    instances = instances[: args.limit]
    fmt_module = _load_format_patch_module()
    out = args.output_dir
    (out / "per_task").mkdir(parents=True, exist_ok=True)

    a_records: list[dict[str, Any]] = []
    d_records: list[dict[str, Any]] = []
    run_a = "A" in args.arms
    run_d = "D" in args.arms
    with prevent_os_sleep():
        for index, instance in enumerate(instances):
            iid = instance["instance_id"]
            pair_path = out / "per_task" / f"{index:02d}_{iid[:60]}.json"
            existing: dict[str, Any] = {}
            if pair_path.exists():
                try:
                    existing = json.loads(
                        pair_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing = {}
            a_rec = existing.get("arm_a")
            d_rec = existing.get("arm_d")

            if run_a:
                log.info("[%d/%d] %s — Arm A (Agentless-lite 2-call)...",
                         index + 1, len(instances), iid)
                a_rec = run_arm_a_task(
                    instance,
                    arm_a_tier=args.arm_a_tier,
                    verifier_mode=args.verifier_mode,
                    repair_budget_usd=args.repair_budget_usd,
                    repair_timeout_s=args.repair_timeout_s,
                    prompt_profile=args.swebench_prompt_profile,
                    provider_allowlist=allowlist,
                    provider_denylist=denylist,
                )
                log.info("[%d/%d] %s — Arm A: %s (apply=%s)",
                         index + 1, len(instances), iid,
                         a_rec["failure_class"], a_rec["apply_ok"])
            if a_rec:
                a_records.append(a_rec)

            if run_d:
                log.info("[%d/%d] %s — Arm D (full pipeline)...",
                         index + 1, len(instances), iid)
                try:
                    d_rec = asyncio.run(
                        asyncio.wait_for(
                            run_arm_d_task(
                                instance,
                                out,
                                budget_usd=args.budget_usd,
                                tier=args.tier,
                                provider_allowlist=allowlist,
                                provider_denylist=denylist,
                                prompt_profile=args.swebench_prompt_profile,
                                verifier_mode=args.verifier_mode,
                                repair_budget_usd=args.repair_budget_usd,
                                repair_tier=args.repair_tier,
                                repair_timeout_s=args.repair_timeout_s,
                                fmt_module=fmt_module,
                            ),
                            timeout=args.task_timeout_s,
                        )
                    )
                except asyncio.TimeoutError:
                    d_rec = {
                        "instance_id": iid,
                        "arm": "D",
                        "patch_non_empty": False,
                        "apply_ok": False,
                        "apply_detail": "task_timeout",
                        "failure_class": "EMPTY_PATCH",
                        "error": f"task_timeout_{args.task_timeout_s:.0f}s",
                    }
                log.info("[%d/%d] %s — Arm D: %s (apply=%s)",
                         index + 1, len(instances), iid,
                         d_rec["failure_class"], d_rec.get("apply_ok"))
            if d_rec:
                d_records.append(d_rec)

            pair_path.write_text(
                json.dumps({"arm_a": a_rec, "arm_d": d_rec}, indent=1,
                           ensure_ascii=False),
                encoding="utf-8",
                newline="\n",
            )

    summary = {
        "block": "MINI_2B_UNGRADED_A_VS_D_APPLICABILITY_AND_FAILURE_CLASS",
        "instances": [i["instance_id"] for i in instances],
        "arm_a": _arm_aggregate(a_records),
        "arm_d": _arm_aggregate(d_records),
        "decision_rule": (
            "D <= A on non-empty+applyability+verifier-clean -> product "
            "diagnosis (C); D > A clearly -> GO graded 2.b"
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    a, d = summary["arm_a"], summary["arm_d"]
    log.info(
        "DONE. Arm A: non_empty=%d/%d apply=%d clean=%d plausible=%d | "
        "Arm D: non_empty=%d/%d apply=%d clean=%d plausible=%d",
        a["patch_non_empty"], a["n"], a["apply_ok"], a["verifier_clean"],
        a["plausible"],
        d["patch_non_empty"], d["n"], d["apply_ok"], d["verifier_clean"],
        d["plausible"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
