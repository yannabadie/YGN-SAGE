"""MINI_2B_UNGRADED_A_VS_D — paired applicability & failure-class bench.

cgpro NEXT_BLOCK (2026-06-11, conv cgpro_b2_unblockers_verify): N=10
paired, same instances both arms, NO official grading. Metrics:
patch_non_empty, git-apply check, verifier outcome pre/post repair,
provider/cost audit, failure class. Decision rule: D <= A on
non-empty+applyability+verifier-clean -> product diagnosis (C);
D > A clearly -> GO graded 2.b.

Arm A here is the CONTROLLED single-call baseline: one reasoner-tier
LLM call with the IDENTICAL patch_focused prompt, worktree and
verifier/repair chain as arm D — orchestration is the only variable
(the cycle-13 wiring doc's 'pure-orchestration delta'). The original
doc's Claude-Code arm A is a different (product-competitiveness)
question, out of scope for the mini.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sage-python" / "scripts"))

import run_mini_ab as mini  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "mod.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "init")
    return repo


GOOD_PATCH = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,3 +1,3 @@\n"
    " a = 1\n"
    "-b = 2\n"
    "+b = 5\n"
    " c = 3\n"
)

BAD_CONTEXT_PATCH = (
    "--- a/mod.py\n"
    "+++ b/mod.py\n"
    "@@ -1,3 +1,3 @@\n"
    " x = 9\n"
    "-y = 8\n"
    "+y = 7\n"
    " z = 6\n"
)


def test_git_apply_check_applies(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    ok, detail = mini._git_apply_check(GOOD_PATCH, str(repo))
    assert ok is True
    assert detail == "applies"


def test_git_apply_check_rejects_bad_context(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    ok, detail = mini._git_apply_check(BAD_CONTEXT_PATCH, str(repo))
    assert ok is False
    assert "error" in detail.lower() or "patch" in detail.lower()


def test_git_apply_check_empty_patch(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    ok, detail = mini._git_apply_check("", str(repo))
    assert ok is False
    assert detail == "empty_patch"


def test_failure_class_taxonomy() -> None:
    """cgpro contract classes: EMPTY_PATCH / NOT_UNIFIED_DIFF /
    COUNT_MISMATCH / CONTENT_MISMATCH / APPLY_FAILED / PLAUSIBLE_PATCH."""
    fc = mini._failure_class
    assert fc(patch="", verifier_outcome="skipped_no_patch",
              apply_ok=False) == "EMPTY_PATCH"
    assert fc(patch="x", verifier_outcome="not_unified_diff",
              apply_ok=False) == "NOT_UNIFIED_DIFF"
    assert fc(patch="x", verifier_outcome="hunk_body_count_mismatch",
              apply_ok=False) == "COUNT_MISMATCH"
    assert fc(patch="x", verifier_outcome="malformed_hunk_header",
              apply_ok=False) == "COUNT_MISMATCH"
    assert fc(patch="x", verifier_outcome="content_mismatch",
              apply_ok=False) == "CONTENT_MISMATCH"
    assert fc(patch="x", verifier_outcome="fuzzy_below_threshold",
              apply_ok=False) == "CONTENT_MISMATCH"
    # verifier says clean but git apply still refuses -> APPLY_FAILED
    assert fc(patch="x", verifier_outcome="clean",
              apply_ok=False) == "APPLY_FAILED"
    assert fc(patch="x", verifier_outcome="clean",
              apply_ok=True) == "PLAUSIBLE_PATCH"
    # unsupported_no_opinion + applies -> still plausible (apply is the
    # decisive cheap signal when the verifier abstains)
    assert fc(patch="x", verifier_outcome="unsupported_no_opinion",
              apply_ok=True) == "PLAUSIBLE_PATCH"
    assert fc(patch="x", verifier_outcome="unsupported_no_opinion",
              apply_ok=False) == "APPLY_FAILED"


def test_arm_a_two_call_task(monkeypatch, tmp_path) -> None:
    """Arm A (Agentless-lite): call 1 localizes files from the tree,
    call 2 emits the diff over their contents; usage summed; same
    verifier chain and apply-check as D."""

    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        async def generate(self, messages, **kwargs):
            self.calls += 1

            class _R:
                usage = {"input_tokens": 500, "output_tokens": 100}

            if self.calls == 1:
                _R.content = "mod.py"
            else:
                _R.content = "```diff\n" + GOOD_PATCH + "```"
            return _R()

    llm = _FakeLLM()
    repo = _make_repo(tmp_path)
    monkeypatch.setattr(
        mini, "_setup_repo_for_canary",
        lambda inst: {
            "repo_context_status": "ready",
            "repo_dir": str(repo),
            "tmp_root": None,
            "repo_url": "stub",
            "base_commit": "stub",
            "checkout_sha": "stub",
            "clone_elapsed_ms": 1,
            "fetch_fallback_used": False,
            "failure_reason": None,
        },
    )
    monkeypatch.setattr(mini, "_cleanup_repo_dir",
                        lambda repo_dir, *, tmp_root=None: "skipped")

    instance = {
        "instance_id": "t-armA",
        "repo": "x/y",
        "base_commit": "deadbeef",
        "problem_statement": "change b to 5",
    }
    result = mini.run_arm_a_task(
        instance,
        llm_factory=lambda: (llm, "google", "gemini-test"),
        verifier_mode="repair",
        repair_budget_usd=0.5,
        repair_timeout_s=5.0,
        prompt_profile="patch_focused",
    )
    assert llm.calls == 2
    assert result["localized_files"] == ["mod.py"]
    assert result["patch_non_empty"] is True
    assert result["apply_ok"] is True
    assert result["failure_class"] == "PLAUSIBLE_PATCH"
    assert result["provider"] == "google"
    assert result["usage"] == {"input_tokens": 1000, "output_tokens": 200}
    assert result["_diff_verifier_outcome"] == "clean"
