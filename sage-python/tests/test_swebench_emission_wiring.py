"""TDD tests for Track 2 task 2.4 — wire SEARCH/REPLACE emission into
``generate_patches``.

Covers the three responsibilities of the wiring:

1. ``SAGE_EMISSION_FORMAT`` controls which prompt template feeds the
   system-prompt / TaskInput.instructions path. The bench itself does
   not render the template directly (the input adapter does), so this
   is verified at the ``normalize_swebench`` boundary.
2. ``generate_patches`` falls back from the unified-diff extractor to
   the SEARCH/REPLACE extractor when the env var opts in AND the
   unified extractor returned empty. Critically — the unified
   extractor runs FIRST in both modes, so a model that ignores the
   SEARCH/REPLACE prompt and still emits a ```diff fence keeps
   working.
3. Every prediction dict gets an ``_extraction_method`` metadata field
   with one of the values ``"unified"`` / ``"search-replace-exact"``
   / ``"search-replace-fuzzy"`` / ``"search-replace-missing"`` /
   ``"empty"``.

Non-goals: no flip of the default (that's T2.5), no real LLM or Docker
calls — every LLM response is stubbed via a ``_FakeSystem``.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import sage.bench.swebench_bench as swebench_mod
from sage.bench.swebench_bench import SWEBenchBench


# ---------------------------------------------------------------------------
# Fixture helpers — mirror test_search_replace_extraction.py so the T2.5
# smoke can run the same shape of checks against real LLMs later.
# ---------------------------------------------------------------------------


_GIT_AVAILABLE = shutil.which("git") is not None


def _init_git_repo(repo_dir: Path) -> None:
    """Initialise a git repo at ``repo_dir`` and commit everything present."""
    import os as _os

    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": _os.environ.get("PATH", ""),
    }
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=repo_dir, check=True, env=env,
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_dir, check=True, env=env,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"],
        cwd=repo_dir, check=True, env=env,
    )


CANONICAL_INSTANCE = {
    "instance_id": "demo__repo-1",
    "repo": "demo/repo",
    "version": "1.0",
    "base_commit": "abc123",
    "problem_statement": "Fix the failing parser edge case.",
    "hints_text": "",
}


class _FakeSystem:
    """Stub that injects a canned response string into the agent pipeline.

    ``generate_patches`` unpacks several attributes from ``self.system``:
    - ``agent_loop._llm`` for model_id
    - ``agent_loop.total_cost_usd`` for cost
    - ``pipeline.last_context`` (set by ``run``) for tool metadata
    - ``_last_execution_path`` (set by ``run``) for execution_path

    ``run`` returns the canned LLM response string verbatim.
    """

    def __init__(self, canned_response: str) -> None:
        self._canned_response = canned_response
        self.agent_loop = SimpleNamespace(
            _llm=SimpleNamespace(model_id="fake-model"),
            total_cost_usd=0.0,
        )
        self.pipeline = SimpleNamespace(last_context=None)
        self._last_execution_path = ""

    async def run(self, task, *, system_hint=None) -> str:  # noqa: ANN001
        self.pipeline.last_context = SimpleNamespace(
            system=3,
            tool_call_count=0,
            tool_turn_count=0,
            executed_commands=[],
        )
        self._last_execution_path = "pipeline"
        return self._canned_response


def _stub_dataset(monkeypatch, instance: dict | None = None) -> dict:
    """Inject a single-instance dataset into the bench loader."""
    inst = instance or CANONICAL_INSTANCE
    monkeypatch.setattr(
        swebench_mod,
        "load_swebench_dataset",
        lambda *args, **kwargs: [inst],
    )
    return inst


def _stub_no_repair(monkeypatch) -> None:
    """Patch-repair loop runs a live LLM call by default — short-circuit
    it so the tests never touch the network and keep
    ``_extraction_method`` as the single attribution signal."""
    async def _identity(patch, repo_dir, llm, problem_statement, instance_id, llm_timeout):  # noqa: ARG001
        return patch, "unchanged"

    import sage.bench.swebench_patch_repair as repair_mod
    monkeypatch.setattr(repair_mod, "try_repair_patch", _identity)


# ---------------------------------------------------------------------------
# 1. Unified mode — graceful default path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_patches_unified_mode_passthrough(monkeypatch):
    """Default env (unset = "unified"): a ```diff fence in the model
    response produces a non-empty patch with
    ``_extraction_method == "unified"``."""
    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    canned = (
        "Here is the patch:\n"
        "```diff\n"
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: None)

    preds = await bench.generate_patches(limit=1)
    assert len(preds) == 1
    assert preds[0]["model_patch"].startswith("diff --git")
    assert preds[0]["_extraction_method"] == "unified"


# ---------------------------------------------------------------------------
# 2. Advisor-flagged graceful degradation — search-replace mode, model
#    ignores the prompt format and still emits a diff fence.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_patches_search_replace_mode_diff_fence_still_works(monkeypatch):
    """Under ``SAGE_EMISSION_FORMAT=search-replace``, a well-formed
    ```diff fence response still produces a valid patch. The wiring
    prefers "the cleanest patch we got" over "match the requested
    format" — step 7 of the Mandatory Workflow still names the diff
    fence, so the model may follow that path even under SR mode."""
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    canned = (
        "Even though the prompt asked for SEARCH/REPLACE, here is a diff:\n"
        "```diff\n"
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: None)

    preds = await bench.generate_patches(limit=1)
    assert len(preds) == 1
    assert preds[0]["model_patch"].startswith("diff --git")
    assert preds[0]["_extraction_method"] == "unified", (
        "unified extractor must win when it returns a non-empty patch, "
        "even under SEARCH/REPLACE mode — the model's choice of format "
        "overrides the prompt's preference."
    )


# ---------------------------------------------------------------------------
# 3. Search-replace mode — blocks-only response extracted end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
@pytest.mark.asyncio
async def test_generate_patches_search_replace_mode_blocks_extracted(
    monkeypatch, tmp_path
):
    """Under ``SAGE_EMISSION_FORMAT=search-replace``, a pure
    SEARCH/REPLACE response (no diff fence) against a real on-disk
    repo produces a patch with
    ``_extraction_method == "search-replace-exact"``."""
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    # Build a tiny repo with a known file the LLM will patch.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "pkg").mkdir()
    (repo_dir / "pkg" / "mod.py").write_text(
        "def foo(x):\n    return x\n", encoding="utf-8"
    )
    _init_git_repo(repo_dir)

    canned = (
        "Here is the fix.\n"
        "\n"
        "## File: pkg/mod.py\n"
        "<<<<<<< SEARCH\n"
        "def foo(x):\n"
        "    return x\n"
        "=======\n"
        "def foo(x):\n"
        "    return x + 1\n"
        ">>>>>>> REPLACE\n"
    )

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    # Return the pre-built repo so generate_patches chdir's into it and
    # the SR extractor can locate pkg/mod.py.
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: str(repo_dir))

    preds = await bench.generate_patches(limit=1)
    assert len(preds) == 1
    assert preds[0]["_extraction_method"] == "search-replace-exact"
    assert preds[0]["model_patch"], "SR extractor must produce a non-empty diff"
    assert "diff --git" in preds[0]["model_patch"]


# ---------------------------------------------------------------------------
# 4. Empty response — both modes produce the same empty record.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_patches_records_extraction_empty_in_unified_mode(monkeypatch):
    """Pure-prose response under unified mode -> empty patch, method empty."""
    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    canned = "I need to explore more files before proposing a patch."

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: None)

    preds = await bench.generate_patches(limit=1)
    assert preds[0]["model_patch"] == ""
    assert preds[0]["_extraction_method"] == "empty"


@pytest.mark.asyncio
async def test_generate_patches_records_extraction_empty_in_search_replace_mode(
    monkeypatch,
):
    """Pure-prose response under SR mode -> empty patch, method empty."""
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    canned = "I need to explore more files before proposing a patch."

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: None)

    preds = await bench.generate_patches(limit=1)
    assert preds[0]["model_patch"] == ""
    assert preds[0]["_extraction_method"] == "empty"


# ---------------------------------------------------------------------------
# 5. Path normalization — Windows-style backslash paths are normalised
#    before the SR extractor is called.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
@pytest.mark.asyncio
async def test_generate_patches_normalizes_backslash_paths(monkeypatch, tmp_path):
    """Spec reviewer flag: LLMs sometimes emit Windows-style
    ``pkg\\mod.py`` even under a forward-slash prompt. The wiring must
    normalise those to ``pkg/mod.py`` before handing blocks to
    ``_blocks_to_unified_diff``, so the diff still applies cleanly.
    """
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "pkg").mkdir()
    (repo_dir / "pkg" / "mod.py").write_text(
        "value = 1\n", encoding="utf-8"
    )
    _init_git_repo(repo_dir)

    # Marker uses a backslash, as a careless LLM on a Windows host might.
    canned = (
        "## File: pkg\\mod.py\n"
        "<<<<<<< SEARCH\n"
        "value = 1\n"
        "=======\n"
        "value = 2\n"
        ">>>>>>> REPLACE\n"
    )

    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _inst: str(repo_dir))

    preds = await bench.generate_patches(limit=1)
    patch = preds[0]["model_patch"]
    assert preds[0]["_extraction_method"] == "search-replace-exact"
    # The emitted diff must name pkg/mod.py with a forward slash, NOT
    # pkg\mod.py — this is what makes git apply --check accept it.
    assert "a/pkg/mod.py" in patch
    assert "pkg\\mod.py" not in patch


# ---------------------------------------------------------------------------
# 6. Prompt-selection wiring — normalize_swebench follows the env var.
# ---------------------------------------------------------------------------


def test_normalize_swebench_instructions_follow_emission_format(monkeypatch):
    """After T2.4, ``normalize_swebench`` picks its ``instructions``
    field via ``get_swebench_template()``, so setting
    ``SAGE_EMISSION_FORMAT=search-replace`` changes what flows into the
    TaskInput. This is the assertion that proves the migration of lines
    308 / 331 in ``sage/input/swebench.py`` from the hardcoded
    ``SWEBENCH_SYSTEM_TEMPLATE`` to the dispatcher actually landed.
    """
    from sage.input import (
        SWEBENCH_SYSTEM_TEMPLATE,
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
        normalize_swebench,
    )

    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    assert normalize_swebench(CANONICAL_INSTANCE).instructions == SWEBENCH_SYSTEM_TEMPLATE

    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    assert (
        normalize_swebench(CANONICAL_INSTANCE).instructions
        == SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE
    )


def test_render_swebench_prompt_follows_emission_format(monkeypatch):
    """Same migration, other side — ``render_swebench_prompt`` picks
    its template via the dispatcher so the rendered prompt text
    carries the SR patch-format section when opted in."""
    from sage.input import normalize_swebench, render_swebench_prompt

    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    unified_prompt = render_swebench_prompt(normalize_swebench(CANONICAL_INSTANCE))
    assert "## Patch Format — Strict" in unified_prompt
    assert "<<<<<<< SEARCH" not in unified_prompt

    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    sr_prompt = render_swebench_prompt(normalize_swebench(CANONICAL_INSTANCE))
    assert "<<<<<<< SEARCH" in sr_prompt
    assert "## Patch Format — Strict\n\nOutput your final patch in unified diff" not in sr_prompt


# ---------------------------------------------------------------------------
# 7. F3 — ``write_predictions`` persists ``_extraction_method`` into the
#    on-disk predictions.jsonl so per-bucket analysis can read it
#    directly instead of grepping agent logs. Finding #4 of the
#    2026-04-23 emission-format smoke motivated this fix.
# ---------------------------------------------------------------------------


def test_write_predictions_persists_extraction_method(tmp_path):
    """The ``_extraction_method`` metadata set by ``generate_patches``
    must survive ``write_predictions`` so that downstream bucket
    analysis can parse the jsonl directly, not grep the agent log."""
    import json

    bench = SWEBenchBench(system=_FakeSystem(""), dataset="lite")
    predictions = [
        {
            "instance_id": "demo__repo-1",
            "model_name_or_path": "sage/fake-model",
            "model_patch": "diff --git a/x b/x\n",
            "_extraction_method": "search-replace-exact",
        }
    ]
    out_path = tmp_path / "p.jsonl"
    bench.write_predictions(predictions, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    # Standard SWE-bench harness keys still present and unchanged.
    assert entry["instance_id"] == "demo__repo-1"
    assert entry["model_name_or_path"] == "sage/fake-model"
    assert entry["model_patch"] == "diff --git a/x b/x\n"
    # The SAGE-specific metadata now rides along.
    assert entry["_extraction_method"] == "search-replace-exact"


def test_write_predictions_omits_extraction_method_when_absent(tmp_path):
    """Negative case — an older/legacy caller that doesn't annotate the
    prediction dict must still produce a valid 3-key record. No
    KeyError, no ``_extraction_method: None`` leaking through."""
    import json

    bench = SWEBenchBench(system=_FakeSystem(""), dataset="lite")
    predictions = [
        {
            "instance_id": "demo__repo-1",
            "model_name_or_path": "sage/fake-model",
            "model_patch": "",
        }
    ]
    out_path = tmp_path / "p.jsonl"
    bench.write_predictions(predictions, out_path)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry == {
        "instance_id": "demo__repo-1",
        "model_name_or_path": "sage/fake-model",
        "model_patch": "",
    }
    assert "_extraction_method" not in entry
