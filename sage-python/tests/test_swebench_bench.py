from __future__ import annotations

from types import SimpleNamespace

import pytest

import sage.bench.swebench_bench as swebench_mod
from sage.bench.swebench_bench import (
    SWEBenchBench,
    _DATASET_MAP,
    _classify_prediction,
    _SENTINEL_MARKER,
)


def test_pro_dataset_wired_to_scaleai():
    # P0.2 wiring guard: SWE-bench Pro must remain reachable so Sprint 6
    # can baseline against it. Mapping mirrors the HF repo at
    # github.com/scaleapi/SWE-bench_Pro-os.
    assert "pro" in _DATASET_MAP
    assert _DATASET_MAP["pro"] == "ScaleAI/SWE-bench_Pro"


def test_legacy_dataset_keys_intact():
    # Regression guard: do not break --dataset lite/verified/full.
    assert _DATASET_MAP["lite"] == "princeton-nlp/SWE-bench_Lite"
    assert _DATASET_MAP["verified"] == "princeton-nlp/SWE-bench_Verified"
    assert _DATASET_MAP["full"] == "princeton-nlp/SWE-bench"


def test_classify_prediction_empty():
    assert _classify_prediction(None) == "empty"
    assert _classify_prediction("") == "empty"


def test_classify_prediction_sentinel():
    # Exact string agent_loop emits (phases/learn.py). Stays in sync via marker constant.
    assert _classify_prediction("[sage: agent exited after 5 steps with no content]") == "sentinel"
    assert _classify_prediction("[sage: agent exited after 17 steps with no content]") == "sentinel"


def test_classify_prediction_real_diff():
    patch = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert _classify_prediction(patch) == "real"


def test_sentinel_marker_matches_learn_phase():
    # Guard against silent divergence — phases/learn.py's EMPTY_STEP_SENTINEL
    # must start with the same prefix that bench's _SENTINEL_MARKER looks for.
    from sage.phases.learn import EMPTY_STEP_SENTINEL
    # The formatted string (with step_count substituted) must contain the marker.
    sample = EMPTY_STEP_SENTINEL.format(step_count=42)
    assert _SENTINEL_MARKER in sample
    assert _classify_prediction(sample) == "sentinel"


def test_extract_patch_returns_empty_for_sentinel():
    """D7 audit (2026-04-18): sentinel text must NOT become a patch.
    Before fix: astropy-14995 emitted "[sage: agent exited after 20 steps
    with no content]\\n" as a 52-char patch. Docker eval rejects it but
    the jsonl counted it as a patch attempt."""
    from sage.bench.swebench_bench import _extract_patch
    sentinel = "[sage: agent exited after 20 steps with no content]"
    assert _extract_patch(sentinel) == ""
    # Sentinel embedded in longer output also strips to empty
    assert _extract_patch(f"some text\n{sentinel}\nmore") == ""


def test_classify_prediction_dict_form_reads_structured_failure():
    """D7 audit: classifier accepts the full prediction dict and reads
    `_structured_failure` to distinguish sentinel-emptied (step budget
    exhausted) from real-empty (generation error). Since _extract_patch
    now strips sentinel text, the patch alone can't distinguish — the
    auxiliary flag does."""
    # sentinel-emptied: patch is "" (stripped by _extract_patch) but
    # _structured_failure flag tells the classifier what happened
    sentinel_pred = {
        "instance_id": "x-1",
        "model_patch": "",
        "_structured_failure": "step_budget_exhausted",
    }
    assert _classify_prediction(sentinel_pred) == "sentinel"

    # real-empty: no flag, empty patch
    real_empty = {
        "instance_id": "x-2",
        "model_patch": "",
        "_structured_failure": "",
    }
    assert _classify_prediction(real_empty) == "empty"

    # real patch
    real = {
        "instance_id": "x-3",
        "model_patch": "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n",
        "_structured_failure": "",
    }
    assert _classify_prediction(real) == "real"

    # Backward compat: legacy jsonl still has sentinel text in the patch
    legacy = {
        "instance_id": "x-4",
        "model_patch": "[sage: agent exited after 20 steps with no content]",
        # No _structured_failure field in old files
    }
    assert _classify_prediction(legacy) == "sentinel"


@pytest.mark.asyncio
async def test_generate_patches_uses_pipeline_context_metadata(monkeypatch):
    instance = {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "version": "1.0",
        "base_commit": "abc123",
        "problem_statement": "Fix the failing parser edge case.",
        "hints_text": "",
    }

    monkeypatch.setattr(
        swebench_mod,
        "load_swebench_dataset",
        lambda *args, **kwargs: [instance],
    )

    class _FakeSystem:
        def __init__(self) -> None:
            self.agent_loop = SimpleNamespace(
                _llm=SimpleNamespace(model_id="fake-model"),
                total_cost_usd=0.125,
            )
            self.pipeline = SimpleNamespace(last_context=None)
            self._last_execution_path = ""

        async def run(self, task: str, *, system_hint: int | None = None) -> str:
            # SWEBenchBench forces S3 routing via system_hint=3 — verify the
            # plumbing reaches this layer so the router override is actually
            # in effect at inference time.
            assert system_hint == 3
            self.pipeline.last_context = SimpleNamespace(
                system=3,
                tool_call_count=2,
                tool_turn_count=1,
                executed_commands=["pwd", "git status"],
            )
            self._last_execution_path = "pipeline"
            return (
                "diff --git a/pkg/file.py b/pkg/file.py\n"
                "--- a/pkg/file.py\n"
                "+++ b/pkg/file.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )

    bench = SWEBenchBench(system=_FakeSystem(), event_bus=None, dataset="lite")
    monkeypatch.setattr(bench, "_setup_repo", lambda _instance: None)

    predictions = await bench.generate_patches(limit=1)

    assert len(predictions) == 1
    assert bench.manifest is not None

    pred = predictions[0]
    trace = bench.manifest.traces[0]

    assert pred["instance_id"] == instance["instance_id"]
    assert pred["_system_used"] == 3
    assert pred["_tool_call_count"] == 2
    assert pred["_tool_turn_count"] == 1
    assert pred["_executed_commands"] == ["pwd", "git status"]
    assert pred["_execution_path"] == "pipeline"
    assert pred["model_patch"].startswith("diff --git")

    assert trace.routing == "S3"
    assert trace.meta["execution_path"] == "pipeline"
    assert trace.meta["tool_call_count"] == 2
    assert trace.meta["tool_turn_count"] == 1
    assert trace.meta["executed_commands"] == ["pwd", "git status"]
