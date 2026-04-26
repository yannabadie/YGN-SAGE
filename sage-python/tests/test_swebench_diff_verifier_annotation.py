"""Prediction annotation tests for diff-verifier reason telemetry."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sage.bench.swebench_bench as swebench_mod
from sage.bench.swebench_bench import SWEBenchBench


CANONICAL_INSTANCE = {
    "instance_id": "demo__repo-1",
    "repo": "demo/repo",
    "version": "1.0",
    "base_commit": "abc123",
    "problem_statement": "Fix the failing parser edge case.",
    "hints_text": "",
}


class _FakeSystem:
    def __init__(self, canned_response: str) -> None:
        self._canned_response = canned_response
        self.agent_loop = SimpleNamespace(
            _llm=SimpleNamespace(model_id="fake-model"),
            total_cost_usd=0.0,
        )
        self.pipeline = SimpleNamespace(last_context=None)
        self._last_execution_path = ""

    async def run(self, task, *, system_hint=None) -> str:  # noqa: ANN001, ARG002
        self.pipeline.last_context = SimpleNamespace(
            system=3,
            tool_call_count=0,
            tool_turn_count=0,
            executed_commands=[],
        )
        self._last_execution_path = "pipeline"
        return self._canned_response


def _stub_dataset(
    monkeypatch: pytest.MonkeyPatch,
    instance: dict | None = None,
) -> dict:
    inst = instance or CANONICAL_INSTANCE
    monkeypatch.setattr(
        swebench_mod,
        "load_swebench_dataset",
        lambda *args, **kwargs: [inst],
    )
    return inst


def _stub_no_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _identity(
        patch,  # noqa: ANN001
        repo_dir,  # noqa: ANN001, ARG001
        llm,  # noqa: ANN001, ARG001
        problem_statement,  # noqa: ANN001, ARG001
        instance_id,  # noqa: ANN001, ARG001
        llm_timeout,  # noqa: ANN001, ARG001
    ):
        return patch, "unchanged"

    import sage.bench.swebench_patch_repair as repair_mod

    monkeypatch.setattr(repair_mod, "try_repair_patch", _identity)


def test_write_predictions_preserves_diff_verifier_reasons_and_outcome(
    tmp_path: Path,
) -> None:
    bench = SWEBenchBench(system=None, dataset="lite")
    predictions = [
        {
            "instance_id": "demo__repo-1",
            "model_name_or_path": "sage/fake-model",
            "model_patch": "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n",
            "_diff_verifier_mismatches": [
                {
                    "file": "pkg/mod.py",
                    "hunk_index": 0,
                    "old_start": 1,
                    "old_count": 2,
                    "kind": "content_mismatch",
                    "match_ratio": 0.5,
                }
            ],
            "_diff_verifier_reasons": ["content_mismatch"],
            "_diff_verifier_outcome": "content_mismatch",
        }
    ]

    out_path = bench.write_predictions(predictions, tmp_path / "p.jsonl")
    entry = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])

    assert entry["_diff_verifier_mismatches"] == predictions[0][
        "_diff_verifier_mismatches"
    ]
    assert entry["_diff_verifier_reasons"] == ["content_mismatch"]
    assert entry["_diff_verifier_outcome"] == "content_mismatch"


@pytest.mark.asyncio
async def test_generate_patches_annotates_diff_verifier_reasons_in_observe_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    monkeypatch.setenv("SAGE_DIFF_VERIFIER_MODE", "observe")
    _stub_dataset(monkeypatch)
    _stub_no_repair(monkeypatch)

    repo_dir = tmp_path / "repo"
    (repo_dir / "pkg").mkdir(parents=True)
    (repo_dir / "pkg" / "mod.py").write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(
        SWEBenchBench,
        "_setup_repo",
        staticmethod(lambda _inst: str(repo_dir)),
    )

    canned = (
        "```diff\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    bench = SWEBenchBench(system=_FakeSystem(canned), dataset="lite")

    predictions = await bench.generate_patches(limit=1, out_dir=tmp_path)
    prediction = predictions[0]

    assert prediction["_diff_verifier_mismatches"] == []
    assert prediction["_diff_verifier_reasons"] == ["clean"]
    assert prediction["_diff_verifier_outcome"] == "clean"
