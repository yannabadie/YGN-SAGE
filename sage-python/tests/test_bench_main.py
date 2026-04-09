from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import sage.bench.__main__ as bench_main
import sage.bench.swebench_bench as swebench_mod


@pytest.mark.asyncio
async def test_run_swebench_generate_only_does_not_require_google_env(monkeypatch, capsys):
    calls: dict[str, object] = {}

    class _FakeBench:
        def __init__(self, system, event_bus, dataset, eval_timeout, max_workers):
            calls["init"] = {
                "system": system,
                "event_bus": event_bus,
                "dataset": dataset,
                "eval_timeout": eval_timeout,
                "max_workers": max_workers,
            }

        async def run_generate_only(self, limit=None):
            calls["limit"] = limit
            return Path("predictions.jsonl")

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(bench_main, "_boot_system", lambda: ("fake-system", "fake-bus"))
    monkeypatch.setattr(swebench_mod, "SWEBenchBench", _FakeBench)

    args = SimpleNamespace(
        dataset="lite",
        swebench_info=False,
        eval_predictions=None,
        eval_timeout=300,
        max_workers=4,
        generate_only=True,
        limit=1,
        output=None,
    )

    await bench_main._run_swebench(args)
    captured = capsys.readouterr()

    assert calls["limit"] == 1
    assert calls["init"] == {
        "system": "fake-system",
        "event_bus": "fake-bus",
        "dataset": "lite",
        "eval_timeout": 300,
        "max_workers": 4,
    }
    assert "Predictions saved to: predictions.jsonl" in captured.out
