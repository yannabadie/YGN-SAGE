import json
import sys
import tempfile
import types
from pathlib import Path

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

from sage.strategy.training import compute_label, export_training_data


def test_compute_label_good_quality():
    assert compute_label(routed_tier=1, quality=0.9, cost_usd=0.001) == 1
    assert compute_label(routed_tier=2, quality=0.85, cost_usd=0.01) == 2


def test_compute_label_low_quality_escalates():
    assert compute_label(routed_tier=1, quality=0.3, cost_usd=0.001) == 2
    assert compute_label(routed_tier=2, quality=0.4, cost_usd=0.01) == 3


def test_compute_label_s3_no_further_escalation():
    assert compute_label(routed_tier=3, quality=0.3, cost_usd=0.03) == 3


def test_export_training_data():
    feedback = [
        {
            "task": "What is 2+2?",
            "routed_tier": 1,
            "actual_quality": 0.95,
            "latency_ms": 50,
            "cost_usd": 0.001,
        },
        {
            "task": "Write bubble sort",
            "routed_tier": 2,
            "actual_quality": 0.7,
            "latency_ms": 200,
            "cost_usd": 0.01,
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.jsonl"
        result = export_training_data(feedback, output_path=out)
        assert result.exists()
        lines = result.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        ex1 = json.loads(lines[0])
        assert ex1["task"] == "What is 2+2?"
        assert ex1["label"] == 1  # good quality, keep S1
