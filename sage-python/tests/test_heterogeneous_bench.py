"""Tests for heterogeneous evaluation benchmark."""
import json
import pytest
from pathlib import Path


def test_eval_set_structure():
    """Evaluation set JSON has required fields."""
    path = Path(__file__).parent.parent / "config" / "heterogeneous_eval.json"
    if not path.exists():
        pytest.skip("Eval set not yet created")
    data = json.loads(path.read_text())
    assert "tasks" in data
    assert len(data["tasks"]) == 50

    categories = {"code": 0, "reasoning": 0, "multi_turn": 0, "research": 0}
    for task in data["tasks"]:
        assert "id" in task
        assert "prompt" in task
        assert "category" in task
        assert "expected_system" in task
        assert "expected_pillar_benefit" in task
        assert task["category"] in categories
        categories[task["category"]] += 1

    assert categories["code"] == 15
    assert categories["reasoning"] == 15
    assert categories["multi_turn"] == 10
    assert categories["research"] == 10


def test_heterogeneous_bench_adapter_exists():
    """HeterogeneousBench adapter can be imported."""
    from sage.bench.heterogeneous_bench import HeterogeneousBench
    assert hasattr(HeterogeneousBench, "run")
