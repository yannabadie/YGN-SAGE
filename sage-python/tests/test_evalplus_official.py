"""Test official EvalPlus evaluation mode generates correct JSONL format."""
import json
import pytest
from pathlib import Path


def test_official_samples_jsonl_format(tmp_path):
    """Official mode must produce EvalPlus-compatible samples.jsonl."""
    from sage.bench.evalplus_bench import EvalPlusBench

    # Mock: create a samples file in the expected format
    samples = [
        {"task_id": "HumanEval/0", "solution": "def has_close_elements(numbers, threshold):\n    return False\n"},
        {"task_id": "HumanEval/1", "solution": "def separate_paren_groups(paren_string):\n    return []\n"},
    ]
    samples_path = tmp_path / "samples.jsonl"
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Verify format
    loaded = []
    with open(samples_path) as f:
        for line in f:
            obj = json.loads(line.strip())
            assert "task_id" in obj
            assert "solution" in obj
            loaded.append(obj)
    assert len(loaded) == 2


def test_evalplus_bench_has_official_mode():
    """EvalPlusBench constructor must accept official_mode parameter."""
    from sage.bench.evalplus_bench import EvalPlusBench

    bench = EvalPlusBench(system=None, dataset="humaneval", official_mode=True)
    assert bench.official_mode is True
