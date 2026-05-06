"""Tests for the SWE-bench Pro grader CSV adapter."""
from __future__ import annotations

import csv
import importlib.util
import sys
import types
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).parent.parent
    / "scripts"
    / "swebench_pro_build_grader_csv.py"
).resolve()


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_build_grader_csv", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_pro_build_grader_csv"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_csv_preserves_repo_for_docker_image_uri(tmp_path, monkeypatch):
    iid = "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
    fake_datasets = types.ModuleType("datasets")

    def load_dataset(name, split):
        assert name == "ScaleAI/SWE-bench_Pro"
        assert split == "test"
        return [
            {
                "instance_id": iid,
                "repo": "NodeBB/NodeBB",
                "before_repo_set_cmd": "git reset --hard abc\n",
                "selected_test_files_to_run": '["test/user/emails.js"]',
                "base_commit": "abc",
                "fail_to_pass": '["test_email_confirmation"]',
                "pass_to_pass": '["test_existing_behavior"]',
            }
        ]

    fake_datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = tmp_path / "grader.csv"
    _load_module().build_csv([iid], output)

    with output.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert reader.fieldnames == [
        "instance_id",
        "repo",
        "before_repo_set_cmd",
        "selected_test_files_to_run",
        "base_commit",
        "fail_to_pass",
        "pass_to_pass",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ]
    assert rows == [
        {
            "instance_id": iid,
            "repo": "NodeBB/NodeBB",
            "before_repo_set_cmd": "git reset --hard abc\n",
            "selected_test_files_to_run": '["test/user/emails.js"]',
            "base_commit": "abc",
            "fail_to_pass": '["test_email_confirmation"]',
            "pass_to_pass": '["test_existing_behavior"]',
            "FAIL_TO_PASS": '["test_email_confirmation"]',
            "PASS_TO_PASS": '["test_existing_behavior"]',
        }
    ]
