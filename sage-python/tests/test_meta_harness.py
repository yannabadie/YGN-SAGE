"""Tests for Meta-Harness config, patcher, and search loop."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sage.meta_harness.config import (
    HarnessConfig,
    ContextConfig,
    PromptConfig,
    ExecutionConfig,
    TopologyConfig,
)
from sage.meta_harness.search_loop import MetaHarnessLoop


class TestHarnessConfig:
    def test_baseline_defaults(self):
        cfg = HarnessConfig()
        assert cfg.id == "baseline"
        assert cfg.context.budget_ratio == 0.70
        assert cfg.context.similarity_threshold == 0.90
        assert cfg.prompts.default_template == "You are acting as: {role}."
        assert cfg.execution.quality_cascade_threshold == 0.30
        assert cfg.execution.max_debate_rounds == 3
        assert cfg.topology.s1_skip_threshold == 1

    def test_json_roundtrip(self):
        cfg = HarnessConfig(
            id="test-001",
            description="test config",
            context=ContextConfig(
                predecessor_format="<agent>{text}</agent>",
                budget_ratio=0.85,
            ),
            prompts=PromptConfig(
                global_suffix="Think step by step.",
            ),
        )
        json_str = cfg.to_json()
        loaded = HarnessConfig.from_json(json_str)
        assert loaded.id == "test-001"
        assert loaded.context.predecessor_format == "<agent>{text}</agent>"
        assert loaded.context.budget_ratio == 0.85
        assert loaded.prompts.global_suffix == "Think step by step."

    def test_save_load(self, tmp_path: Path):
        cfg = HarnessConfig(id="save-test", description="persistence test")
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = HarnessConfig.load(path)
        assert loaded.id == "save-test"
        assert loaded.description == "persistence test"

    def test_diff(self):
        a = HarnessConfig(id="a", context=ContextConfig(budget_ratio=0.70))
        b = HarnessConfig(id="b", context=ContextConfig(budget_ratio=0.85))
        diff = a.diff(b)
        assert "id" in diff
        assert "context.budget_ratio" in diff
        assert diff["context.budget_ratio"] == (0.70, 0.85)

    def test_diff_same_config(self):
        a = HarnessConfig()
        b = HarnessConfig()
        diff = a.diff(b)
        assert len(diff) == 0

    def test_role_overrides(self):
        cfg = HarnessConfig(
            prompts=PromptConfig(
                role_overrides={
                    "coder": "You are an expert Python programmer. {role}.",
                    "reviewer": "You are a critical code reviewer.",
                },
            ),
        )
        assert "coder" in cfg.prompts.role_overrides
        json_str = cfg.to_json()
        loaded = HarnessConfig.from_json(json_str)
        assert loaded.prompts.role_overrides["coder"] == "You are an expert Python programmer. {role}."


class TestMetaHarnessLoop:
    def test_init_workspace(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()

        assert (tmp_path / "mh" / "baseline" / "config.json").exists()
        assert (tmp_path / "mh" / "leaderboard.json").exists()
        assert (tmp_path / "mh" / "PROPOSER_INSTRUCTIONS.md").exists()
        assert (tmp_path / "mh" / "candidates").is_dir()

        # Baseline config should be valid
        cfg = HarnessConfig.load(tmp_path / "mh" / "baseline" / "config.json")
        assert cfg.id == "baseline"

    def test_next_candidate_id_empty(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()
        assert loop.next_candidate_id() == "001"

    def test_next_candidate_id_sequential(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()

        (loop.candidates_dir / "001").mkdir()
        (loop.candidates_dir / "002").mkdir()
        assert loop.next_candidate_id() == "003"

    def test_leaderboard_roundtrip(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()

        lb = loop._load_leaderboard()
        assert lb == []

        loop._save_leaderboard([{"candidate_id": "001", "aggregate_score": 0.5}])
        lb = loop._load_leaderboard()
        assert len(lb) == 1
        assert lb[0]["aggregate_score"] == 0.5

    def test_status_empty(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()
        status = loop.status()
        assert "No candidates" in status

    def test_status_with_entries(self, tmp_path: Path):
        loop = MetaHarnessLoop(workspace=tmp_path / "mh")
        loop.init_workspace()

        loop._save_leaderboard([
            {
                "candidate_id": "001",
                "description": "test candidate",
                "aggregate_score": 0.75,
                "aggregate_pass_rate": 0.8,
                "token_usage": 5000,
                "total_latency_ms": 12000,
                "evaluated_at": "2026-04-02T00:00:00Z",
                "parent_id": "baseline",
            },
        ])
        status = loop.status()
        assert "001" in status
        assert "0.750" in status


class TestPatcher:
    """Test HarnessPatcher with mock runner."""

    def test_patch_unpatch_max_rounds(self):
        from sage.meta_harness.patcher import HarnessPatcher

        cfg = HarnessConfig(execution=ExecutionConfig(max_debate_rounds=5))
        patcher = HarnessPatcher(cfg)

        class MockRunner:
            _max_rounds = 3
            _gather_predecessor_context = lambda self, idx: ""
            _context_budget_per_predecessor = lambda self, n, idx=0: 1000
            _execute_node = None
            _node_outputs = {}
            graph = None

        runner = MockRunner()
        assert runner._max_rounds == 3

        patcher.patch_runner(runner)
        assert runner._max_rounds == 5

        patcher.unpatch_runner(runner)
        assert runner._max_rounds == 3

    def test_context_manager(self):
        from sage.meta_harness.patcher import HarnessPatcher

        cfg = HarnessConfig(execution=ExecutionConfig(max_debate_rounds=7))
        patcher = HarnessPatcher(cfg)

        class MockRunner:
            _max_rounds = 3
            _gather_predecessor_context = lambda self, idx: ""
            _context_budget_per_predecessor = lambda self, n, idx=0: 1000
            _execute_node = None
            _node_outputs = {}
            graph = None

        runner = MockRunner()

        with patcher.patched(runner=runner):
            assert runner._max_rounds == 7

        assert runner._max_rounds == 3
