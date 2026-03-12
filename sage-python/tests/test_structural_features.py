"""Test Python reimplementation of StructuralFeatures (was Rust features.rs).

Field-for-field match with Rust: word_count, has_code_block, has_question_mark,
keyword_complexity (0.0-1.0), keyword_uncertainty (0.0-1.0), tool_required.
"""
import pytest
from sage.strategy.structural_features import StructuralFeatures


class TestStructuralFeatures:
    def test_simple_factual(self):
        sf = StructuralFeatures.extract("What is the capital of France?")
        assert sf.has_question_mark is True
        assert sf.has_code_block is False
        assert sf.tool_required is False
        assert sf.keyword_complexity < 0.35

    def test_code_task(self):
        sf = StructuralFeatures.extract("Write a Python function to sort a list")
        assert sf.has_code_block is False
        assert sf.has_question_mark is False
        assert abs(sf.keyword_complexity - 0.35) < 0.01

    def test_algo_task_high_complexity(self):
        sf = StructuralFeatures.extract(
            "Debug the race condition in the concurrent queue implementation"
        )
        assert sf.keyword_complexity >= 0.50

    def test_code_block_detection(self):
        task = "Fix this code:\n```python\ndef foo():\n    pass\n```"
        sf = StructuralFeatures.extract(task)
        assert sf.has_code_block is True
        assert sf.keyword_complexity >= 0.55

    def test_tool_required(self):
        sf = StructuralFeatures.extract("Search the web for recent Rust async tutorials")
        assert sf.tool_required is True

    def test_uncertainty_keywords(self):
        sf = StructuralFeatures.extract("Maybe investigate the intermittent flaky test")
        assert sf.keyword_uncertainty >= 0.75

    def test_long_task_scaling(self):
        padding = "word " * 130
        sf = StructuralFeatures.extract(f"Implement an algorithm that {padding}")
        assert sf.word_count > 120
        assert abs(sf.keyword_complexity - 0.70) < 0.01

    def test_empty(self):
        sf = StructuralFeatures.extract("")
        assert sf.word_count == 0
        assert abs(sf.keyword_complexity - 0.2) < 0.01
        assert sf.keyword_uncertainty == 0.0
