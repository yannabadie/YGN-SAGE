"""Tests for the role-prompt registry.

The registry converts bare role strings like "planner", "coder_worker",
"output_formatter" into rich system prompts. Matching is substring-based
(lowercased), so template variants from Rust factories all resolve.
"""
from __future__ import annotations

import pytest

from sage.topology.role_prompts import (
    ROLE_PROMPTS,
    get_role_prompt,
)


def test_empty_role_returns_none():
    assert get_role_prompt("") is None
    assert get_role_prompt(None) is None


def test_unknown_role_returns_none():
    """Roles that don't match any alias should fall through to None so the
    runner uses its harness default template."""
    assert get_role_prompt("xyz-random-role") is None
    assert get_role_prompt("foo") is None


@pytest.mark.parametrize(
    "role",
    ["planner", "Planner", "input_processor", "DECOMPOSER", "text_planner"],
)
def test_planner_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "PLANNER" in prompt
    assert "checklist" in prompt.lower()


@pytest.mark.parametrize(
    "role",
    ["coder", "actor", "coder_worker", "Coder-1"],
)
def test_coder_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "CODER" in prompt
    # Must mandate at least one tool-use step so one-shot patches don't win.
    assert "execute_bash" in prompt
    assert "AT LEAST 3" in prompt


@pytest.mark.parametrize(
    "role",
    ["synthesizer", "aggregator", "output_formatter", "formatter"],
)
def test_synthesizer_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "SYNTHESIZER" in prompt
    # Must forbid new tool calls — this is the terminal node.
    assert "Never emit new tool calls" in prompt


@pytest.mark.parametrize(
    "role",
    ["verifier", "validator", "critic", "judge"],
)
def test_verifier_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "VERIFIER" in prompt
    assert "PASS" in prompt and "FAIL" in prompt


@pytest.mark.parametrize(
    "role",
    ["worker", "worker_0", "thinker", "brainstormer", "worker-42"],
)
def test_worker_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "WORKER" in prompt


@pytest.mark.parametrize(
    "role",
    ["source", "seed", "trigger"],
)
def test_source_aliases_resolve(role):
    prompt = get_role_prompt(role)
    assert prompt is not None
    assert "SOURCE" in prompt


def test_synthesizer_wins_over_worker():
    """Role strings that contain both 'worker' and 'synthesizer' should
    pick synthesizer first (the more specific terminal role)."""
    assert "SYNTHESIZER" in get_role_prompt("synthesizer_worker")


def test_all_prompts_have_hard_rules_section():
    """Every prompt must have a format discipline to anchor the model."""
    for aliases, prompt in ROLE_PROMPTS:
        assert "Output" in prompt or "output" in prompt, f"{aliases} missing Output section"
        assert len(prompt) > 200, f"{aliases} prompt too short ({len(prompt)} chars)"


def test_registry_has_expected_role_groups():
    """Sanity: we expect at least 6 distinct role groups — planner, coder,
    synthesizer, verifier, source, worker."""
    all_aliases = {a for aliases, _ in ROLE_PROMPTS for a in aliases}
    assert {"planner", "coder", "synthesizer", "verifier", "source", "worker"} <= all_aliases
