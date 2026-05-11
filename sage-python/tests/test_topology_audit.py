"""Tests for slice 10A — topology + control-surface audit.

cgpro VERIFY 2026-05-11 RF#C MODIFY: improve observability around
bandit-stochastic topology selection so paired-reruns can attribute
outcomes to prompt/profile changes vs Thompson noise. Do NOT make
topology deterministic.

Covers:
- ``_topology_audit_from_file`` extracts every required field
- Sentinel node detection
- Provider-policy substitution detection
- Control-surface completeness booleans
- Missing-file path returns a well-shaped default
- Real slice 9 artefact fingerprint (regression catch)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "sage-python" / "scripts" / "run_dryrun_arm_d.py"


def _load_arm_d() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_dryrun_arm_d", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


arm_d = _load_arm_d()


def _make_events(events_path: Path, events: list[dict]) -> None:
    events_path.write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )


def test_topology_audit_missing_file_returns_defaults(tmp_path: Path) -> None:
    """Audit of a non-existent file must return a shape that downstream
    consumers can iterate over without KeyError.
    """
    result = arm_d._topology_audit_from_file(tmp_path / "doesnt-exist.jsonl")
    assert result["topology_template"] is None
    assert result["topology_id"] is None
    assert result["nodes"] == []
    assert result["oracle"] is None
    assert result["control_surface"] == {
        "routing_decision_emitted": False,
        "topology_selected_emitted": False,
        "model_assigned_for_all_nodes": False,
        "oracle_verdict_emitted": False,
        "cli_complete_emitted": False,
    }
    assert result["provider_policy_substitution_detected"] is False


def test_topology_audit_extracts_topology_fields(tmp_path: Path) -> None:
    """topology_selected event populates template/id/node_count/edge_count."""
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "topology_selected",
            "template_type": "sequential",
            "topology_id": "01ABCDEF",
            "node_count": 3,
            "edge_count": 2,
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    assert result["topology_template"] == "sequential"
    assert result["topology_id"] == "01ABCDEF"
    assert result["node_count"] == 3
    assert result["edge_count"] == 2
    assert result["control_surface"]["topology_selected_emitted"] is True
    assert result["control_surface"]["routing_decision_emitted"] is False


def test_topology_audit_extracts_routing_fields(tmp_path: Path) -> None:
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "routing_decision",
            "routing_source": "rust_system_router",
            "system": 3,
            "domain": "code",
            "confidence": 0.8788,
            "model_id": "gpt-5.4-pro",
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    assert result["routing_source"] == "rust_system_router"
    assert result["routing_system"] == 3
    assert result["routing_domain"] == "code"
    assert abs(result["routing_confidence"] - 0.8788) < 1e-6
    assert result["routing_model_id"] == "gpt-5.4-pro"
    assert result["control_surface"]["routing_decision_emitted"] is True


def test_topology_audit_detects_provider_policy_substitution(tmp_path: Path) -> None:
    """Routing chose model X, execution used model Y → substitution detected."""
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "routing_decision",
            "model_id": "gpt-5.4-pro",
        },
        {
            "event_type": "model_assigned",
            "node_id": "0",
            "node_role": "coder",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
        },
        {
            "event_type": "node_completed",
            "node_id": "0",
            "node_role": "coder",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "output_length": 1500,
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    assert result["provider_policy_substitution_detected"] is True
    assert result["routing_model_id"] == "gpt-5.4-pro"
    assert result["nodes"][0]["completed_model_id"] == "deepseek-v4-pro"


def test_topology_audit_no_substitution_when_routing_matches(tmp_path: Path) -> None:
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "routing_decision",
            "model_id": "deepseek-v4-pro",
        },
        {
            "event_type": "model_assigned",
            "node_id": "0",
            "node_role": "coder",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
        },
        {
            "event_type": "node_completed",
            "node_id": "0",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "output_length": 1500,
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    assert result["provider_policy_substitution_detected"] is False


def test_topology_audit_detects_sentinel_via_payload(tmp_path: Path) -> None:
    """Sentinel detection: payload is the EMPTY_STEP_SENTINEL string."""
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "node_completed",
            "node_id": "0",
            "node_role": "planner",
            "model_id": "deepseek-v4-flash",
            "provider_id": "deepseek",
            "output_length": 51,
            "payload": "[sage: agent exited after 5 steps with no content]",
        },
        {
            "event_type": "node_completed",
            "node_id": "1",
            "node_role": "coder",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "output_length": 2433,
            "payload": "real coder output 2433 chars...",
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    nodes = {n["node_id"]: n for n in result["nodes"]}
    assert nodes["0"]["is_sentinel"] is True
    assert nodes["1"]["is_sentinel"] is False


def test_topology_audit_detects_sentinel_via_output_length(tmp_path: Path) -> None:
    """When payload is absent (redacted runs), sentinel detection falls back
    to output_length <= 51 heuristic.
    """
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "node_completed",
            "node_id": "0",
            "output_length": 51,
            # no payload field
        },
        {
            "event_type": "node_completed",
            "node_id": "1",
            "output_length": 2000,
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    nodes = {n["node_id"]: n for n in result["nodes"]}
    assert nodes["0"]["is_sentinel"] is True
    assert nodes["1"]["is_sentinel"] is False


def test_topology_audit_control_surface_all_emitted(tmp_path: Path) -> None:
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {"event_type": "routing_decision", "model_id": "X"},
        {"event_type": "topology_selected", "template_type": "sequential",
         "topology_id": "T1", "node_count": 1, "edge_count": 0},
        {"event_type": "model_assigned", "node_id": "0", "node_role": "r",
         "model_id": "X", "provider_id": "P"},
        {"event_type": "oracle_verdict", "payload": {"trainable": False}},
        {"event_type": "cli_complete", "payload": {"outcome": "success"}},
    ])
    result = arm_d._topology_audit_from_file(events_path)
    cs = result["control_surface"]
    assert cs["routing_decision_emitted"] is True
    assert cs["topology_selected_emitted"] is True
    assert cs["model_assigned_for_all_nodes"] is True
    assert cs["oracle_verdict_emitted"] is True
    assert cs["cli_complete_emitted"] is True


def test_topology_audit_model_assigned_partial_means_incomplete(tmp_path: Path) -> None:
    """topology_selected says node_count=3 but only 2 model_assigned →
    control_surface.model_assigned_for_all_nodes = False.
    """
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {"event_type": "topology_selected", "template_type": "sequential",
         "topology_id": "T1", "node_count": 3, "edge_count": 2},
        {"event_type": "model_assigned", "node_id": "0", "model_id": "X", "provider_id": "P"},
        {"event_type": "model_assigned", "node_id": "1", "model_id": "X", "provider_id": "P"},
        # node 2 never assigned
    ])
    result = arm_d._topology_audit_from_file(events_path)
    assert result["control_surface"]["model_assigned_for_all_nodes"] is False


def test_topology_audit_oracle_payload_captured(tmp_path: Path) -> None:
    events_path = tmp_path / "t.events.jsonl"
    _make_events(events_path, [
        {
            "event_type": "oracle_verdict",
            "payload": {
                "trainable": False,
                "verdict_source": "abstain",
                "quality_label": "unknown",
                "score": None,
                "reason_codes": ["hierarchy_exhausted"],
                "extra_field": "ignored",  # only documented fields kept
            },
        },
    ])
    result = arm_d._topology_audit_from_file(events_path)
    oracle = result["oracle"]
    assert oracle is not None
    assert oracle["trainable"] is False
    assert oracle["verdict_source"] == "abstain"
    assert oracle["reason_codes"] == ["hierarchy_exhausted"]
    assert "extra_field" not in oracle


def test_topology_audit_slice_9_real_fingerprint() -> None:
    """Run topology_audit on the slice 9 NodeBB events and pin the
    fingerprint. If a future refactor changes the audit logic, this
    catches it.
    """
    events_path = (
        _REPO_ROOT
        / "docs"
        / "benchmarks"
        / "2026-05-11-canary-patch-focused-prompt-profile"
        / "run"
        / "per_task"
        / "instance_NodeBB__NodeBB-76c6e30282906ac664f2c9278fc90999b27b1f48-vd59a5728dfc977f44533186ace531248c2917516.events.jsonl"
    )
    if not events_path.is_file():
        pytest.skip(f"slice 9 artefact missing: {events_path}")

    result = arm_d._topology_audit_from_file(events_path)
    # NodeBB was sequential 2-node, routing chose gpt-5.4-pro (openai),
    # provider allowlist forced substitution to deepseek + google.
    assert result["topology_template"] == "sequential"
    assert result["node_count"] == 2
    assert result["routing_model_id"] == "gpt-5.4-pro"
    assert result["routing_source"] == "rust_system_router"
    assert result["routing_system"] == 3
    assert result["routing_domain"] == "code"
    assert result["provider_policy_substitution_detected"] is True
    # No sentinel on NodeBB (both coder and mixer produced content)
    assert all(not n.get("is_sentinel") for n in result["nodes"])
    # Control surface complete
    cs = result["control_surface"]
    assert all(cs.values())
    # Oracle abstained
    assert result["oracle"]["trainable"] is False
    assert result["oracle"]["verdict_source"] == "abstain"


def test_topology_audit_slice_9_teleport_has_planner_sentinel() -> None:
    """teleport was sequential 3-node with planner sentinel. The audit
    must catch the sentinel.
    """
    events_path = (
        _REPO_ROOT
        / "docs"
        / "benchmarks"
        / "2026-05-11-canary-patch-focused-prompt-profile"
        / "run"
        / "per_task"
        / "instance_gravitational__teleport-6eaaf3a27e64f4ef4ef855bd35d7ec338cf17460-v626ec2a48416b10a88641359a169d99e935ff037.events.jsonl"
    )
    if not events_path.is_file():
        pytest.skip(f"slice 9 artefact missing: {events_path}")

    result = arm_d._topology_audit_from_file(events_path)
    nodes_by_role = {n.get("completed_role"): n for n in result["nodes"]}
    assert "planner" in nodes_by_role
    assert nodes_by_role["planner"]["is_sentinel"] is True
    assert nodes_by_role["coder"]["is_sentinel"] is False
    assert nodes_by_role["synthesizer"]["is_sentinel"] is False
