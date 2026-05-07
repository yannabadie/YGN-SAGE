"""Test the rust_core_status dashboard generator.

RED phase: test defines the required JSON schema before the module exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REQUIRED_KEYS = [
    "generated_at_utc",
    "source_commit_sha",
    "sage_core_build_commit_sha",
    "sage_core_build_profile",
    "routing",
    "topology",
    "sandbox",
    "memory",
]

REQUIRED_ROUTING_KEYS = [
    "system_router_gt_accuracy",
    "knn_gt_accuracy",
    "gt_dataset_path",
    "last_eval_artifact",
]

REQUIRED_TOPOLOGY_KEYS = [
    "six_paths_tested",
]

REQUIRED_TOPOLOGY_PATH_KEYS = [
    "smmu_hit",
    "archive_hit",
    "llm_synthesis",
    "mutation",
    "mcts_search",
    "template_fallback",
]

REQUIRED_SANDBOX_KEYS = [
    "embedded_wasm_available",
    "validate_and_execute_subprocess_fallback",
    "raw_exec_requires_env",
]

REQUIRED_MEMORY_KEYS = [
    "smmu_persistent",
]


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_rust_core_status_json_has_required_top_level_keys(key: str) -> None:
    """Every required top-level key must be present in the output."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    assert key in status, f"Missing required key: {key!r}"


@pytest.mark.parametrize("key", REQUIRED_ROUTING_KEYS)
def test_rust_core_status_routing_section_has_required_keys(key: str) -> None:
    """Every required routing key must be present."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    routing = status.get("routing", {})
    assert key in routing, f"Missing routing key: {key!r}"


@pytest.mark.parametrize("key", REQUIRED_TOPOLOGY_KEYS)
def test_rust_core_status_topology_section_has_required_keys(key: str) -> None:
    """Every required topology key must be present."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    topology = status.get("topology", {})
    assert key in topology, f"Missing topology key: {key!r}"


@pytest.mark.parametrize("key", REQUIRED_TOPOLOGY_PATH_KEYS)
def test_rust_core_status_topology_paths_have_required_keys(key: str) -> None:
    """Every topology path must be represented."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    paths = status.get("topology", {}).get("paths", {})
    assert key in paths, f"Missing topology path key: {key!r}"
    assert isinstance(paths[key], bool), (
        f"topology path {key!r} must be bool, got {type(paths[key])!r}"
    )


@pytest.mark.parametrize("key", REQUIRED_SANDBOX_KEYS)
def test_rust_core_status_sandbox_section_has_required_keys(key: str) -> None:
    """Every required sandbox key must be present."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    sandbox = status.get("sandbox", {})
    assert key in sandbox, f"Missing sandbox key: {key!r}"


@pytest.mark.parametrize("key", REQUIRED_MEMORY_KEYS)
def test_rust_core_status_memory_section_has_required_keys(key: str) -> None:
    """Every required memory key must be present."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    memory = status.get("memory", {})
    assert key in memory, f"Missing memory key: {key!r}"


def test_rust_core_status_json_is_valid_json() -> None:
    """Output must be serializable as valid JSON."""
    from sage.ops.rust_core_status import generate_status

    status = generate_status()
    dumped = json.dumps(status, indent=2)
    assert isinstance(dumped, str)
    assert len(dumped) > 0


def test_rust_core_status_cli_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--json flag writes to stdout."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "sage.ops.rust_core_status", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    parsed = json.loads(result.stdout)
    for key in REQUIRED_KEYS:
        assert key in parsed, f"CLI output missing key: {key!r}"
