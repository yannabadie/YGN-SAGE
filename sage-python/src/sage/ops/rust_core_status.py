"""Generate a machine-readable Rust Core status dashboard (json).

Usage::

    python -m sage.ops.rust_core_status --json
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys


def _git_head_sha() -> str:
    """Return the current git HEAD SHA, or "unknown"."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _wasm_available() -> bool | None:
    """Return True iff the embedded Wasm sandbox is available."""
    try:
        import sage_core
        return bool(sage_core.embedded_wasm_available())
    except (ImportError, AttributeError):
        return None


def _sage_core_build_attrs() -> dict[str, str]:
    """Read sage_core build-info attributes."""
    try:
        import sage_core
        return {
            "commit_sha": str(getattr(sage_core, "__commit_sha__", "unknown")),
            "build_profile": str(getattr(sage_core, "__build_profile__", "unknown")),
            "version": str(getattr(sage_core, "__version__", "unknown")),
        }
    except ImportError:
        return {"commit_sha": "unknown", "build_profile": "unknown", "version": "unknown"}


def generate_status() -> dict:
    """Return a machine-readable Rust Core status snapshot.

    The schema matches the Phase 0 dashboard defined in AUDIT/AUDITRUST.md.
    """
    build = _sage_core_build_attrs()
    return {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit_sha": _git_head_sha(),
        "sage_core_build_commit_sha": build["commit_sha"],
        "sage_core_build_profile": build["build_profile"],
        "sage_core_version": build["version"],
        "routing": {
            "system_router_gt_accuracy": None,
            "knn_gt_accuracy": None,
            "gt_dataset_path": "sage-python/config/routing_ground_truth.json",
            "last_eval_artifact": None,
        },
        "topology": {
            "six_paths_tested": False,
            "paths": {
                "smmu_hit": False,
                "archive_hit": False,
                "llm_synthesis": False,
                "mutation": False,
                "mcts_search": False,
                "template_fallback": False,
            },
        },
        "sandbox": {
            "embedded_wasm_available": _wasm_available(),
            "validate_and_execute_subprocess_fallback": False,
            "raw_exec_requires_env": True,
        },
        "memory": {
            "smmu_persistent": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--json" in argv:
        print(json.dumps(generate_status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
