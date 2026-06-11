"""Post-emission grounding guard — cgpro GROUNDING DESIGN_LOCKED
(2026-06-11, sequence 'Post-émission grounding guard')."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sage.patch_artifacts import patch_path_coverage


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "real.py").write_text("a = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True,
                   capture_output=True)
    return repo


def test_patch_with_missing_existing_path_is_grounding_path_missing(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    patch = (
        "--- a/src/real.py\n+++ b/src/real.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
        "--- a/src/invented.py\n+++ b/src/invented.py\n"
        "@@ -1,1 +1,1 @@\n-x\n+y\n"
    )
    cov = patch_path_coverage(patch, str(repo))
    assert cov["missing"] == ["src/invented.py"]
    assert cov["referenced"] == ["src/real.py", "src/invented.py"]
    assert cov["coverage"] == 0.5


def test_new_file_allowed_only_when_profile_permits_new_files(
    monkeypatch, tmp_path
) -> None:
    repo = _repo(tmp_path)
    patch = (
        "--- /dev/null\n+++ b/src/brand_new.py\n@@ -0,0 +1,1 @@\n+z = 1\n"
    )
    monkeypatch.delenv("SAGE_ARTIFACT_ALLOW_NEW_FILES", raising=False)
    cov = patch_path_coverage(patch, str(repo))
    assert cov["new_files"] == ["src/brand_new.py"]
    assert cov["missing"] == ["src/brand_new.py"]  # blocked by default
    monkeypatch.setenv("SAGE_ARTIFACT_ALLOW_NEW_FILES", "1")
    cov2 = patch_path_coverage(patch, str(repo))
    assert cov2["missing"] == []
    assert cov2["coverage"] == 1.0


def test_patch_path_coverage_telemetry_records_missing_and_total_paths(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    good = (
        "--- a/src/real.py\n+++ b/src/real.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
    )
    cov = patch_path_coverage(good, str(repo))
    assert cov["missing"] == [] and cov["coverage"] == 1.0
    assert patch_path_coverage("", str(repo))["coverage"] == 0.0
