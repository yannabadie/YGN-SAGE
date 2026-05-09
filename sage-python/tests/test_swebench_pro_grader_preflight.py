"""Tests for SWE-bench Pro local grader preflight."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parent.parent / "scripts"
_MODULE_PATH = _SCRIPTS / "swebench_pro_grader_preflight.py"
_SPEC = importlib.util.spec_from_file_location(
    "swebench_pro_grader_preflight", _MODULE_PATH,
)
preflight = importlib.util.module_from_spec(_SPEC)
sys.modules["swebench_pro_grader_preflight"] = preflight
_SPEC.loader.exec_module(preflight)


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["cmd"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_missing_docker_cli_blocks_local_grading(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    report = preflight.build_report(
        grader_repo=tmp_path / "grader",
        min_free_disk_gb=0.0,
    )

    assert report["decision"] == "NO_GO_LOCAL_DOCKER"
    assert report["local_grading_ready"] is False
    assert "docker_cli_missing" in report["blockers"]


def test_docker_daemon_failure_preserves_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[0] == "git":
            return preflight.CommandResult(cmd, 0, "abc123", "")
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "default", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"npipe:////./pipe/docker"', "")
        assert cmd[:2] == ["docker", "info"]
        return preflight.CommandResult(
            command=cmd,
            returncode=1,
            stdout='""',
            stderr="failed to connect to the docker API",
        )

    monkeypatch.setattr(preflight, "run_command", fake_run_command)

    report = preflight.build_report(
        grader_repo=tmp_path / "grader",
        min_free_disk_gb=0.0,
    )

    assert report["decision"] == "NO_GO_LOCAL_DOCKER"
    assert "docker_daemon_unreachable" in report["blockers"]
    assert (
        report["checks"]["docker_daemon"]["stderr"]
        == "failed to connect to the docker API"
    )


def test_windows_docker_engine_blocks_local_grading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "default", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"npipe:////./pipe/docker"', "")
        if cmd[-1] == "{{json .ServerVersion}}":
            return preflight.CommandResult(cmd, 0, '"26.1.0"', "")
        if cmd[-1] == "{{.OSType}}":
            return preflight.CommandResult(cmd, 0, "windows", "")
        return preflight.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(preflight, "run_command", fake_run_command)
    (tmp_path / "grader").mkdir()
    (tmp_path / "grader" / "swe_bench_pro_eval.py").write_text("", encoding="utf-8")

    report = preflight.build_report(
        grader_repo=tmp_path / "grader",
        min_free_disk_gb=0.0,
    )

    assert report["decision"] == "NO_GO_LOCAL_DOCKER"
    assert "docker_engine_not_linux" in report["blockers"]


def test_dirty_grader_repo_blocks_even_when_docker_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "default", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"npipe:////./pipe/docker"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{json .ServerVersion}}":
            return preflight.CommandResult(cmd, 0, '"26.1.0"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{.OSType}}":
            return preflight.CommandResult(cmd, 0, "linux", "")
        if cmd[:3] == ["docker", "run", "--rm"]:
            return preflight.CommandResult(cmd, 0, "hello", "")
        if cmd[0] == "git" and "rev-parse" in cmd:
            return preflight.CommandResult(
                cmd, 0, preflight.DEFAULT_EXPECTED_GRADER_COMMIT, "",
            )
        if cmd[0] == "git" and "remote" in cmd:
            return preflight.CommandResult(
                cmd,
                0,
                "origin\thttps://github.com/scaleapi/SWE-bench_Pro-os.git (fetch)\n",
                "",
            )
        if cmd[0] == "git" and "status" in cmd:
            return preflight.CommandResult(cmd, 0, " M swe_bench_pro_eval.py", "")
        return preflight.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(preflight, "run_command", fake_run_command)
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / ".git").mkdir()
    (grader / "swe_bench_pro_eval.py").write_text("", encoding="utf-8")

    report = preflight.build_report(grader_repo=grader, min_free_disk_gb=0.0)

    assert report["decision"] == "NO_GO_GRADER_REPO_DIRTY"
    assert "grader_repo_dirty" in report["blockers"]
    assert report["checks"]["grader_repo"]["dirty_status"] == "M swe_bench_pro_eval.py"


def test_linux_docker_and_clean_grader_are_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "default", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"npipe:////./pipe/docker"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{json .ServerVersion}}":
            return preflight.CommandResult(cmd, 0, '"26.1.0"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{.OSType}}":
            return preflight.CommandResult(cmd, 0, "linux", "")
        if cmd[:3] == ["docker", "run", "--rm"]:
            return preflight.CommandResult(cmd, 0, "hello", "")
        if cmd[0] == "git" and "rev-parse" in cmd:
            return preflight.CommandResult(
                cmd, 0, preflight.DEFAULT_EXPECTED_GRADER_COMMIT, "",
            )
        if cmd[0] == "git" and "remote" in cmd:
            return preflight.CommandResult(
                cmd,
                0,
                "origin\thttps://github.com/scaleapi/SWE-bench_Pro-os.git (fetch)\n",
                "",
            )
        if cmd[0] == "git" and "status" in cmd:
            return preflight.CommandResult(cmd, 0, "", "")
        return preflight.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(preflight, "run_command", fake_run_command)
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / ".git").mkdir()
    (grader / "swe_bench_pro_eval.py").write_text("", encoding="utf-8")

    report = preflight.build_report(grader_repo=grader, min_free_disk_gb=0.0)

    assert report["decision"] == "READY_LOCAL_DOCKER"
    assert report["local_grading_ready"] is True
    assert report["blockers"] == []


def test_clean_but_wrong_grader_commit_blocks_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "default", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"npipe:////./pipe/docker"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{json .ServerVersion}}":
            return preflight.CommandResult(cmd, 0, '"26.1.0"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{.OSType}}":
            return preflight.CommandResult(cmd, 0, "linux", "")
        if cmd[:3] == ["docker", "run", "--rm"]:
            return preflight.CommandResult(cmd, 0, "hello", "")
        if cmd[0] == "git" and "rev-parse" in cmd:
            return preflight.CommandResult(cmd, 0, "wrong", "")
        if cmd[0] == "git" and "remote" in cmd:
            return preflight.CommandResult(
                cmd,
                0,
                "origin\thttps://github.com/scaleapi/SWE-bench_Pro-os.git (fetch)\n",
                "",
            )
        if cmd[0] == "git" and "status" in cmd:
            return preflight.CommandResult(cmd, 0, "", "")
        return preflight.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(preflight, "run_command", fake_run_command)
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / ".git").mkdir()
    (grader / "swe_bench_pro_eval.py").write_text("", encoding="utf-8")

    report = preflight.build_report(grader_repo=grader, min_free_disk_gb=0.0)

    assert report["decision"] == "NO_GO_GRADER_REPO"
    assert "grader_repo_commit_mismatch" in report["blockers"]


def test_remote_docker_context_blocks_local_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight.shutil,
        "which",
        lambda name: "docker" if name == "docker" else None,
    )

    def fake_run_command(cmd, timeout):
        if cmd[:3] == ["docker", "context", "show"]:
            return preflight.CommandResult(cmd, 0, "remote", "")
        if cmd[:3] == ["docker", "context", "inspect"]:
            return preflight.CommandResult(cmd, 0, '"ssh://builder.example"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{json .ServerVersion}}":
            return preflight.CommandResult(cmd, 0, '"26.1.0"', "")
        if cmd[:2] == ["docker", "info"] and cmd[-1] == "{{.OSType}}":
            return preflight.CommandResult(cmd, 0, "linux", "")
        if cmd[:3] == ["docker", "run", "--rm"]:
            return preflight.CommandResult(cmd, 0, "hello", "")
        if cmd[0] == "git" and "rev-parse" in cmd:
            return preflight.CommandResult(
                cmd, 0, preflight.DEFAULT_EXPECTED_GRADER_COMMIT, "",
            )
        if cmd[0] == "git" and "remote" in cmd:
            return preflight.CommandResult(
                cmd,
                0,
                "origin\thttps://github.com/scaleapi/SWE-bench_Pro-os.git (fetch)\n",
                "",
            )
        if cmd[0] == "git" and "status" in cmd:
            return preflight.CommandResult(cmd, 0, "", "")
        return preflight.CommandResult(cmd, 0, "", "")

    monkeypatch.setattr(preflight, "run_command", fake_run_command)
    grader = tmp_path / "grader"
    grader.mkdir()
    (grader / ".git").mkdir()
    (grader / "swe_bench_pro_eval.py").write_text("", encoding="utf-8")

    report = preflight.build_report(grader_repo=grader, min_free_disk_gb=0.0)

    assert report["decision"] == "NO_GO_LOCAL_DOCKER"
    assert "docker_remote_context_unverified" in report["blockers"]
