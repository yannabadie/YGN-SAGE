#!/usr/bin/env python3
"""Preflight local SWE-bench Pro grading readiness.

This gate is intentionally narrower than provider health checks: it only
answers whether this checkout can run the official Pro grader locally with
Linux Docker, using a clean pinned grader checkout. If not, a benchmark run may
still generate predictions, but it must not be called official graded evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRADER_REPO = REPO_ROOT / "external" / "SWE-bench_Pro-os"
DEFAULT_EXPECTED_GRADER_REMOTE = "github.com/scaleapi/SWE-bench_Pro-os"
DEFAULT_EXPECTED_GRADER_COMMIT = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
DEFAULT_MIN_FREE_DISK_GB = 120.0


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stdout"] = self.stdout.strip()
        data["stderr"] = self.stderr.strip()
        return data


def run_command(cmd: list[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(
            command=cmd,
            returncode=124,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


def _clean_scalar(stdout: str) -> str:
    value = stdout.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.strip()


def _git_check(
    path: Path,
    timeout: float,
    *,
    expected_remote: str | None,
    expected_commit: str | None,
) -> dict[str, Any]:
    check: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "eval_script_exists": (path / "swe_bench_pro_eval.py").exists(),
        "git_commit": None,
        "expected_commit": expected_commit,
        "commit_matches_expected": None,
        "remote_urls": [],
        "expected_remote": expected_remote,
        "remote_matches_expected": None,
        "dirty_status": None,
    }
    if not check["exists"] or not check["eval_script_exists"]:
        return check
    if not (path / ".git").exists():
        check["dirty_status"] = "not_a_git_checkout"
        return check

    commit = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout)
    status = run_command(["git", "-C", str(path), "status", "--short"], timeout)
    remotes = run_command(["git", "-C", str(path), "remote", "-v"], timeout)
    if commit.returncode == 0:
        git_commit = commit.stdout.strip()
        check["git_commit"] = git_commit
        check["commit_matches_expected"] = (
            git_commit == expected_commit if expected_commit else True
        )
    else:
        check["git_error"] = commit.as_dict()
    if status.returncode == 0:
        check["dirty_status"] = status.stdout.strip()
    else:
        check["git_status_error"] = status.as_dict()
        check["dirty_status"] = "unknown"
    if remotes.returncode == 0:
        remote_urls = _remote_urls_from_git_output(remotes.stdout)
        check["remote_urls"] = remote_urls
        check["remote_matches_expected"] = (
            _matches_expected_remote(remote_urls, expected_remote)
            if expected_remote
            else True
        )
    else:
        check["git_remote_error"] = remotes.as_dict()
        check["remote_matches_expected"] = False if expected_remote else None
    return check


def _remote_urls_from_git_output(stdout: str) -> list[str]:
    urls: list[str] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in urls:
            urls.append(parts[1])
    return urls


def _normalize_remote_url(value: str) -> str:
    normalized = value.strip().lower()
    for prefix in ("https://", "http://", "git@"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    normalized = normalized.replace(":", "/", 1)
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.strip("/")


def _matches_expected_remote(remote_urls: list[str], expected_remote: str | None) -> bool:
    if not expected_remote:
        return True
    expected = _normalize_remote_url(expected_remote)
    return any(_normalize_remote_url(url) == expected for url in remote_urls)


def _docker_context_check(docker_exe: str, timeout: float) -> dict[str, Any]:
    docker_host = os.environ.get("DOCKER_HOST", "")
    docker_context_env = os.environ.get("DOCKER_CONTEXT", "")
    context_show = run_command([docker_exe, "context", "show"], timeout)
    active_context = _clean_scalar(context_show.stdout) if context_show.returncode == 0 else ""
    inspect: CommandResult | None = None
    endpoint = ""
    if active_context:
        inspect = run_command(
            [
                docker_exe,
                "context",
                "inspect",
                active_context,
                "--format",
                "{{json .Endpoints.docker.Host}}",
            ],
            timeout,
        )
        if inspect.returncode == 0:
            endpoint = _clean_scalar(inspect.stdout)
    endpoint_lower = endpoint.lower()
    remote_endpoint = endpoint_lower.startswith(
        ("tcp://", "ssh://", "http://", "https://")
    )
    remote_override = bool(docker_host)
    return {
        "ok": not remote_override and not remote_endpoint,
        "docker_host_env_set": remote_override,
        "docker_context_env": docker_context_env or None,
        "active_context": active_context or None,
        "endpoint": endpoint or None,
        "remote_endpoint": remote_endpoint,
        "context_show": context_show.as_dict(),
        "context_inspect": inspect.as_dict() if inspect is not None else None,
    }


def _host_resource_check(path: Path, *, min_free_disk_gb: float) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    return {
        "path": str(path),
        "free_disk_gb": round(free_gb, 2),
        "min_free_disk_gb": min_free_disk_gb,
        "disk_ok": free_gb >= min_free_disk_gb,
        "note": "SWE-bench Docker guidance commonly requires large local image/cache space.",
    }


def build_report(
    *,
    grader_repo: Path = DEFAULT_GRADER_REPO,
    timeout: float = 10.0,
    expected_grader_remote: str | None = DEFAULT_EXPECTED_GRADER_REMOTE,
    expected_grader_commit: str | None = DEFAULT_EXPECTED_GRADER_COMMIT,
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB,
) -> dict[str, Any]:
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    resource_check = _host_resource_check(REPO_ROOT, min_free_disk_gb=min_free_disk_gb)
    checks["host_resources"] = resource_check
    if not resource_check["disk_ok"]:
        blockers.append("host_disk_below_swebench_minimum")

    docker_exe = shutil.which("docker")
    checks["docker_cli"] = {
        "available": docker_exe is not None,
        "path": docker_exe,
    }
    if docker_exe is None:
        blockers.append("docker_cli_missing")
        checks["docker_daemon"] = {
            "ok": False,
            "reason": "docker CLI not found on PATH",
        }
        checks["docker_ostype"] = {
            "ok": False,
            "reason": "not checked because docker CLI is missing",
        }
    else:
        docker_context = _docker_context_check(docker_exe, timeout)
        checks["docker_context"] = docker_context
        if not docker_context["ok"]:
            blockers.append("docker_remote_context_unverified")
        daemon = run_command(
            [docker_exe, "info", "--format", "{{json .ServerVersion}}"],
            timeout,
        )
        server_version = _clean_scalar(daemon.stdout)
        daemon_ok = daemon.returncode == 0 and bool(server_version)
        checks["docker_daemon"] = {
            **daemon.as_dict(),
            "ok": daemon_ok,
            "server_version": server_version or None,
        }
        if not daemon_ok:
            blockers.append("docker_daemon_unreachable")
            checks["docker_ostype"] = {
                "ok": False,
                "reason": "not checked because docker daemon is unreachable",
            }
        else:
            ostype = run_command(
                [docker_exe, "info", "--format", "{{.OSType}}"],
                timeout,
            )
            docker_ostype = _clean_scalar(ostype.stdout).lower()
            linux_ok = ostype.returncode == 0 and docker_ostype == "linux"
            checks["docker_ostype"] = {
                **ostype.as_dict(),
                "ok": linux_ok,
                "ostype": docker_ostype or None,
            }
            if not linux_ok:
                blockers.append("docker_engine_not_linux")
            else:
                smoke = run_command(
                    [docker_exe, "run", "--rm", "hello-world"],
                    max(timeout, 60.0),
                )
                smoke_ok = smoke.returncode == 0
                checks["docker_run_smoke"] = {
                    **smoke.as_dict(),
                    "ok": smoke_ok,
                    "image": "hello-world",
                }
                if not smoke_ok:
                    blockers.append("docker_run_smoke_failed")

    grader_check = _git_check(
        grader_repo,
        timeout,
        expected_remote=expected_grader_remote,
        expected_commit=expected_grader_commit,
    )
    checks["grader_repo"] = grader_check
    if not grader_check["exists"]:
        blockers.append("grader_repo_missing")
    elif not grader_check["eval_script_exists"]:
        blockers.append("grader_eval_script_missing")
    elif grader_check["remote_matches_expected"] is False:
        blockers.append("grader_repo_remote_mismatch")
    elif grader_check["commit_matches_expected"] is False:
        blockers.append("grader_repo_commit_mismatch")
    elif grader_check["dirty_status"]:
        blockers.append("grader_repo_dirty")

    modal_exe = shutil.which("modal")
    checks["modal_cli"] = {
        "available": modal_exe is not None,
        "path": modal_exe,
        "note": "remote grading candidate only",
    }
    if modal_exe is None:
        checks["modal_token"] = {
            "ok": False,
            "reason": "not checked because modal CLI is missing",
        }
    else:
        modal_token = run_command([modal_exe, "token", "info"], timeout)
        checks["modal_token"] = {
            **modal_token.as_dict(),
            "ok": modal_token.returncode == 0,
            "note": "auth only; SWE-bench Pro job wiring is not proven",
        }

    if not blockers:
        decision = "READY_LOCAL_DOCKER"
    elif any(b.startswith("docker_") for b in blockers):
        decision = "NO_GO_LOCAL_DOCKER"
    elif "grader_repo_dirty" in blockers:
        decision = "NO_GO_GRADER_REPO_DIRTY"
    else:
        decision = "NO_GO_GRADER_REPO"

    head = run_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], timeout)
    status = run_command(["git", "-C", str(REPO_ROOT), "status", "--short"], timeout)

    return {
        "schema_version": "swebench_pro_grader_preflight_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "path": str(REPO_ROOT),
            "commit": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        },
        "grader_repo": str(grader_repo),
        "local_grading_ready": decision == "READY_LOCAL_DOCKER",
        "decision": decision,
        "blockers": blockers,
        "checks": checks,
        "next_action": (
            "Do not launch or label SWE-bench Pro N=5 as official locally until "
            "decision is READY_LOCAL_DOCKER. Generate-only traces may continue "
            "if marked ungraded. Otherwise move grading to a clean Linux Docker "
            "runner with sufficient disk/RAM or a Modal setup with token auth "
            "and job wiring verified."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grader-repo",
        type=Path,
        default=DEFAULT_GRADER_REPO,
        help="Path to scaleapi/SWE-bench_Pro-os checkout",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Command timeout in seconds",
    )
    parser.add_argument(
        "--expected-grader-remote",
        default=DEFAULT_EXPECTED_GRADER_REMOTE,
        help="Expected grader repository remote, normalized before comparison",
    )
    parser.add_argument(
        "--expected-grader-commit",
        default=DEFAULT_EXPECTED_GRADER_COMMIT,
        help="Expected grader commit SHA. Pass an empty string only for exploratory runs.",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=DEFAULT_MIN_FREE_DISK_GB,
        help="Minimum free disk space required before READY_LOCAL_DOCKER",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the JSON report to stdout",
    )
    args = parser.parse_args(argv)

    report = build_report(
        grader_repo=args.grader_repo,
        timeout=args.timeout,
        expected_grader_remote=args.expected_grader_remote or None,
        expected_grader_commit=args.expected_grader_commit or None,
        min_free_disk_gb=args.min_free_disk_gb,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.json or args.output is None:
        print(encoded)
    return 0 if report["local_grading_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
