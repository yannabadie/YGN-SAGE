"""Docker-based sandbox manager for isolated code execution."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxConfig:
    """Configuration for a sandbox instance."""
    image: str = "python:3.13-slim"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout: int = 300
    network_enabled: bool = False
    work_dir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result of executing code in a sandbox."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


class Sandbox:
    """A single sandboxed execution environment."""

    def __init__(self, sandbox_id: str, config: SandboxConfig, container_id: str | None = None):
        self.id = sandbox_id
        self.config = config
        self.container_id = container_id
        self._alive = True

    @property
    def alive(self) -> bool:
        return self._alive

    async def execute(self, command: str) -> SandboxResult:
        """Execute a command in this sandbox."""
        if not self._alive:
            return SandboxResult(stdout="", stderr="Sandbox is not alive", exit_code=1)

        if self.container_id is None:
            # Fallback: run locally via subprocess (no Docker)
            return await self._execute_local(command)

        return await self._execute_docker(command)

    async def _execute_local(self, command: str) -> SandboxResult:
        """Fallback: execute locally when Docker is not available."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return SandboxResult(stdout="", stderr="Timed out", exit_code=137, timed_out=True)

    async def _execute_docker(self, command: str) -> SandboxResult:
        """Execute in Docker container."""
        docker_cmd = f"docker exec {self.container_id} sh -c {_shell_escape(command)}"
        proc = await asyncio.create_subprocess_shell(
            docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.timeout
            )
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await asyncio.create_subprocess_shell(f"docker kill {self.container_id}")
            return SandboxResult(stdout="", stderr="Timed out", exit_code=137, timed_out=True)

    async def destroy(self) -> None:
        """Destroy this sandbox."""
        self._alive = False
        if self.container_id:
            proc = await asyncio.create_subprocess_shell(
                f"docker rm -f {self.container_id}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()


class SandboxManager:
    """Manages sandbox lifecycle: create, execute, snapshot, destroy."""

    def __init__(self, use_docker: bool = False):
        self._sandboxes: dict[str, Sandbox] = {}
        self._use_docker = use_docker

    async def create(self, config: SandboxConfig | None = None) -> Sandbox:
        """Create a new sandbox."""
        config = config or SandboxConfig()
        sandbox_id = str(uuid.uuid4())[:8]

        container_id = None
        if self._use_docker:
            container_id = await self._create_container(sandbox_id, config)

        sandbox = Sandbox(sandbox_id=sandbox_id, config=config, container_id=container_id)
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    async def get(self, sandbox_id: str) -> Sandbox | None:
        """Get a sandbox by ID."""
        return self._sandboxes.get(sandbox_id)

    async def destroy(self, sandbox_id: str) -> bool:
        """Destroy a sandbox."""
        sandbox = self._sandboxes.pop(sandbox_id, None)
        if sandbox is None:
            return False
        await sandbox.destroy()
        return True

    async def destroy_all(self) -> int:
        """Destroy all sandboxes. Returns count destroyed."""
        count = len(self._sandboxes)
        for sandbox in list(self._sandboxes.values()):
            await sandbox.destroy()
        self._sandboxes.clear()
        return count

    def list_sandboxes(self) -> list[str]:
        """List all active sandbox IDs."""
        return [sid for sid, s in self._sandboxes.items() if s.alive]

    async def _create_container(self, sandbox_id: str, config: SandboxConfig) -> str:
        """Create a Docker container for the sandbox."""
        network_flag = "" if config.network_enabled else "--network=none"
        env_flags = " ".join(f"-e {k}={v}" for k, v in config.env.items())

        cmd = (
            f"docker run -d --name sage-sandbox-{sandbox_id} "
            f"--memory={config.memory_limit} "
            f"--cpus={config.cpu_limit} "
            f"{network_flag} {env_flags} "
            f"-w {config.work_dir} "
            f"{config.image} sleep infinity"
        )

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create container: {stderr.decode()}")

        return stdout.decode().strip()


def _shell_escape(s: str) -> str:
    """Escape a string for safe shell usage."""
    return "'" + s.replace("'", "'\\''") + "'"
