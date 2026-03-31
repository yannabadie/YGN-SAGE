"""Isolated sandbox executor using bubblewrap (bwrap) on Linux.

Falls back to plain subprocess on non-Linux platforms with a warning.

Security: On Linux (RunPod H100, WSL2), wraps execution with bwrap
for namespace isolation, read-only filesystem, and die-with-parent.
On Windows/Mac, logs a warning and delegates to plain subprocess.
"""
import logging
import os
import platform
import shutil
import subprocess
import tempfile

from sage._python import PYTHON

log = logging.getLogger("isolated_executor")

BWRAP_AVAILABLE = platform.system() == "Linux" and shutil.which("bwrap") is not None


def execute_isolated(code: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute Python code in an isolated sandbox.

    On Linux with bwrap available:
      - Unshares all namespaces (pid, net, ipc, uts, cgroup)
      - Mounts root filesystem read-only
      - Provides writable /tmp and /run via tmpfs
      - Kills child if parent dies (die-with-parent)

    On other platforms:
      - Falls back to plain subprocess with timeout only
      - Logs a warning on first invocation

    Returns:
        (stdout, stderr, returncode) tuple.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False,
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        if BWRAP_AVAILABLE:
            cmd = [
                "bwrap",
                "--unshare-all",        # Unshare all namespaces
                "--ro-bind", "/", "/",  # Read-only root
                "--tmpfs", "/tmp",      # Writable /tmp
                "--tmpfs", "/run",      # Writable /run
                "--proc", "/proc",      # Mount /proc
                "--dev", "/dev",        # Mount /dev
                "--die-with-parent",    # Kill if parent dies
                "--",
                PYTHON, script_path,
            ]
            log.debug("Executing in bwrap sandbox: %s", script_path)
        else:
            cmd = [PYTHON, script_path]
            if not hasattr(execute_isolated, "_warned"):
                log.warning(
                    "bwrap not available on %s — running without OS-level isolation",
                    platform.system(),
                )
                execute_isolated._warned = True

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout expired", -1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
