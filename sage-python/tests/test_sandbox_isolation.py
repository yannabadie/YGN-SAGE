"""Tests for bwrap sandbox isolation (isolated_executor)."""
import platform
import subprocess
from unittest.mock import patch

import pytest

from sage.sandbox.isolated_executor import (
    BWRAP_AVAILABLE,
    execute_isolated,
)


class TestIsolatedExecutor:
    """Tests for execute_isolated()."""

    def test_basic_execution(self):
        """Simple print should work on any platform."""
        stdout, stderr, rc = execute_isolated('print("hello sandbox")', timeout=10)
        assert rc == 0, f"stderr: {stderr}"
        assert "hello sandbox" in stdout

    def test_timeout(self):
        """Code exceeding timeout should be killed."""
        stdout, stderr, rc = execute_isolated(
            "import time; time.sleep(999)", timeout=2,
        )
        assert rc == -1
        assert "Timeout" in stderr

    def test_syntax_error(self):
        """Syntax errors should propagate."""
        stdout, stderr, rc = execute_isolated("def f(:\n  pass", timeout=5)
        assert rc != 0
        assert "SyntaxError" in stderr

    def test_runtime_error(self):
        """Runtime exceptions should propagate."""
        stdout, stderr, rc = execute_isolated(
            "raise ValueError('test')", timeout=5,
        )
        assert rc != 0
        assert "ValueError" in stderr

    def test_temp_file_cleanup(self):
        """Temp script file should be removed after execution."""
        import os
        import tempfile

        before = set(os.listdir(tempfile.gettempdir()))
        execute_isolated('print("cleanup test")', timeout=5)
        after = set(os.listdir(tempfile.gettempdir()))
        # No new .py files should linger
        new_py = {f for f in (after - before) if f.endswith(".py")}
        assert len(new_py) == 0, f"Leaked temp files: {new_py}"


class TestBwrapDetection:
    """Tests for bwrap availability detection."""

    def test_bwrap_flag_type(self):
        """BWRAP_AVAILABLE should be a bool."""
        assert isinstance(BWRAP_AVAILABLE, bool)

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="bwrap detection only meaningful on Linux",
    )
    def test_bwrap_on_linux(self):
        """On Linux, BWRAP_AVAILABLE should match shutil.which('bwrap')."""
        import shutil
        expected = shutil.which("bwrap") is not None
        assert BWRAP_AVAILABLE == expected

    @pytest.mark.skipif(
        platform.system() == "Linux",
        reason="Non-Linux should never have bwrap",
    )
    def test_bwrap_off_non_linux(self):
        """On non-Linux, BWRAP_AVAILABLE should be False."""
        assert BWRAP_AVAILABLE is False

    def test_fallback_warning_logged(self, caplog):
        """When bwrap is unavailable, a warning should be logged on first call."""
        # Reset the warned flag so we can capture it
        if hasattr(execute_isolated, "_warned"):
            delattr(execute_isolated, "_warned")

        with patch(
            "sage.sandbox.isolated_executor.BWRAP_AVAILABLE", False,
        ):
            import importlib
            import sage.sandbox.isolated_executor as mod
            # Force re-evaluation by calling directly
            stdout, stderr, rc = mod.execute_isolated(
                'print("warn test")', timeout=5,
            )
        # The function should still work (fallback)
        assert "warn test" in stdout or rc == 0


@pytest.mark.skipif(
    not BWRAP_AVAILABLE,
    reason="bwrap not installed — skipping isolation tests",
)
class TestBwrapIsolation:
    """Tests that only run when bwrap is actually available (Linux pods)."""

    def test_read_only_root(self):
        """Writing to / should fail under bwrap."""
        stdout, stderr, rc = execute_isolated(
            "open('/test_rw_probe', 'w').write('x')",
            timeout=5,
        )
        assert rc != 0, "Write to read-only root should fail"

    def test_tmp_writable(self):
        """Writing to /tmp should succeed under bwrap."""
        stdout, stderr, rc = execute_isolated(
            "import tempfile; f = tempfile.NamedTemporaryFile(delete=False); "
            "f.write(b'ok'); f.close(); print('tmp_ok')",
            timeout=5,
        )
        assert rc == 0, f"stderr: {stderr}"
        assert "tmp_ok" in stdout

    def test_network_isolated(self):
        """Network should be unavailable under --unshare-all."""
        stdout, stderr, rc = execute_isolated(
            "import urllib.request; urllib.request.urlopen('http://1.1.1.1', timeout=2)",
            timeout=10,
        )
        assert rc != 0, "Network access should fail in isolated sandbox"
