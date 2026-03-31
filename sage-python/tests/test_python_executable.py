"""Tests for cross-platform Python interpreter path (Axis 3)."""
import os
import sys

from sage._python import PYTHON


def test_python_is_non_empty():
    assert PYTHON
    assert isinstance(PYTHON, str)
    assert len(PYTHON) > 0


def test_python_executable_exists():
    # PYTHON should point to something that exists
    assert os.path.isfile(PYTHON) or os.access(PYTHON, os.X_OK), (
        f"PYTHON={PYTHON!r} is not an executable file"
    )


def test_python_equals_sys_executable():
    # When sys.executable is set (normal case), PYTHON should equal it
    if sys.executable:
        assert PYTHON == sys.executable
