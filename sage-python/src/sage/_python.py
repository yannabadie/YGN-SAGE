"""Cross-platform Python interpreter path.

Always use PYTHON instead of hardcoded "python" or "python3" in subprocess
calls. sys.executable is the only reliable cross-platform approach --- it
works in virtualenvs, conda, pyenv, Windows Store Python, and WSL2.
"""
import shutil
import sys

PYTHON: str = (
    sys.executable
    or shutil.which("python3")
    or shutil.which("python")
    or "python3"
)
