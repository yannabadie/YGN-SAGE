import subprocess
import sys


def test_evolution_help():
    result = subprocess.run(
        [sys.executable, "-m", "sage.evolution", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "optimize" in result.stdout.lower() or "usage" in result.stdout.lower()
