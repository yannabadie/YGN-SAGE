"""Sandbox-based evaluation stage for the Evolution Pillar.

Executes candidate code inside a Docker sandbox and returns an EvalResult.
"""
from __future__ import annotations

import logging
from typing import Callable, Awaitable

from sage.evolution.evaluator import EvalResult
from sage.sandbox.manager import SandboxManager

class SandboxEvaluator:
    """Evaluates code execution results within a secure sandbox."""

    def __init__(self, sandbox_manager: SandboxManager, test_script_template: str):
        """
        Args:
            sandbox_manager: The manager to create/destroy sandboxes.
            test_script_template: A template string containing {code} to be injected.
                                  Should output a recognizable score pattern on stdout.
        """
        self.sandbox_manager = sandbox_manager
        self.test_script_template = test_script_template
        self.logger = logging.getLogger(__name__)

    async def evaluate(self, code: str) -> EvalResult:
        """Run the mutated code in a fresh sandbox and parse its score."""
        sandbox = await self.sandbox_manager.create()
        try:
            # Prepare the test script
            script = self.test_script_template.format(code=code)
            
            # Write script to sandbox
            # In a real environment, you'd write to a file, but here we can pipe it
            # assuming it's a python script
            cmd = f'python3 -c "{_shell_escape(script)}"'
            
            result = await sandbox.execute(cmd)
            
            if result.timed_out:
                return EvalResult(score=0.0, passed=False, stage="sandbox", error="Execution timed out")
            if result.exit_code != 0:
                return EvalResult(score=0.0, passed=False, stage="sandbox", error=f"Exit code {result.exit_code}: {result.stderr}")

            # Parse score from stdout (e.g. expecting "SCORE: 0.95")
            score = 0.0
            passed = True
            try:
                for line in result.stdout.split("\n"):
                    if line.startswith("SCORE:"):
                        score = float(line.replace("SCORE:", "").strip())
            except Exception as e:
                return EvalResult(score=0.0, passed=False, stage="sandbox", error=f"Failed to parse score: {e}")

            return EvalResult(
                score=score,
                passed=passed,
                stage="sandbox",
                details={"stdout": result.stdout, "stderr": result.stderr}
            )
        finally:
            await self.sandbox_manager.destroy(sandbox.id)

def _shell_escape(s: str) -> str:
    """Escape a string for safe shell usage."""
    return s.replace('"', '"').replace('$', '\$').replace('`', '`')
