"""eBPF-based evaluation stage for the Evolution Pillar.

Executes candidate code (compiled to eBPF/ELF) inside a Rust-backed
solana_rbpf VM in <1ms. Essential for the ASI <1ms execution mandate.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sage.evolution.evaluator import EvalResult, Evaluator
try:
    import sage_core
except ImportError:
    import types as _types
    sage_core = _types.ModuleType("sage_core")

class EbpfEvaluator(Evaluator):
    """Evaluates eBPF byte-code using the sage_core Rust extension."""

    def __init__(self, execution_timeout_ms: int = 50):
        self.execution_timeout_ms = execution_timeout_ms
        self.logger = logging.getLogger(__name__)

    async def evaluate(self, code: str | bytes) -> EvalResult:
        """Run the provided ELF bytes or raw bytecode in the eBPF sandbox.
        
        Args:
            code: The ELF binary bytes or raw eBPF bytecode.
        """
        sandbox = sage_core.EbpfSandbox()
        
        try:
            start_time = time.perf_counter()
            
            # SOTA 2026: Direct raw bytecode execution without C-compiler overhead
            if isinstance(code, bytes):
                # We attempt to load it as RAW text bytes for benchmark proofs
                try:
                    sandbox.load_raw(code)
                except Exception:
                    # Fallback to ELF loader if it's a full ELF
                    sandbox.load_elf(code)
            else:
                self.logger.error("EbpfEvaluator requires binary bytes.")
                return EvalResult(score=0.0, passed=False, stage="ebpf_sandbox", error="Code must be bytes")
            
            # Allocate 1MB of working memory for the VM
            mem = bytearray(1024 * 1024)
            
            # Execute with sub-millisecond precision
            instruction_count, result_code = sandbox.execute(list(mem))
            
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Convert result_code to a float score
            score = float(result_code)
            passed = (result_code > 0)

            return EvalResult(
                score=score,
                passed=passed,
                stage="ebpf_sandbox",
                details={
                    "execution_time_ms": exec_time_ms,
                    "instruction_count": instruction_count,
                    "result_code": result_code,
                    "backend": "solana_rbpf (Raw BPF/ELF)"
                }
            )

        except Exception as e:
            self.logger.error(f"eBPF execution failed: {e}")
            return EvalResult(
                score=0.0,
                passed=False,
                stage="ebpf_sandbox",
                error=str(e)
            )
