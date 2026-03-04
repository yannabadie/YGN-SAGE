import subprocess
import json
import logging
import shutil
import sys
import os
from typing import Any, Dict

from sage.llm.google import GoogleProvider
from sage.llm.base import Message, Role, LLMConfig

class CodexExecProvider:
    """Interface to OpenAI Codex CLI 5.3 (SOTA 2026) with Gemini fallback.
    
    Uses 'codex exec' with high reasoning effort for code quality evaluation.
    If the CLI is missing or fails, falls back to Gemini 3.1 Pro Preview.
    """
    
    def __init__(self, effort: str = "xhigh"):
        self.effort = effort
        self.logger = logging.getLogger(__name__)

    async def review_code(self, code: str, objective: str) -> Dict[str, Any]:
        """Review code structure and SOTA alignment using Codex 5.3 or Gemini."""
        task = "Review this code for the following objective: " + objective + "\n"
        task += "Check for:\n1. SOTA alignment (SIMD, Cache locality, Branchless).\n"
        task += "2. Functional correctness and edge cases.\n"
        task += "3. Logical performance gains over a standard NumPy baseline.\n\n"
        task += "Provide a structured review in JSON format with a structural_score (0.0 to 1.0) and technical_comments."

        full_prompt = task + "\n\nCODE TO REVIEW:\n```python\n" + code + "\n```"
        
        # Try local 'codex' CLI first (if available)
        codex_path = shutil.which("codex")
        if codex_path:
            # SOTA 2026: Using codex exec with --json
            cmd = [codex_path, "exec", full_prompt, "--json"]
            try:
                self.logger.info(f"Using local codex CLI at {codex_path}")
                # Use shell=True on Windows to handle script aliases/extensions correctly
                use_shell = sys.platform == "win32"
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True, 
                    encoding="utf-8",
                    shell=use_shell
                )
                if result and result.stdout:
                    for line in reversed(result.stdout.splitlines()):
                        try:
                            data = json.loads(line)
                            if data.get("type") == "item.completed" and "text" in data.get("item", {}):
                                text = data["item"]["text"]
                                if "{" in text:
                                    return json.loads(text[text.find("{"):text.rfind("}")+1])
                        except:
                            continue
            except Exception as e:
                self.logger.warning(f"Local codex CLI failed: {e}. Falling back to Gemini.")

        # Fallback to Gemini 3.1 Pro Preview (ASI SOTA)
        try:
            self.logger.info("Using Gemini 3.1 Pro Preview for high-quality review.")
            provider = GoogleProvider()
            config = LLMConfig(provider="google", model="gemini-3.1-pro-preview", temperature=0.1)
            
            response = await provider.generate([
                Message(role=Role.SYSTEM, content="You are a SOTA AI Research Reviewer. Always respond in valid JSON."),
                Message(role=Role.USER, content=full_prompt)
            ], config=config)
            
            # Extract JSON from response
            text = response.content
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
            
            return {
                "structural_score": 0.5,
                "technical_comments": f"Extracted text: {text[:200]}"
            }
        except Exception as e:
            self.logger.error(f"Gemini fallback failed: {e}")
            return {"error": str(e), "structural_score": 0.0}
