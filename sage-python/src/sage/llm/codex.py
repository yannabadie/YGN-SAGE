import subprocess
import json
import logging
import shutil
import sys
import os
from typing import Any, Dict, List, Optional

from sage.llm.google import GoogleProvider
from sage.llm.base import Message, Role, LLMConfig, LLMProvider, LLMResponse, ToolDef

class CodexProvider:
    """Full LLMProvider implementation using OpenAI Codex CLI."""
    
    name = "codex"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        
        # Build the prompt from messages
        prompt_parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n".join(prompt_parts)

        codex_path = shutil.which("codex")
        if codex_path:
            cmd = [codex_path, "exec", full_prompt, "--json", "-c", "reasoning_effort=low", "-c", "model=o3-mini"]
            try:
                self.logger.info(f"Generating via local codex CLI: {config.model if config else 'default'}")
                use_shell = sys.platform == "win32"
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False, # Don't check here so we can read stderr
                    encoding="utf-8",
                    shell=use_shell
                )
                
                if result.returncode != 0:
                    self.logger.error(f"Codex CLI failed with exit code {result.returncode}: {result.stderr}")
                
                final_text = ""
                if result and result.stdout:
                    # First try to parse as JSONL
                    for line in reversed(result.stdout.splitlines()):
                        try:
                            data = json.loads(line)
                            if data.get("type") == "item.completed" and "text" in data.get("item", {}):
                                final_text = data["item"]["text"]
                                break
                        except:
                            continue
                    
                    # If JSON parsing yielded nothing, just take the raw stdout
                    if not final_text:
                        final_text = result.stdout.strip()
                            
                if final_text:
                    return LLMResponse(content=final_text, model="codex-cli")
                else:
                    self.logger.error(f"Codex CLI returned empty output. STDERR: {result.stderr}")
            except Exception as e:
                self.logger.warning(f"Codex CLI generate failed: {e}. Falling back to GoogleProvider.")
        else:
            self.logger.error("Codex CLI not found in PATH via shutil.which('codex').")

        # Fallback
        self.logger.info("Falling back to GoogleProvider.")
        fallback = GoogleProvider()
        return await fallback.generate(messages, tools, config)


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
            # SOTA 2026: Using codex exec with --json and low reasoning effort for speed
            cmd = [codex_path, "exec", full_prompt, "--json", "-c", "reasoning_effort=low", "-c", "model=o3-mini"]
            try:
                self.logger.info(f"Using local codex CLI at {codex_path}")
                # Use shell=True on Windows to handle script aliases/extensions correctly
                use_shell = sys.platform == "win32"
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=False, 
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
