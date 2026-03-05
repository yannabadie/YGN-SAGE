"""OpenAI Codex CLI provider (gpt-5.3-codex default, configurable).

Wraps the `codex exec` command for non-interactive agent calls.
Supports structured JSON output via --output-schema.
Falls back to GoogleProvider if the CLI is missing or fails.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from sage.llm.base import LLMConfig, LLMResponse, Message, Role, ToolDef
from sage.llm.google import GoogleProvider

logger = logging.getLogger(__name__)

# Default model (matches ~/.codex/config.toml)
DEFAULT_MODEL = "gpt-5.3-codex"


def _ensure_additional_properties_false(schema: dict) -> dict:
    """Recursively add additionalProperties: false to all objects in a JSON schema.

    Required by OpenAI structured output — see
    https://developers.openai.com/api/docs/guides/structured-outputs
    """
    schema = copy.deepcopy(schema)
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                _ensure_additional_properties_false(prop)
        # Handle nested definitions ($defs / definitions)
        for defn in schema.get("$defs", {}).values():
            if isinstance(defn, dict):
                _ensure_additional_properties_false(defn)
    if "items" in schema and isinstance(schema["items"], dict):
        _ensure_additional_properties_false(schema["items"])
    return schema


def _extract_text_from_jsonl(stdout: str) -> str:
    """Parse Codex CLI JSONL output and return the final agent message text.

    The CLI emits lines like:
        {"type":"item.completed","item":{"id":"...","type":"agent_message","text":"hello"}}
        {"type":"turn.completed","usage":{...}}
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # Primary format: item.completed with text field
            if data.get("type") == "item.completed":
                item = data.get("item", {})
                if "text" in item:
                    return item["text"]
        except (json.JSONDecodeError, KeyError):
            continue
    return ""


class CodexProvider:
    """LLMProvider using OpenAI Codex CLI (codex exec).

    Default model: gpt-5.3-codex (from ~/.codex/config.toml).
    Override via LLMConfig.model or the -m flag.
    """

    name = "codex"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]] = None,
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")

        full_prompt = "\n".join(prompt_parts)
        model = config.model if config and config.model else DEFAULT_MODEL

        codex_path = shutil.which("codex")
        if not codex_path:
            self.logger.warning("Codex CLI not in PATH. Falling back to GoogleProvider.")
            return await self._fallback(messages, tools, config)

        # Build command
        cmd = [codex_path, "exec", full_prompt, "--json"]

        # Only pass -m if a non-default model is requested
        if model != DEFAULT_MODEL:
            cmd.extend(["-m", model])

        # Reasoning effort from config extra, default to "low" for speed
        effort = "low"
        if config and config.extra.get("reasoning_effort"):
            effort = config.extra["reasoning_effort"]
        cmd.extend(["-c", f"model_reasoning_effort={effort}"])

        # Structured output via --output-schema temp file
        schema_tmpfile = None
        if config and config.json_schema is not None:
            schema = config.json_schema
            if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
                schema = schema.model_json_schema()
            # OpenAI requires additionalProperties: false at every object level
            schema = _ensure_additional_properties_false(schema)
            try:
                schema_tmpfile = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                json.dump(schema, schema_tmpfile)
                schema_tmpfile.close()
                cmd.extend(["--output-schema", schema_tmpfile.name])
            except Exception as e:
                self.logger.warning(f"Failed to write schema temp file: {e}")

        try:
            self.logger.info(f"Codex CLI: model={model} effort={effort}")
            use_shell = sys.platform == "win32"
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                shell=use_shell,
                timeout=120,
            )

            if result.returncode != 0:
                self.logger.error(
                    f"Codex CLI exit {result.returncode}: {result.stderr[:500]}"
                )

            final_text = ""
            if result.stdout:
                final_text = _extract_text_from_jsonl(result.stdout)
                # Fallback: raw stdout if JSONL parsing found nothing
                if not final_text:
                    final_text = result.stdout.strip()

            if final_text:
                return LLMResponse(content=final_text, model=model)
            else:
                self.logger.error(f"Codex CLI empty output. STDERR: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            self.logger.warning("Codex CLI timed out (120s). Falling back to Google.")
        except Exception as e:
            self.logger.warning(f"Codex CLI failed: {e}. Falling back to Google.")
        finally:
            if schema_tmpfile is not None:
                try:
                    os.unlink(schema_tmpfile.name)
                except OSError:
                    pass

        return await self._fallback(messages, tools, config)

    async def _fallback(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]],
        config: Optional[LLMConfig],
    ) -> LLMResponse:
        """Fall back to GoogleProvider."""
        self.logger.info("Falling back to GoogleProvider.")
        fallback = GoogleProvider()
        return await fallback.generate(messages, tools, config)


class CodexExecProvider:
    """High-effort Codex CLI interface for code review tasks.

    Uses reasoning_effort=xhigh for quality evaluation.
    Falls back to Gemini 3.1 Pro if the CLI is unavailable.
    """

    def __init__(self, effort: str = "xhigh"):
        self.effort = effort
        self.logger = logging.getLogger(__name__)

    async def review_code(self, code: str, objective: str) -> Dict[str, Any]:
        """Review code structure and SOTA alignment."""
        task = (
            f"Review this code for the following objective: {objective}\n"
            "Check for:\n"
            "1. SOTA alignment (SIMD, Cache locality, Branchless).\n"
            "2. Functional correctness and edge cases.\n"
            "3. Logical performance gains over a standard NumPy baseline.\n\n"
            "Provide a structured review in JSON format with a structural_score "
            "(0.0 to 1.0) and technical_comments."
        )
        full_prompt = task + f"\n\nCODE TO REVIEW:\n```python\n{code}\n```"

        codex_path = shutil.which("codex")
        if codex_path:
            cmd = [
                codex_path, "exec", full_prompt,
                "--json",
                "-c", f"model_reasoning_effort={self.effort}",
            ]
            try:
                self.logger.info(f"Codex CLI review (effort={self.effort})")
                use_shell = sys.platform == "win32"
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    encoding="utf-8",
                    shell=use_shell,
                    timeout=180,
                )
                if result.stdout:
                    text = _extract_text_from_jsonl(result.stdout)
                    if text and "{" in text:
                        return json.loads(text[text.find("{"):text.rfind("}") + 1])
            except Exception as e:
                self.logger.warning(f"Codex CLI review failed: {e}")

        # Fallback to Gemini 3.1 Pro
        try:
            self.logger.info("Falling back to Gemini 3.1 Pro for review.")
            provider = GoogleProvider()
            cfg = LLMConfig(
                provider="google",
                model="gemini-3.1-pro-preview",
                temperature=0.1,
            )
            response = await provider.generate(
                [
                    Message(role=Role.SYSTEM, content="You are a SOTA code reviewer. Respond in valid JSON."),
                    Message(role=Role.USER, content=full_prompt),
                ],
                config=cfg,
            )
            text = response.content
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            return {"structural_score": 0.5, "technical_comments": text[:200]}
        except Exception as e:
            self.logger.error(f"Gemini fallback failed: {e}")
            return {"error": str(e), "structural_score": 0.0}
