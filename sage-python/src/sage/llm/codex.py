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
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, prop in properties.items():
                if isinstance(prop, dict):
                    properties[key] = _ensure_additional_properties_false(prop)

    # Handle nested definitions ($defs / definitions)
    defs = schema.get("$defs", {})
    if isinstance(defs, dict):
        for key, defn in defs.items():
            if isinstance(defn, dict):
                defs[key] = _ensure_additional_properties_false(defn)

    legacy_defs = schema.get("definitions", {})
    if isinstance(legacy_defs, dict):
        for key, defn in legacy_defs.items():
            if isinstance(defn, dict):
                legacy_defs[key] = _ensure_additional_properties_false(defn)

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _ensure_additional_properties_false(schema["items"])

    for combiner in ("allOf", "anyOf", "oneOf"):
        if combiner in schema and isinstance(schema[combiner], list):
            schema[combiner] = [
                _ensure_additional_properties_false(item) if isinstance(item, dict) else item
                for item in schema[combiner]
            ]

    if "not" in schema and isinstance(schema["not"], dict):
        schema["not"] = _ensure_additional_properties_false(schema["not"])
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
            # v1 format: item.completed with direct text field
            if data.get("type") == "item.completed":
                item = data.get("item", {})
                if "text" in item:
                    return item["text"]
                # v2+ format: content parts list
                content = item.get("content")
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_text = part.get("text")
                        if isinstance(part_text, str):
                            text_parts.append(part_text)
                    if text_parts:
                        return "".join(text_parts)
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

    def capabilities(self) -> dict[str, bool]:
        """Declare what this provider actually supports."""
        return {
            "structured_output": False,
            "tool_role": False,      # Converted to plain text
            "file_search": False,
            "grounding": False,
            "system_prompt": False,  # Codex skips system messages
            "streaming": False,
        }

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDef]] = None,
        config: Optional[LLMConfig] = None,
        file_search_store_names: Optional[List[str]] = None,
    ) -> LLMResponse:
        # Codex CLI works best with direct task instructions.
        # System messages are skipped (Codex has its own agent persona).
        if file_search_store_names:
            logger.warning(
                "CodexProvider: dropping file_search_store_names=%s "
                "(Codex CLI does not support File Search grounding — "
                "these stores will NOT be queried)",
                file_search_store_names,
            )
        parts: list[str] = []
        has_tool_msgs = False
        for msg in messages:
            if msg.role == Role.USER:
                parts.append(msg.content)
            elif msg.role == Role.ASSISTANT:
                parts.append(f"[Previous response]: {msg.content}")
            elif msg.role == Role.TOOL:
                has_tool_msgs = True
                name = getattr(msg, "name", "tool")
                parts.append(f"[Tool result ({name})]: {msg.content}")
            # System messages intentionally skipped for Codex CLI
        if has_tool_msgs:
            logger.warning(
                "CodexProvider: rewriting 'tool' role messages to plain text "
                "(semantic loss — tool results will appear as user text, "
                "not as structured tool responses)",
            )

        full_prompt = "\n\n".join(parts)

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
        """Fall back to GoogleProvider with a Gemini model (not the Codex model ID)."""
        self.logger.info("Falling back to GoogleProvider.")
        fallback = GoogleProvider()
        # Override model to a valid Gemini model — Codex model IDs don't work on Gemini
        from sage.llm.router import ModelRouter
        fallback_config = ModelRouter.get_config("fast")
        if config:
            fallback_config.temperature = config.temperature
            fallback_config.max_tokens = config.max_tokens
            fallback_config.json_schema = config.json_schema
        return await fallback.generate(messages, tools, fallback_config)


class CodexExecProvider:
    """High-effort Codex CLI interface for code review tasks.

    Uses reasoning_effort=xhigh for quality evaluation.
    Falls back to Gemini 3.1 Pro if the CLI is unavailable.
    """

    def __init__(self, effort: str = "xhigh"):
        self.effort = effort
        self.logger = logging.getLogger(__name__)

    async def review_code(self, code: str, objective: str) -> Dict[str, Any]:
        """Review code structure and performance characteristics."""
        task = (
            f"Review this code for the following objective: {objective}\n"
            "Check for:\n"
            "1. Performance patterns (SIMD, cache locality, branchless).\n"
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
                    Message(role=Role.SYSTEM, content="You are a code reviewer. Respond in valid JSON."),
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
