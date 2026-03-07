"""Google Gemini provider."""
from __future__ import annotations

import os
import logging
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolDef

logger = logging.getLogger(__name__)

class GoogleProvider:
    """LLM provider for Google Gemini models with native grounding."""

    name = "google"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def capabilities(self) -> dict[str, bool]:
        """Declare what this provider actually supports."""
        return {
            "structured_output": True,
            "tool_role": True,
            "file_search": True,
            "grounding": True,
            "system_prompt": True,
            "streaming": True,
        }

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
        use_google_search: bool = True,
        file_search_store_names: list[str] | None = None,
    ) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install 'ygn-sage[google]'")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Set it via environment variable or pass api_key= to GoogleProvider."
            )
        client = genai.Client(api_key=self.api_key)

        # Default model
        model = "gemini-3.1-pro-preview"
        if config and config.model:
            model = config.model
            
        logger.info(f"Using Google Gemini model: {model}")

        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        has_tool_role_rewrite = False
        for msg in messages:
            if msg.role.value == "system":
                system_instruction = msg.content
            else:
                if msg.role.value == "tool":
                    has_tool_role_rewrite = True
                role = "model" if msg.role.value == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))
        if has_tool_role_rewrite:
            logger.warning(
                "GoogleProvider: rewriting 'tool' role to 'user' for Gemini API "
                "(semantic loss — tool results will appear as user messages)",
            )

        # Configure grounding tools (google_search and file_search are mutually exclusive)
        gemini_tools = []
        if file_search_store_names:
            try:
                gemini_tools.append(types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=file_search_store_names
                    )
                ))
            except (AttributeError, TypeError) as e:
                logger.warning(f"FileSearch tool injection failed: {e}")
        elif use_google_search:
            gemini_tools.append(types.Tool(google_search=types.GoogleSearch()))

        # Structured JSON output support
        response_mime_type = None
        response_json_schema = None
        if config and config.json_schema is not None:
            schema = config.json_schema
            # If it's a Pydantic model class, extract its JSON schema
            if isinstance(schema, type) and hasattr(schema, 'model_json_schema'):
                schema = schema.model_json_schema()
            response_mime_type = 'application/json'
            response_json_schema = schema
            # Gemini API: structured output (response_mime_type) is incompatible with tools
            gemini_tools = []

        generate_config = types.GenerateContentConfig(
            max_output_tokens=config.max_tokens if config else None,
            temperature=config.temperature if config else 0.1,
            system_instruction=system_instruction,
            tools=gemini_tools if gemini_tools else None,
            response_mime_type=response_mime_type,
            response_schema=response_json_schema,
        )

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=generate_config,
            )
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise e

        # Extract grounding metadata if available
        grounding_metadata = getattr(response, 'grounding_metadata', None)
        content_text = response.text or ""
        
        if grounding_metadata and getattr(grounding_metadata, 'search_entry_point', None):
            logger.info("Grounding sources detected in response.")

        return LLMResponse(
            content=content_text,
            tool_calls=[],
            model=model,
        )
