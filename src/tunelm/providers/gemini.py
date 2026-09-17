from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from tunelm.providers.base import LLMProvider, Message


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install provider extras: pip install -e '.[providers]'") from exc
        self.model = model
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def generate(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if response_schema else None,
            response_json_schema=response_schema,
        )
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        return response.text or ""
