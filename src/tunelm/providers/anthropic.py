from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from tunelm.providers.base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Install provider extras: pip install -e '.[providers]'") from exc
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if response_schema:
            system += "\nThe response must validate against this JSON Schema:\n" + json.dumps(
                response_schema
            )
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[dict(m) for m in messages if m["role"] != "system"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")
