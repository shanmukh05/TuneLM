"""OpenAI-compatible chat-completions providers (OpenRouter, DeepSeek, etc.)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from tunelm.providers.base import LLMProvider, Message


class ChatCompletionsProvider(LLMProvider):
    """Thin wrapper around OpenAI SDK chat.completions for compatible APIs."""

    def __init__(
        self,
        name: str,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install provider extras: pip install -e '.[providers]'") from exc
        self.name = name
        self.model = model
        key = api_key or os.getenv(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is required for provider {name}")
        self.client = OpenAI(
            base_url=base_url,
            api_key=key,
            default_headers=default_headers or {},
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        payload = [dict(message) for message in messages]
        if response_schema and not any(message.get("role") == "system" for message in payload):
            payload.insert(
                0,
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not wrap the JSON in Markdown.",
                },
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
