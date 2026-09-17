"""OpenAI Responses API provider with strict JSON-schema compatibility."""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from typing import Any

from tunelm.providers.base import LLMProvider, Message


def openai_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic JSON Schema for OpenAI strict structured output."""
    schema = copy.deepcopy(schema)

    def walk(node: dict[str, Any]) -> None:
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties)
            node["additionalProperties"] = False
            for value in properties.values():
                if isinstance(value, dict):
                    walk(value)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items)
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for value in defs.values():
                if isinstance(value, dict):
                    walk(value)

    def strip(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            for value in node.values():
                strip(value)
        elif isinstance(node, list):
            for value in node:
                strip(value)

    walk(schema)
    strip(schema)
    return schema


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install provider extras: pip install -e '.[providers]'") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [dict(message) for message in messages],
            "max_output_tokens": max_tokens,
        }
        if response_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "tunelm_response",
                    "strict": True,
                    "schema": openai_strict_json_schema(response_schema),
                }
            }

        from openai import BadRequestError

        attempts = (
            {**kwargs, "temperature": temperature},
            kwargs,
        )
        for attempt_kwargs in attempts:
            try:
                return self.client.responses.create(**attempt_kwargs).output_text
            except BadRequestError as exc:
                message = str(exc).casefold()
                if "temperature" in message and attempt_kwargs is attempts[0]:
                    continue
                raise
        raise RuntimeError("OpenAI request failed")
