"""Native teacher adapters for OpenAI-compatible, Gemini, and Claude APIs.

DeepSeek and OpenRouter use the OpenAI-compatible client with their own keys
and endpoints. Every backend returns the same locally validated JSON response.
SDKs are loaded only for the selected provider, and clients close when a
generation run ends.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Protocol

from pydantic import Field, field_validator

from tunelm.schemas import Record

DEFAULT_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "deepseek": ("DEEPSEEK_API_KEY", None),
    "openrouter": ("OPENROUTER_API_KEY", None),
}
DEFAULT_URL = {
    "deepseek": "https://api.deepseek.com",
    "openrouter": "https://openrouter.ai/api/v1",
}


class TeacherConfig(Record):
    """Provider selection. The model id is set on the profile; API keys stay in the environment."""

    provider: Literal["openai", "gemini", "anthropic", "deepseek", "openrouter"] = "openai"
    model: str | None = Field(default=None, min_length=1)
    model_env: str | None = None
    key_env: str | None = None
    base_url_env: str | None = None
    max_tokens: int = Field(default=4096, ge=128, le=65536)
    http_retries: int = Field(default=2, ge=0, le=5)
    thinking_level: Literal["low", "medium", "high"] | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value):
        return "anthropic" if value == "claude" else value

    @field_validator("model")
    @classmethod
    def valid_model(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Model names cannot be blank")
        return value.strip() if value is not None else None


class TeacherModel(TeacherConfig):
    """One named model profile and its fraction of the accepted dataset."""

    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    fraction: float = Field(ge=0, le=1)


class TeacherBackend(Protocol):
    """Own a provider client; return response JSON or a useful generation failure."""

    last_usage: dict | None
    system: str
    response_model: type[Record]

    def generate(self, model: str, user: str, max_tokens: int) -> str: ...
    def close(self) -> None: ...


class OpenAIBackend:
    """Use the existing JSON-mode OpenAI-compatible endpoint."""

    def __init__(self, key: str, base_url: str | None, config: TeacherConfig):
        from openai import OpenAI

        self.last_usage = None
        self.client = OpenAI(
            api_key=key, base_url=base_url, timeout=120, max_retries=config.http_retries
        )
        # OpenAI renamed the limit. DeepSeek and OpenRouter still expect max_tokens.
        self.token_arg = (
            "max_completion_tokens" if config.provider == "openai" else "max_tokens"
        )

    def generate(self, model: str, user: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": self.system
                    + "\nJSON output schema:\n"
                    + json.dumps(self.response_model.model_json_schema()),
                },
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            **{self.token_arg: max_tokens},
        )
        self.last_usage = response.usage.model_dump(mode="json") if response.usage else None
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("OpenAI teacher response was incomplete or blocked")
        return response.choices[0].message.content or ""

    def close(self) -> None:
        self.client.close()


class GeminiBackend:
    """Use the Gemini Developer API with a native JSON Schema response."""

    def __init__(self, key: str, base_url: str | None, config: TeacherConfig):
        from google import genai
        from google.genai import types

        self.last_usage = None
        self.thinking_level = config.thinking_level
        self.client = genai.Client(
            api_key=key,
            vertexai=False,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=120000,
                retry_options=types.HttpRetryOptions(attempts=config.http_retries + 1),
            ),
        )

    def generate(self, model: str, user: str, max_tokens: int) -> str:
        from google.genai import types

        schema = gemini_schema(self.response_model.model_json_schema())
        response = self.client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=self.system,
                response_mime_type="application/json",
                response_json_schema=schema,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level
                else None,
            ),
        )
        self.last_usage = (
            response.usage_metadata.model_dump(mode="json") if response.usage_metadata else None
        )
        if (
            not response.candidates
            or response.candidates[0].finish_reason != types.FinishReason.STOP
        ):
            raise ValueError("Gemini teacher response was incomplete or blocked")
        return response.text or ""

    def close(self) -> None:
        self.client.close()


class AnthropicBackend:
    """Use Claude Messages structured output, retaining local constraint validation."""

    def __init__(self, key: str, base_url: str | None, config: TeacherConfig):
        from anthropic import Anthropic

        self.last_usage = None
        self.client = Anthropic(
            api_key=key, base_url=base_url, timeout=120, max_retries=config.http_retries
        )

    def generate(self, model: str, user: str, max_tokens: int) -> str:
        from anthropic import transform_schema

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=self.system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": transform_schema(self.response_model)}
            },
        )
        self.last_usage = response.usage.model_dump(mode="json") if response.usage else None
        if response.stop_reason != "end_turn":
            raise ValueError("Claude teacher response was incomplete or blocked")
        return "".join(block.text for block in response.content if block.type == "text")

    def close(self) -> None:
        self.client.close()


BACKENDS = {
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
    "anthropic": AnthropicBackend,
    "deepseek": OpenAIBackend,
    "openrouter": OpenAIBackend,
}


def gemini_schema(value):
    """Translate nested local contracts to Gemini's supported JSON Schema subset."""
    if isinstance(value, list):
        return [gemini_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {k: gemini_schema(v) for k, v in value.items() if k not in {"default", "minLength"}}
    if "const" in result:
        result["enum"] = [result.pop("const")]
    if "prefixItems" in result:
        items = result.pop("prefixItems")
        result["items"] = items[0] if all(item == items[0] for item in items) else {"anyOf": items}
    return result


class Teacher:
    """Select one backend per run while preserving a shared prompt/JSON contract."""

    def __init__(
        self,
        config: TeacherConfig,
        *,
        model_id: str | None = None,
        system: str,
        response_model: type[Record],
    ):
        self.id = model_id
        self.provider = config.provider
        named = os.environ.get(config.model_env, "") if config.model_env else ""
        self.model = (config.model or named).strip()
        default_key, default_base = DEFAULT_ENV[self.provider]
        key_env = config.key_env or default_key
        key = os.environ.get(key_env)
        if not self.model or not key:
            raise ValueError(f"Set a teacher model and {key_env} for provider {self.provider}")
        base_env = config.base_url_env or default_base
        base_url = os.environ.get(base_env) if base_env else DEFAULT_URL.get(self.provider)
        self.max_tokens = config.max_tokens
        if config.thinking_level and config.provider != "gemini":
            raise ValueError("thinking_level is supported only by the Gemini backend")
        self.backend: TeacherBackend = BACKENDS[self.provider](key, base_url, config)
        self.backend.system = system
        self.backend.response_model = response_model
        self.response_model = response_model

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model}"

    def generate(self, payload: dict, feedback: list[str]) -> Record:
        user = json.dumps({**payload, "feedback": feedback})
        self.backend.last_usage = None
        raw = self.backend.generate(self.model, user, self.max_tokens)
        return self.response_model.model_validate_json(raw)

    @property
    def last_usage(self) -> dict | None:
        """Return native billed-token metadata for the most recent attempt, when supplied."""
        return self.backend.last_usage

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.backend.close()
