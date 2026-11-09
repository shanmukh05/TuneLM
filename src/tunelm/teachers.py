"""Native teacher adapters for OpenAI-compatible, Gemini, and Claude APIs.

Every backend returns the same locally validated JSON response. SDKs are loaded
only for the selected provider, and clients close when a generation run ends.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Protocol

from pydantic import Field, field_validator

from tunelm.schemas import Record

SYSTEM = """Create grounded Strudel training data. Return a JSON object with ONLY
schema_version=2, prompt (a natural user request), decisions (exactly five concise
strings: tempo/key/structure; drums; bass/harmony; timbre/modulation; mixing/form),
and code (complete executable Strudel JS). Use the supplied VERIFIED composition
and registry/memory. For backtranslate, preserve the supplied code exactly; only
describe facts present in the code. For recombine, adapt the grounded composition
to the supplied plan. Honor the persona and original_code preservation. No API
inventions, no Tidal Haskell, no network. Keep decisions as brief musical choices.
Follow prompt_brief when provided; it takes priority over the persona label.
Write a plausible end-user request for MUSIC, not a description of how to implement
the supplied program. Requests may be terse, vague, mood-led, genre-led, a blend,
or about an activity or imagined scene. Do not require the user to specify tempo,
scale, instruments, or APIs. Omitted details are YOUR musical choices, stated in
decisions and code, not retroactively added to the user's request. Any details the
request does specify must match the result. Only technical prompts ask for code
methods or exact production settings. Keep each request natural and varied.
Use variation for full compositions, allow deliberate static atomic loops.
"""

DEFAULT_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
}


class TeacherConfig(Record):
    """Provider selection and environment-variable names, with no stored credentials."""

    provider: Literal["openai", "gemini", "anthropic"] = "openai"
    model: str | None = Field(default=None, min_length=1)
    model_env: str = "TUNELM_TEACHER_MODEL"
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


class TeacherReply(Record):
    """One provider-independent synthetic prompt, composition summary, and program."""

    prompt: str = Field(min_length=1)
    decisions: list[str] = Field(min_length=5, max_length=5)
    code: str = Field(min_length=1)


class TeacherBackend(Protocol):
    """Own a provider client; return response JSON or a useful generation failure."""

    last_usage: dict | None

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

    def generate(self, model: str, user: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
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

        schema = TeacherReply.model_json_schema()
        version = schema["properties"]["schema_version"]
        version["enum"] = [version.pop("const")]
        version.pop("default", None)
        # Keep wire constraints within Gemini's subset; Pydantic still validates every field.
        for name in ["prompt", "code"]:
            schema["properties"][name].pop("minLength", None)
        response = self.client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
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
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={
                "format": {"type": "json_schema", "schema": transform_schema(TeacherReply)}
            },
        )
        self.last_usage = response.usage.model_dump(mode="json") if response.usage else None
        if response.stop_reason != "end_turn":
            raise ValueError("Claude teacher response was incomplete or blocked")
        return "".join(block.text for block in response.content if block.type == "text")

    def close(self) -> None:
        self.client.close()


BACKENDS = {"openai": OpenAIBackend, "gemini": GeminiBackend, "anthropic": AnthropicBackend}


class Teacher:
    """Select one backend per run while preserving a shared prompt/JSON contract."""

    def __init__(self, config: TeacherConfig, *, model_id: str | None = None):
        self.id = model_id
        self.provider = config.provider
        self.model = (config.model or os.environ.get(config.model_env, "")).strip()
        default_key, default_base = DEFAULT_ENV[self.provider]
        key_env = config.key_env or default_key
        key = os.environ.get(key_env)
        if not self.model or not key:
            raise ValueError(f"Set a teacher model and {key_env} for provider {self.provider}")
        base_env = config.base_url_env or default_base
        base_url = os.environ.get(base_env) if base_env else None
        self.max_tokens = config.max_tokens
        if config.thinking_level and config.provider != "gemini":
            raise ValueError("thinking_level is supported only by the Gemini backend")
        self.backend: TeacherBackend = BACKENDS[self.provider](key, base_url, config)

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model}"

    def generate(self, payload: dict, feedback: list[str]) -> TeacherReply:
        user = json.dumps({**payload, "feedback": feedback})
        self.backend.last_usage = None
        raw = self.backend.generate(self.model, user, self.max_tokens)
        return TeacherReply.model_validate_json(raw)

    @property
    def last_usage(self) -> dict | None:
        """Return native billed-token metadata for the most recent attempt, when supplied."""
        return self.backend.last_usage

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.backend.close()
