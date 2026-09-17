from __future__ import annotations

from tunelm.providers.openai_compatible import ChatCompletionsProvider


class OpenRouterProvider(ChatCompletionsProvider):
    name = "openrouter"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(
            self.name,
            model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            api_key_env="OPENROUTER_API_KEY",
        )
