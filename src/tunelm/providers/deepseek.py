from __future__ import annotations

from tunelm.providers.openai_compatible import ChatCompletionsProvider


class DeepSeekProvider(ChatCompletionsProvider):
    name = "deepseek"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(
            self.name,
            model,
            base_url="https://api.deepseek.com",
            api_key=api_key,
            api_key_env="DEEPSEEK_API_KEY",
        )
