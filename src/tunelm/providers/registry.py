"""Parse model entries from YAML and construct provider routers."""

from __future__ import annotations

import os
from typing import Any

from tunelm.providers.base import LLMProvider, ProviderChoice, ProviderRouter

PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def provider_api_key(provider: str) -> str | None:
    env_name = PROVIDER_API_KEYS.get(provider)
    if not env_name:
        return None
    return os.getenv(env_name)


def model_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return enabled model entries keyed by config model id."""
    if models := config.get("models"):
        return {str(model_id): dict(spec) for model_id, spec in models.items()}
    legacy: dict[str, dict[str, Any]] = {}
    for provider_name, values in config.get("providers", {}).items():
        legacy[str(provider_name)] = {
            "enabled": values.get("enabled", False),
            "provider": provider_name,
            "model": values.get("model"),
        }
    return legacy


def model_name(spec: dict[str, Any]) -> str:
    name = spec.get("model") or spec.get("model_name")
    if not name:
        raise ValueError("each model entry requires `model` or `model_name`")
    return str(name)


def create_provider(model_id: str, spec: dict[str, Any]) -> LLMProvider:
    """Instantiate one provider for a configured model entry."""
    backend = str(spec["provider"])
    model = model_name(spec)
    if backend == "openai":
        from tunelm.providers.openai import OpenAIProvider

        provider = OpenAIProvider(model)
    elif backend == "gemini":
        from tunelm.providers.gemini import GeminiProvider

        provider = GeminiProvider(model)
    elif backend == "anthropic":
        from tunelm.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model)
    elif backend == "openrouter":
        from tunelm.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider(model)
    elif backend == "deepseek":
        from tunelm.providers.deepseek import DeepSeekProvider

        provider = DeepSeekProvider(model)
    else:
        raise ValueError(f"Unsupported provider backend: {backend}")
    provider.model_id = model_id
    return provider


def enabled_model_entries(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for model_id, spec in model_specs(config).items():
        if not spec.get("enabled", False):
            continue
        backend = str(spec["provider"])
        if not provider_api_key(backend):
            continue
        entries.append((model_id, spec))
    return entries


def llm_models_available(config: dict[str, Any]) -> bool:
    return bool(enabled_model_entries(config))


def router_from_models(config: dict[str, Any]) -> ProviderRouter:
    weights = config.get("weights", {})
    choices: list[ProviderChoice] = []
    for model_id, spec in enabled_model_entries(config):
        provider = create_provider(model_id, spec)
        choices.append(ProviderChoice(provider, float(weights.get(model_id, 1.0))))
    if not choices:
        raise RuntimeError("No enabled models have API keys configured")
    return ProviderRouter(
        choices,
        strategy=config.get("strategy", "weighted"),
        seed=int(config.get("seed", 42)),
    )


def single_model_routers(config: dict[str, Any]) -> list[tuple[str, ProviderRouter]]:
    """One fixed router per enabled model for parallel sharding."""
    routers: list[tuple[str, ProviderRouter]] = []
    for model_id, spec in enabled_model_entries(config):
        provider = create_provider(model_id, spec)
        routers.append(
            (
                model_id,
                ProviderRouter(
                    [ProviderChoice(provider, 1.0)],
                    strategy="single_provider",
                    seed=int(config.get("seed", 42)),
                ),
            )
        )
    if not routers:
        raise RuntimeError("No enabled models have API keys configured")
    return routers
