from __future__ import annotations

from tunelm.providers.base import LLMProvider, ProviderChoice, ProviderRouter
from tunelm.providers.registry import (
    create_provider,
    enabled_model_entries,
    llm_models_available,
    model_specs,
    provider_api_key,
    router_from_models,
    single_model_routers,
)


def llm_providers_available(config: dict) -> bool:
    """Return True when at least one enabled model has its API key set."""
    return llm_models_available(config)


def prompt_router_from_config(config: dict) -> ProviderRouter | None:
    """Build a router for prompt authoring, or None when unavailable."""
    author = config.get("prompt_author")
    merged = {**config, **author} if author else config
    try:
        return router_from_config(merged)
    except (ValueError, RuntimeError):
        return None


def require_prompt_router_from_config(config: dict) -> ProviderRouter:
    """Build a prompt-author router or raise when models or API keys are missing."""
    router = prompt_router_from_config(config)
    if router is None:
        raise RuntimeError(
            "Prompt authoring requires at least one enabled model with an API key "
            "under models (or compose.prompt_author.models for SFT compose)."
        )
    return router


def router_from_config(config: dict) -> ProviderRouter:
    return router_from_models(config)


__all__ = [
    "LLMProvider",
    "ProviderChoice",
    "ProviderRouter",
    "create_provider",
    "enabled_model_entries",
    "llm_models_available",
    "llm_providers_available",
    "model_specs",
    "prompt_router_from_config",
    "require_prompt_router_from_config",
    "provider_api_key",
    "router_from_config",
    "router_from_models",
    "single_model_routers",
]
