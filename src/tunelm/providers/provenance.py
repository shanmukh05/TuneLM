"""Map a chosen LLM provider instance to dataset provenance metadata."""

from __future__ import annotations

from tunelm.providers.base import LLMProvider
from tunelm.schemas import ModelProvenance


def provider_provenance(provider: LLMProvider, *, style: str | None = None) -> ModelProvenance:
    return ModelProvenance(
        model_id=str(getattr(provider, "model_id", provider.name)),
        provider=provider.name,
        model=provider.model,
        style=style,
    )
