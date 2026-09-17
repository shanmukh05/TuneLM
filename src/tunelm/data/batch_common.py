"""Shared Gemini batch helpers for TuneLM SFT and RL data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tunelm.providers.gemini_batch import BatchRequest, GeminiBatchClient
from tunelm.providers.registry import enabled_model_entries, model_name, provider_api_key


def batch_settings(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "poll_seconds": 30,
        "inline_max_requests": 100,
        "regenerate_validate": True,
    }
    return {**defaults, **config.get("batch", {})}


def batch_enabled(config: dict[str, Any]) -> bool:
    return bool(batch_settings(config)["enabled"])


def refuse_overwrite(path: Path, config: dict[str, Any]) -> None:
    if config.get("append", False):
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; set append: true or remove it")


def require_single_gemini_model(config: dict[str, Any], *, label: str = "generation") -> tuple[str, str, str]:
    """Return model_id, provider name, and API model slug."""
    entries = enabled_model_entries(config)
    gemini_entries = [(model_id, spec) for model_id, spec in entries if spec.get("provider") == "gemini"]
    if len(gemini_entries) != 1:
        raise RuntimeError(
            f"Batch {label} requires exactly one enabled Gemini model with an API key"
        )
    model_id, spec = gemini_entries[0]
    return model_id, "gemini", model_name(spec)


def run_batch(
    config: dict[str, Any],
    requests: list[BatchRequest],
    *,
    display_name: str,
    model: str,
) -> dict[str, str]:
    settings = batch_settings(config)
    client = GeminiBatchClient(model=model, api_key=provider_api_key("gemini"))
    results = client.run(
        requests,
        display_name=display_name,
        poll_seconds=int(settings["poll_seconds"]),
        inline_max_requests=int(settings["inline_max_requests"]),
    )
    texts: dict[str, str] = {}
    errors: list[str] = []
    for key, result in results.items():
        if result.text:
            texts[key] = result.text
        else:
            errors.append(f"{key}: {result.error or 'empty response'}")
    if errors:
        raise RuntimeError("Gemini batch returned errors:\n" + "\n".join(errors[:10]))
    return texts
