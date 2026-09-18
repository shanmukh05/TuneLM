from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "tunelm").is_dir():
            return candidate
    return start


@lru_cache(maxsize=1)
def repo_root() -> Path:
    return find_repo_root()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve paths relative to the repository."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    extends = value.pop("extends", None)
    if extends:
        base_path = Path(extends)
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        value = _deep_merge(load_config(base_path), value)
    value["_config_path"] = str(config_path)
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path
