"""Load repository-owned musical vocabularies with process-local caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from tunelm.config import repo_root


class VocabStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or repo_root() / "configs" / "data" / "vocab"

    @lru_cache(maxsize=None)
    def load(self, name: str) -> dict[str, Any]:
        path = self.root / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
        if not isinstance(value, dict):
            raise ValueError(f"Vocabulary {name} must be a mapping")
        return value

    @property
    def instruments(self) -> list[str]:
        return list(self.load("instruments")["canonical"])

    @property
    def instrument_aliases(self) -> dict[str, str | list[str]]:
        return dict(self.load("instruments").get("aliases", {}))

    @property
    def instrument_sounds(self) -> dict[str, str]:
        return dict(self.load("instruments").get("sounds", {}))

    @property
    def ensembles(self) -> list[list[str]]:
        return [list(item) for item in self.load("instruments").get("ensembles", [])]

    @property
    def moods(self) -> list[str]:
        return list(self.load("moods")["values"])

    @property
    def genres(self) -> list[str]:
        return list(self.load("genres")["values"])

    @property
    def structures(self) -> list[str]:
        return list(self.load("structures")["values"])

    @property
    def rhythms(self) -> list[str]:
        return list(self.load("rhythms")["values"])

    @property
    def textures(self) -> list[str]:
        return list(self.load("textures")["values"])

    @property
    def compose_templates(self) -> dict[str, Any]:
        return self.load("compose_templates")

    @property
    def edit_operations(self) -> list[dict[str, Any]]:
        return list(self.load("edit_operations")["operations"])

    @property
    def repair_templates(self) -> dict[str, Any]:
        return self.load("repair_templates")

    @property
    def tempo_ranges(self) -> list[tuple[int, int]]:
        return [(int(start), int(end)) for start, end in self.load("tempo_ranges")["values"]]


@lru_cache(maxsize=1)
def load_vocab() -> VocabStore:
    return VocabStore()
