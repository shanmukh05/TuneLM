"""Resume helpers for staged and full dataset generation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tunelm.io import read_jsonl


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(read_jsonl(path))


def resume_remaining(config: dict[str, Any], path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Return how many new rows to create and rows already stored at ``path``."""
    target = int(config.get("count", 100))
    if not config.get("append", False):
        return target, []
    existing = load_existing_rows(path)
    return max(0, target - len(existing)), existing


def existing_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows if row.get("id")}


def next_index(existing_ids: Iterable[str], prefix: str) -> int:
    """Next numeric suffix for ids shaped like ``{prefix}{index:06d}``."""
    max_index = -1
    for task_id in existing_ids:
        if not task_id.startswith(prefix):
            continue
        suffix = task_id.removeprefix(prefix)
        try:
            max_index = max(max_index, int(suffix))
        except ValueError:
            continue
    return max_index + 1
