"""Task mix weights and optional task-type filters for dataset generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

TASK_TYPES = ("compose", "edit", "repair")


def configured_task_types(config: dict[str, Any]) -> frozenset[str] | None:
    """Return enabled task types from config, or None when all mix types run."""
    raw = config.get("task_types")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("task_types must be a non-empty list of compose, edit, repair")
    types = frozenset(str(item) for item in raw)
    unknown = types - set(TASK_TYPES)
    if unknown:
        raise ValueError(f"Unknown task_types: {sorted(unknown)}")
    return types


def effective_mix(config: dict[str, Any]) -> dict[str, float]:
    """Apply task_types filter to mix weights, renormalizing enabled types."""
    mix = dict(config.get("mix") or {"compose": 0.7, "edit": 0.2, "repair": 0.1})
    allowed = configured_task_types(config)
    if not allowed:
        return mix
    filtered = {name: float(mix.get(name, 0.0)) if name in allowed else 0.0 for name in TASK_TYPES}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("task_types excludes every weighted type in mix; enable at least one")
    return {name: weight / total for name, weight in filtered.items()}


def _task_type_value(task: Any) -> str:
    task_type = getattr(task, "task_type", None)
    if task_type is not None:
        return str(getattr(task_type, "value", task_type))
    return str(task["task_type"])


def filter_tasks_by_type(tasks: Iterable[Any], config: dict[str, Any]) -> list[Any]:
    """Keep only tasks whose type is listed in config.task_types."""
    allowed = configured_task_types(config)
    if not allowed:
        return list(tasks)
    return [task for task in tasks if _task_type_value(task) in allowed]


def task_counts(count: int, mix: dict[str, float]) -> tuple[int, int, int]:
    compose_count = round(count * float(mix.get("compose", 0)))
    edit_count = round(count * float(mix.get("edit", 0)))
    repair_count = count - compose_count - edit_count
    if min(compose_count, edit_count, repair_count) < 0:
        raise ValueError("Task mix produced a negative count; verify mix weights")
    return compose_count, edit_count, repair_count
