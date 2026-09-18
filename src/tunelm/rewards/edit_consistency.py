from __future__ import annotations

from typing import Any

from tunelm.schemas import EditTask, ExpectedChanges
from tunelm.strudel.analyzer import analyze_code, instruments_satisfy


def _direction(before: float, after: float, expected: str) -> float:
    if expected == "increase":
        return float(after > before)
    if expected == "decrease":
        return float(after < before)
    return float(after == before)


def _combined_decrease_score(before: dict[str, Any], after: dict[str, Any]) -> float:
    layer_decreased = (after.get("layer_count") or 0) < (before.get("layer_count") or 0)
    drums_decreased = (after.get("drum_density") or 0) < (before.get("drum_density") or 0)
    return float(layer_decreased or drums_decreased)


def _field_change_score(
    field: str, expected: str | None, before: dict[str, Any], after: dict[str, Any]
) -> float | None:
    if not expected:
        return None
    before_value = before.get(field) or 0
    if expected == "decrease" and field in {"drum_density", "layer_count"} and before_value <= 1:
        return None
    return _direction(before_value, after.get(field) or 0, expected)


def _change_scores(
    before: dict[str, Any], after: dict[str, Any], changes: ExpectedChanges
) -> list[float]:
    combo_decrease = changes.drum_density == "decrease" and changes.layer_count == "decrease"
    scores = [_combined_decrease_score(before, after)] if combo_decrease else []
    for field in ("tempo", "drum_density", "layer_count"):
        if combo_decrease and field in {"drum_density", "layer_count"}:
            continue
        score = _field_change_score(field, getattr(changes, field), before, after)
        if score is not None:
            scores.append(score)
    return scores


def _preservation_scores(
    before: dict[str, Any], after: dict[str, Any], preserve: list[str]
) -> list[float]:
    scores: list[float] = []
    for field in preserve:
        if field == "notes":
            scores.append(float(set(before.get("notes") or []) <= set(after.get("notes") or [])))
        elif field in {"tempo", "drum_density", "layer_count", "instruments"}:
            scores.append(float(before.get(field) == after.get(field)))
    return scores


def score_edit_consistency(candidate_code: str, task: EditTask | dict[str, Any]) -> float:
    if not isinstance(task, EditTask):
        task = EditTask.model_validate(task)
    before = analyze_code(task.original_code)
    after = analyze_code(candidate_code)
    changes = task.expected_changes
    scores = _change_scores(before, after, changes)
    scores.extend(
        float(instruments_satisfy(name, after["instruments"])) for name in changes.add_instruments
    )
    scores.extend(
        float(not instruments_satisfy(name, after["instruments"]))
        for name in changes.remove_instruments
    )
    scores.extend(_preservation_scores(before, after, task.preserve))
    return sum(scores) / len(scores) if scores else 1.0
