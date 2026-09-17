from __future__ import annotations

from typing import Any

from tunelm.schemas import EditTask
from tunelm.strudel.analyzer import analyze_code, canonical_instrument, instruments_satisfy


def _direction(before: float, after: float, expected: str) -> float:
    if expected == "increase":
        return float(after > before)
    if expected == "decrease":
        return float(after < before)
    return float(after == before)


def score_edit_consistency(candidate_code: str, task: EditTask | dict[str, Any]) -> float:
    if not isinstance(task, EditTask):
        task = EditTask.model_validate(task)
    before = analyze_code(task.original_code)
    after = analyze_code(candidate_code)
    scores: list[float] = []
    changes = task.expected_changes
    combo_decrease = (
        changes.drum_density == "decrease" and changes.layer_count == "decrease"
    )
    if combo_decrease:
        drum_before = before.get("drum_density") or 0
        drum_after = after.get("drum_density") or 0
        layer_before = before.get("layer_count") or 0
        layer_after = after.get("layer_count") or 0
        simplified = (layer_after < layer_before) or (drum_after < drum_before)
        scores.append(float(simplified))
    for field in ("tempo", "drum_density", "layer_count"):
        expected = getattr(changes, field)
        if not expected:
            continue
        if combo_decrease and field in {"drum_density", "layer_count"}:
            continue
        before_value = before.get(field) or 0
        if expected == "decrease" and field == "drum_density" and before_value <= 1:
            continue
        if expected == "decrease" and field == "layer_count" and before_value <= 1:
            continue
        scores.append(_direction(before_value, after.get(field) or 0, expected))
    actual_instruments = {canonical_instrument(x) for x in after["instruments"]}
    scores.extend(
        float(instruments_satisfy(name, after["instruments"])) for name in changes.add_instruments
    )
    scores.extend(
        float(not instruments_satisfy(name, after["instruments"]))
        for name in changes.remove_instruments
    )
    for field in task.preserve:
        if field == "notes":
            before_notes = set(before.get("notes") or [])
            after_notes = set(after.get("notes") or [])
            scores.append(float(before_notes <= after_notes))
            continue
        if field in {"tempo", "drum_density", "layer_count", "instruments"}:
            scores.append(float(before.get(field) == after.get(field)))
    return sum(scores) / len(scores) if scores else 1.0
