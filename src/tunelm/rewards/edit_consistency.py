from __future__ import annotations

import re
from typing import Any

from tunelm.schemas import EditTask, ExpectedChanges
from tunelm.strudel.analyzer import analyze_code, instruments_satisfy


def _direction(before: float, after: float, expected: str) -> float:
    if expected == "increase":
        return float(after > before)
    if expected == "decrease":
        return float(after < before)
    return float(after == before)


def _field_change_score(
    field: str, expected: str | None, before: dict[str, Any], after: dict[str, Any]
) -> float | None:
    if not expected:
        return None
    before_value = before.get(field) or 0
    return _direction(before_value, after.get(field) or 0, expected)


def _change_scores(
    before: dict[str, Any], after: dict[str, Any], changes: ExpectedChanges
) -> list[float]:
    scores = []
    for field in ("tempo", "drum_density", "layer_count"):
        score = _field_change_score(field, getattr(changes, field), before, after)
        if score is not None:
            scores.append(score)
    return scores


def _named_layer(code: str, name: str) -> str | None:
    match = re.search(rf"\bconst\s+{re.escape(name)}\s*=", code)
    if match is None:
        return None
    start = match.end()
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(code)):
        char = code[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == ";":
            return re.sub(r"\s+", "", code[start:index])
        elif depth == 0 and char == "\n" and re.match(
            r"\s*(?:const\s+\w+\s*=|stack\s*\()", code[index + 1 :]
        ):
            return re.sub(r"\s+", "", code[start:index])
    return re.sub(r"\s+", "", code[start:])


def _preservation_scores(
    before: dict[str, Any], after: dict[str, Any], preserve: list[str],
    original_code: str, candidate_code: str,
) -> list[float]:
    scores: list[float] = []
    for field in preserve:
        if field == "notes":
            scores.append(float(set(before.get("notes") or []) <= set(after.get("notes") or [])))
        elif field in {"tempo", "drum_density", "layer_count", "instruments"}:
            scores.append(float(before.get(field) == after.get(field)))
        else:
            original_layer = _named_layer(original_code, field)
            if original_layer is not None:
                scores.append(float(original_layer == _named_layer(candidate_code, field)))
    return scores


def score_edit_consistency(candidate_code: str, task: EditTask | dict[str, Any]) -> float:
    if not isinstance(task, EditTask):
        task = EditTask.model_validate(task)
    before = analyze_code(task.original_code)
    after = analyze_code(candidate_code)
    changes = task.expected_changes
    requested_change = any(
        getattr(changes, field) in {"increase", "decrease"}
        for field in ("tempo", "drum_density", "layer_count")
    ) or bool(changes.add_instruments or changes.remove_instruments)
    if requested_change and candidate_code.strip() == task.original_code.strip():
        return 0.0
    change_scores = _change_scores(before, after, changes)
    change_scores.extend(
        float(instruments_satisfy(name, after["instruments"])) for name in changes.add_instruments
    )
    change_scores.extend(
        float(not instruments_satisfy(name, after["instruments"]))
        for name in changes.remove_instruments
    )
    if change_scores and any(score < 1.0 for score in change_scores):
        return 0.0
    scores = change_scores
    scores.extend(
        _preservation_scores(before, after, task.preserve, task.original_code, candidate_code)
    )
    return sum(scores) / len(scores) if scores else 1.0
