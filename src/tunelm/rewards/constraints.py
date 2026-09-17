"""Score explicit, statically verifiable task constraints.

Mood and scale remain outside this module until TuneLM has a defensible audio or
embedding verifier for them.
"""

from __future__ import annotations

from typing import Any

from tunelm.schemas import TaskConstraints, TempoRange
from tunelm.strudel.analyzer import DRUM_NAMES, canonical_instrument


def has_verifiable_constraints(constraints: TaskConstraints | dict) -> bool:
    """Return whether the constraint reward has at least one supported check."""
    if not isinstance(constraints, TaskConstraints):
        constraints = TaskConstraints.model_validate(constraints or {})
    return bool(
        constraints.tempo is not None
        or constraints.required_instruments
        or constraints.forbidden_instruments
        or constraints.min_layers is not None
        or constraints.max_layers is not None
    )


def _tempo_score(actual: float | None, expected: float | TempoRange | None) -> float | None:
    if expected is None:
        return None
    if actual is None:
        return 0.0
    if isinstance(expected, (int, float)):
        tolerance = max(2.0, float(expected) * 0.03)
        return max(0.0, 1.0 - abs(actual - float(expected)) / tolerance)
    if expected.min <= actual <= expected.max:
        return 1.0
    distance = min(abs(actual - expected.min), abs(actual - expected.max))
    return max(0.0, 1.0 - distance / 10.0)


def _instrument_scores(actual: set[str], constraints: TaskConstraints) -> list[float]:
    def present(name: str) -> bool:
        canonical = canonical_instrument(name)
        return canonical in actual or (canonical == "drums" and bool(actual & DRUM_NAMES))

    required = [float(present(name)) for name in constraints.required_instruments]
    forbidden = [float(not present(name)) for name in constraints.forbidden_instruments]
    return required + forbidden


def _layer_scores(layer_count: int, constraints: TaskConstraints) -> list[float]:
    scores = []
    if constraints.min_layers is not None:
        scores.append(float(layer_count >= constraints.min_layers))
    if constraints.max_layers is not None:
        scores.append(float(layer_count <= constraints.max_layers))
    return scores


def score_constraints(
    features: dict[str, Any], constraints: TaskConstraints | dict
) -> float | None:
    """Return mean supported-constraint satisfaction, or `None` when inapplicable."""
    if not isinstance(constraints, TaskConstraints):
        constraints = TaskConstraints.model_validate(constraints or {})
    scores: list[float] = []
    tempo = _tempo_score(features.get("tempo"), constraints.tempo)
    if tempo is not None:
        scores.append(tempo)
    actual = {canonical_instrument(item) for item in features.get("instruments", [])}
    scores.extend(_instrument_scores(actual, constraints))
    scores.extend(_layer_scores(features.get("layer_count", 0), constraints))
    # Mood is intentionally excluded until an audio/embedding verifier exists.
    return sum(scores) / len(scores) if scores else None
