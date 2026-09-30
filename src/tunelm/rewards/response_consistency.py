"""Check response metadata against the events and tempo of executable code."""

from __future__ import annotations

from tunelm.schemas import ModelResponse
from tunelm.strudel.analyzer import DRUM_NAMES, canonical_instrument, instruments_satisfy


def score_response_consistency(response: ModelResponse, features: dict) -> float:
    actual_tempo = features.get("tempo")
    actual_sounds = set(features.get("instruments") or [])
    tolerance = max(2.0, float(actual_tempo or 0) * 0.05)
    checks = [
        float(actual_tempo is not None and abs(response.controls.bpm - actual_tempo) <= tolerance)
    ]
    if isinstance(response.plan.tempo, (float, int)):
        checks.append(
            float(actual_tempo is not None and abs(response.plan.tempo - actual_tempo) <= tolerance)
        )
    for layer in response.controls.layers:
        checks.extend(
            float(instruments_satisfy(sound, actual_sounds)) for sound in layer.sounds
        )
    planned = {canonical_instrument(name) for name in response.plan.instruments}
    actual = {canonical_instrument(name) for name in actual_sounds}
    checks.append(float(bool(planned & actual) or ("drums" in planned and bool(actual & DRUM_NAMES))))
    return sum(checks) / len(checks)
