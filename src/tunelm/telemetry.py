"""Compute deterministic musical proxies from native Strudel Haps.

These features describe symbolic structure; they do not measure audio quality.
"""

from __future__ import annotations

import json
import math
import re

from tunelm.schemas import Telemetry

PITCH_CLASS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}


def midi_pitch(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"([a-gA-G])([#b]?)(-?\d+)", str(value))
    if not match:
        return None
    letter, accidental, octave = match.groups()
    offset = {"#": 1, "b": -1, "": 0}[accidental]
    return float((int(octave) + 1) * 12 + PITCH_CLASS[letter.lower()] + offset)


def scale_classes(key: str) -> set[int]:
    root, mode = key.split(":")
    pitch = int(midi_pitch(root + "4")) % 12
    return {(pitch + step) % 12 for step in SCALE_INTERVALS[mode]}


def _controls(event: dict) -> dict:
    value = event["value"]
    if not isinstance(value, dict):
        return {"note": value}
    if value.get("bank") and value.get("s"):
        return {**value, "s": str(value["bank"]) + "_" + str(value["s"])}
    return value


def audible_onsets(events: list[dict]) -> list[dict]:
    return [e for e in events if e["onset"] and _controls(e).get("gain", 1) > 0]


def _fingerprints(events: list[dict], cycles: int, controls: bool) -> list[str]:
    buckets = [[] for _ in range(cycles)]
    for event in events:
        cycle = math.floor(event["begin"])
        if not 0 <= cycle < cycles:
            continue
        values = _controls(event)
        keys = sorted(values) if controls else ("s", "note", "n")
        values = {k: _quantize(values[k]) for k in keys if k in values}
        token = [round(event["begin"] - cycle, 6), round(event["end"] - event["begin"], 6), values]
        buckets[cycle].append(json.dumps(token, sort_keys=True))
    return ["|".join(sorted(bucket)) for bucket in buckets]


def _quantize(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _quantize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_quantize(v) for v in value]
    return value


def summarize(events: list[dict], cycles: int) -> Telemetry:
    """Count onsets once; retain empty cycles when comparing temporal fingerprints."""
    onsets = audible_onsets(events)
    controls = [_controls(e) for e in onsets]
    pitches = sorted({p for c in controls if (p := midi_pitch(c.get("note"))) is not None})
    structural = _fingerprints(onsets, cycles, False)
    automation = _fingerprints(onsets, cycles, True)
    gains = [c.get("gain", 1) for c in controls]
    off_grid = sum(abs(e["begin"] * 4 - round(e["begin"] * 4)) > 1e-6 for e in onsets)
    return Telemetry(
        cycles=cycles,
        event_count=len(onsets),
        events_per_cycle=len(onsets) / cycles,
        sound_sources=sorted({str(c["s"]) for c in controls if "s" in c}),
        pitches=pitches,
        pitch_range=(min(pitches), max(pitches)) if pitches else None,
        syncopation=off_grid / max(1, len(onsets)),
        structural_variation=len(set(structural)) / cycles,
        control_variation=len(set(automation)) / cycles,
        active_cycles=sum(bool(c) for c in structural),
        gain_min=min(gains, default=0),
        gain_max=max(gains, default=0),
    )


def evolution_windows(events: list[dict], cycles: int) -> list[float]:
    signatures = _fingerprints(audible_onsets(events), cycles, False)
    return [
        len(set(signatures[start : start + 8])) / len(signatures[start : start + 8])
        for start in range(0, cycles, 8)
    ]
