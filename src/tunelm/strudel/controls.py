"""Infer and verify structured `controls` metadata from Strudel code.

Used to bootstrap fixtures and to check that model-emitted controls align with
executable `strudel_code` during SFT validation.
"""

from __future__ import annotations

import re
from typing import Literal

from tunelm.schemas import LayerControl, MusicControls
from tunelm.strudel.analyzer import DRUM_NAMES, analyze_code, canonical_instrument

LayerRole = Literal[
    "rhythm",
    "bass",
    "melody",
    "harmony",
    "pads",
    "percussion",
    "texture",
    "fx",
    "other",
]


def _infer_key(code: str) -> str | None:
    match = re.search(r"""\.scale\(\s*['"]([^'"]+)['"]""", code)
    return match.group(1) if match else None


def _guess_role(sounds: list[str]) -> LayerRole:
    lowered = {canonical_instrument(sound) for sound in sounds}
    if lowered & DRUM_NAMES:
        return "rhythm"
    if any("bass" in sound for sound in lowered):
        return "bass"
    if any("pad" in sound for sound in lowered):
        return "pads"
    if len(lowered) <= 2:
        return "melody"
    return "harmony"


def infer_controls_from_code(
    code: str,
    *,
    mood: list[str] | None = None,
    bpm: float | None = None,
) -> MusicControls:
    """Best-effort controls object derived from Strudel source."""
    features = analyze_code(code)
    resolved_bpm = float(bpm if bpm is not None else features.get("tempo") or 120.0)
    instruments = [canonical_instrument(name) for name in features.get("instruments", [])]

    drum_sounds = [name for name in instruments if name in DRUM_NAMES]
    bass_sounds = [name for name in instruments if "bass" in name and name not in drum_sounds]
    pad_sounds = [name for name in instruments if "pad" in name]
    claimed = set(drum_sounds + bass_sounds + pad_sounds)
    other = [name for name in instruments if name not in claimed]

    layers: list[LayerControl] = []
    if drum_sounds:
        layers.append(LayerControl(id="drums", role="rhythm", sounds=drum_sounds))
    if bass_sounds:
        layers.append(LayerControl(id="bass", role="bass", sounds=bass_sounds))
    if pad_sounds:
        layers.append(LayerControl(id="pads", role="pads", sounds=pad_sounds))
    if other:
        layers.append(LayerControl(id="lead", role=_guess_role(other), sounds=other))
    if not layers:
        layers.append(LayerControl(id="main", role="other", sounds=instruments or ["pattern"]))

    return MusicControls(
        bpm=resolved_bpm,
        key=_infer_key(code),
        mood=list(mood or []),
        layers=layers,
    )


def controls_match_code(controls: MusicControls, features: dict) -> bool:
    """Return True when typed controls roughly agree with executed features."""
    tempo = features.get("tempo")
    if tempo is not None:
        tolerance = max(2.0, float(tempo) * 0.05)
        if abs(controls.bpm - float(tempo)) > tolerance:
            return False

    control_sounds = {
        canonical_instrument(sound) for layer in controls.layers for sound in layer.sounds
    }
    actual = {canonical_instrument(name) for name in features.get("instruments", [])}
    if actual and control_sounds and not (actual & control_sounds):
        return False

    layer_count = int(features.get("layer_count") or 1)
    if len(controls.layers) > layer_count + 1:
        return False
    return True
