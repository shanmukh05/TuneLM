"""Shared executable musical gates for dataset acceptance and GRPO rewards.

All components are symbolic proxies; reward traces retain the full decomposition.
"""

from __future__ import annotations

import math

from tunelm.responses import parse_completion
from tunelm.schemas import Plan
from tunelm.telemetry import audible_onsets, evolution_windows, midi_pitch, scale_classes
from tunelm.verifier import Verifier


def _harmony(events: list[dict], key: str | None) -> float | None:
    if not key:
        return None
    pitches = [
        midi_pitch(e["value"].get("note"))
        for e in audible_onsets(events)
        if isinstance(e["value"], dict) and "note" in e["value"]
    ]
    pitches = [p for p in pitches if p is not None]
    if not pitches:
        return 0.0
    valid = scale_classes(key)
    return sum(round(p) % 12 in valid for p in pitches) / len(pitches)


def _role_coverage(result: dict, plan: Plan) -> float:
    """Register/sound proxies do not pretend each instrument is a separate voice."""
    telemetry = result["telemetry"]
    sounds = {s.split("_")[-1] for s in telemetry["sound_sources"]}
    pitches = telemetry["pitches"]
    present = {
        "drums_core": bool(sounds & {"bd", "sd", "cp"}),
        "perc_tops": bool(sounds & {"hh", "oh", "rim", "sh", "rd"}),
        "bass": any(p <= 55 for p in pitches),
        "chords_pad": any(48 <= p <= 84 for p in pitches),
        "arp_lead": any(p >= 60 for p in pitches),
        "fx_texture": bool(sounds & {"noise", "white", "pink", "brown"}),
    }
    return sum(present[v.role] for v in plan.voices) / len(plan.voices)


def _instructions(result: dict, plan: Plan, code: str) -> float:
    checks = []
    if "bpm" in plan.explicit:
        checks.append(result.get("cpm") is not None and abs(result["cpm"] * 4 - plan.bpm) < 0.01)
    for voice in plan.voices:
        checks.extend(method in result["methods"] for method in voice.techniques)
    if plan.original_code:
        checks.append(plan.original_code.strip() in code)
    if plan.category == "arrange" and not plan.intent_only:
        checks.append("arrange" in result["methods"] or "cat" in result["methods"])
    return sum(checks) / len(checks) if checks else 1.0


def score(text: str, plan: Plan, verifier: Verifier | None = None) -> dict:
    """Hard gate on format/execution, then average applicable musical components."""
    try:
        _, code = parse_completion(text)
    except ValueError as error:
        return {
            "total": 0.0,
            "components": {"format": 0.0},
            "valid": False,
            "diagnostic": {"stage": "format", "error": str(error)},
        }
    result = (verifier or Verifier()).request(code, cycles=plan.cycles)
    if not result["valid"]:
        return {
            "total": 0.0,
            "components": {"format": 1.0, "execution": 0.0},
            "valid": False,
            "diagnostic": {k: v for k, v in result.items() if k != "events"},
        }
    telemetry = result["telemetry"]
    windows = evolution_windows(result["events"], result["cycles"])
    evolution = (
        1.0
        if plan.short_loop
        else sum(min(1.0, max(0.0, (v - 0.125) / 0.375)) for v in windows) / len(windows)
    )
    components = {
        "format": 1.0,
        "execution": 1.0,
        "mix_proxy": float(0 < telemetry["gain_max"] <= 2),
        "instructions": _instructions(result, plan, code),
    }
    if not plan.intent_only:
        components["roles"] = _role_coverage(result, plan)
    if plan.category != "atomic" and not plan.short_loop:
        components["evolution"] = evolution
    key = plan.key if not plan.intent_only or "key" in plan.explicit else None
    harmony = _harmony(result["events"], key)
    if harmony is not None:
        components["harmony"] = harmony
    # ponytail: equal weights are the initial baseline; tune only against listening review.
    values = [v for k, v in components.items() if k not in {"format", "execution"}]
    total = math.fsum(values) / len(values)
    diagnostic = {k: v for k, v in result.items() if k != "events"}
    return {
        "total": total,
        "components": components,
        "valid": True,
        "diagnostic": diagnostic,
        "evolution_windows": windows,
    }


def quality_failures(scored: dict, plan: Plan) -> list[str]:
    if not scored["valid"]:
        return [str(scored["diagnostic"])]
    failures = []
    components = scored["components"]
    if components["instructions"] < 1:
        failures.append("Explicit instructions or preservation failed")
    if not plan.intent_only and components.get("harmony", 1) < 0.9:
        failures.append("Scale adherence below 0.9")
    if components["mix_proxy"] < 1:
        failures.append("Gain exceeds the configured symbolic mixing range")
    if components.get("roles", 1) < 1:
        failures.append("Requested complementary register/sound roles are missing")
    if (
        not plan.intent_only
        and not plan.short_loop
        and plan.category != "atomic"
        and max(scored["evolution_windows"]) < 0.25
    ):
        failures.append("No 8-cycle window contains at least two structural variations")
    return failures
