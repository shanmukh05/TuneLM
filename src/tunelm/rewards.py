"""Shared executable musical gates for dataset acceptance and GRPO rewards.

`score` gates SFT teacher answers against their plan. `contract_score` is the GRPO
reward: one verify checked against the request contract, plus junk guards and
thinking/code agreement (docs/rl-reward-design.md). All components are symbolic
proxies; reward traces retain the full decomposition.
"""

from __future__ import annotations

import json
import math
import re
from functools import cache, lru_cache

from tunelm.io import load_config
from tunelm.responses import parse_completion
from tunelm.schemas import Plan, RequestContract
from tunelm.telemetry import (
    SCALE_INTERVALS,
    audible_onsets,
    evolution_windows,
    midi_pitch,
    scale_classes,
    section_profiles,
)
from tunelm.verifier import Verifier

SHARES = {"contract": 0.8, "honesty": 0.2}
GUARD_FAILURE = 0.5
WEIGHTS = {"forbid": 2.0, "preserve": 2.0, "vocals": 0.5, "shape": 0.5, "space": 0.5}
VOCALS = "vocals"
LOW_NOTE = 48
KEY_SHARE = 0.9
CLAIMED_KEY_SHARE = 0.8
# ponytail: bands set from verified SFT telemetry percentiles (p2/p98); recalibrate on rollouts.
MIN_EVENTS_PER_CYCLE, MAX_EVENTS_PER_CYCLE = 0.5, 64
MIN_VARIATION = 0.125
MAX_SOUNDS = 10
MAX_GAIN = 1.0
MIN_SCALE_FIT = 0.6
SPARSE_MAX_EVENTS, BUSY_MIN_EVENTS = 8.0, 12.0
SHAPE_RATIO = 1.5
OVERLONG_WINDOW, OVERLONG_PENALTY = 256, 0.2
BPM_CLAIM = re.compile(r"\b(\d{2,3}(?:\.\d+)?)\s*bpm\b", re.I)
KEY_CLAIM = re.compile(r"\b([A-G][#b]?)[ :](" + "|".join(SCALE_INTERVALS) + r")\b")
SCALES = [
    {(root + step) % 12 for step in steps}
    for root in range(12)
    for steps in SCALE_INTERVALS.values()
]


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


def _gate(text: str, cycles: int, verifier: Verifier | None) -> tuple[str, str, dict] | dict:
    """Return (thinking, code, verify result), or the zero-reward record that stopped it."""
    try:
        thinking, code = parse_completion(text)
    except ValueError as error:
        return {
            "total": 0.0,
            "components": {"format": 0.0},
            "valid": False,
            "diagnostic": {"stage": "format", "error": str(error)},
        }
    result = (verifier or Verifier()).request(code, cycles=cycles)
    if not result["valid"]:
        return {
            "total": 0.0,
            "components": {"format": 1.0, "execution": 0.0},
            "valid": False,
            "diagnostic": {k: v for k, v in result.items() if k != "events"},
        }
    return thinking, code, result


def score(text: str, plan: Plan, verifier: Verifier | None = None) -> dict:
    """Hard gate on format/execution, then average applicable musical components."""
    gated = _gate(text, plan.cycles, verifier)
    if isinstance(gated, dict):
        return gated
    _, code, result = gated
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


@cache
def sound_families() -> dict[str, dict]:
    """Closed contract families; every listed sound must exist in the runtime catalog."""
    families = load_config("configs/registry/sound_families.yaml")["families"]
    catalog = set(load_config("configs/registry/sounds.yaml")["sounds"])
    for name, family in families.items():
        unknown = set(family.get("sounds", [])) - catalog
        if unknown or set(family) - {"sounds", "kit_pieces", "low_notes"}:
            raise ValueError(f"Invalid sound family {name}: {sorted(unknown) or sorted(family)}")
    return families


def _in_family(sound: str, family: dict) -> bool:
    if sound in family.get("sounds", ()):
        return True
    return not sound.startswith("gm_") and sound.split("_")[-1] in family.get("kit_pieces", ())


def _present(name: str, telemetry: dict) -> bool:
    family = sound_families()[name]
    if any(_in_family(sound, family) for sound in telemetry["sound_sources"]):
        return True
    return family.get("low_notes", False) and any(p <= LOW_NOTE for p in telemetry["pitches"])


def voice_count(sounds: list[str], named: tuple[str, ...] = ()) -> int:
    """Count parts: sounds in one named family are one part, the drum kit is one part.

    Slap and fretless basses for "bass alone" are one bass; an unnamed extra
    sound still counts on its own.
    """
    families = sound_families()
    groups = [*named, "drums"]
    return len({next((g for g in groups if _in_family(s, families[g])), s) for s in sounds})


def _plain(text: str) -> str:
    text = text.translate(str.maketrans("‘’“”", "''\"\""))
    return " ".join(text.casefold().split())


def check_contract(prompt: str, contract: RequestContract, original_code: str | None) -> None:
    """Reject claims that are not quoted from the request, not checkable, or too few."""
    _check_quotes(prompt, contract)
    _check_families(contract)
    _check_literals(contract)
    forbid = [claim for claim in contract.forbid if claim.family != VOCALS]
    strong = [*contract.require, *forbid, contract.voices, contract.tempo, contract.key]
    strong = [claim for claim in strong if claim] + ([original_code] if original_code else [])
    if not strong or len(strong) + bool(contract.shape) + bool(contract.space) < 2:
        raise ValueError(
            "The request needs at least two checkable details, and one must be a named "
            "instrument, a refusal, a part count, a tempo, a key, or supplied code"
        )


def _claims(contract: RequestContract) -> list:
    singles = [contract.voices, contract.tempo, contract.key, contract.shape, contract.space]
    return [*contract.require, *contract.forbid, *filter(None, [*singles, contract.loop])]


def _check_quotes(prompt: str, contract: RequestContract) -> None:
    text = _plain(prompt)
    for claim in _claims(contract):
        if _plain(claim.quote) not in text:
            raise ValueError(f"Contract quote is not copied from the request: {claim.quote!r}")


def _check_families(contract: RequestContract) -> None:
    named = [claim.family for claim in [*contract.require, *contract.forbid]]
    unknown = sorted(set(named) - sound_families().keys())
    if unknown:
        raise ValueError(f"Unknown sound families {unknown}; use only names from families")
    if len(set(named)) != len(named):
        raise ValueError("A family appears more than once in require/forbid")


def _check_literals(contract: RequestContract) -> None:
    voices, tempo, key = contract.voices, contract.tempo, contract.key
    if voices and (voices.min > voices.max or len(contract.require) > voices.max):
        raise ValueError("voices must be min <= max and cover every required family")
    if tempo and (f"{tempo.bpm:g}" not in tempo.quote or "bpm" not in _plain(tempo.quote)):
        raise ValueError("tempo needs the written number and 'bpm' in its quote")
    if key and key.key.split(":")[1] not in _plain(key.quote):
        raise ValueError("key needs the written mode in its quote")


def contract_score(
    text: str, contract: RequestContract, plan: Plan, verifier: Verifier | None = None
) -> dict:
    """GRPO reward in [0, 1]: gate x 0.5^(failed craft guards) x (contract, honesty).

    Guards multiply so a degenerate program that ticks the contract still ranks
    below a real attempt. `plan` supplies only evaluation cycles and co-creation
    `original_code`; its advisory voices, key, and tempo are not scored. Length
    shaping is applied by the trainer adapter, which sees token counts.
    """
    gated = _gate(text, plan.cycles, verifier)
    if isinstance(gated, dict):
        return gated
    thinking, code, result = gated
    contract_checks = {
        **_family_checks(result["telemetry"], contract),
        **_literal_checks(result, contract, plan, code),
    }
    craft = _craft_checks(result["telemetry"], contract)
    honesty = _honesty_checks(thinking, result)
    weight = math.fsum(w for w, _ in contract_checks.values())
    parts = {
        "contract": math.fsum(w * v for w, v in contract_checks.values()) / weight
        if weight
        else 1.0,
        "honesty": sum(honesty.values()) / len(honesty) if honesty else 1.0,
    }
    guards = GUARD_FAILURE ** sum(passed < 1 for passed in craft.values())
    components = {
        "format": 1.0,
        "execution": 1.0,
        **parts,
        "craft": guards,
        **{name: value for name, (_, value) in contract_checks.items()},
        **craft,
        **honesty,
    }
    return {
        "total": guards * math.fsum(SHARES[name] * value for name, value in parts.items()),
        "components": components,
        "valid": True,
        "diagnostic": {k: v for k, v in result.items() if k != "events"},
    }


def _family_checks(telemetry: dict, contract: RequestContract) -> dict[str, tuple[float, float]]:
    checks = {
        f"require.{c.family}": (1.0, float(_present(c.family, telemetry))) for c in contract.require
    }
    for claim in contract.forbid:
        weight = WEIGHTS["vocals"] if claim.family == VOCALS else WEIGHTS["forbid"]
        checks[f"forbid.{claim.family}"] = (weight, float(not _present(claim.family, telemetry)))
    if contract.voices:
        # Drum pieces count apart toward the minimum ("three dry hits") and as one kit
        # toward the maximum ("solo, with a little percussion" is still two).
        sounds = telemetry["sound_sources"]
        parts = voice_count(sounds, tuple(c.family for c in contract.require))
        fits = len(sounds) >= contract.voices.min and parts <= contract.voices.max
        checks["voices"] = (1.0, float(fits))
    return checks


def _literal_checks(
    result: dict, contract: RequestContract, plan: Plan, code: str
) -> dict[str, tuple[float, float]]:
    checks = {}
    if contract.tempo:
        cpm = result.get("cpm")
        checks["tempo"] = (1.0, float(cpm is not None and abs(cpm * 4 - contract.tempo.bpm) <= 1))
    if contract.key:
        checks["key"] = (1.0, float(_harmony(result["events"], contract.key.key) >= KEY_SHARE))
    if contract.shape:
        sections = section_profiles(result["events"], result["cycles"])
        checks["shape"] = (WEIGHTS["shape"], float(_shaped(sections, contract.shape.shape)))
    if contract.space:
        checks["space"] = (
            WEIGHTS["space"],
            float(_spaced(result["telemetry"], contract.space.space)),
        )
    if plan.original_code:
        kept = _preserved(plan.original_code.strip(), code, result["telemetry"], plan.cycles)
        checks["preserve"] = (WEIGHTS["preserve"], float(kept))
    return checks


def _shaped(sections: list[dict], shape: str) -> bool:
    """Some later section grows (build), shrinks (thin_out), or differs (contrast)."""
    pairs = [(a, b) for i, a in enumerate(sections) for b in sections[i + 1 :]]
    if shape == "build":
        return any(_grew(a, b) for a, b in pairs)
    if shape == "thin_out":
        return any(_grew(b, a) for a, b in pairs)
    return any(_grew(a, b) or _grew(b, a) or _differs(a, b) for a, b in pairs)


def _grew(before: dict, after: dict) -> bool:
    louder = after["onsets"] >= SHAPE_RATIO * max(1, before["onsets"])
    return louder or len(after["sounds"]) > len(before["sounds"])


def _differs(left: dict, right: dict) -> bool:
    union = left["sounds"] | right["sounds"]
    return bool(union) and len(left["sounds"] & right["sounds"]) / len(union) <= 0.5


def _spaced(telemetry: dict, space: str) -> bool:
    if space == "sparse":
        return telemetry["events_per_cycle"] <= SPARSE_MAX_EVENTS
    return telemetry["events_per_cycle"] >= BUSY_MIN_EVENTS


@lru_cache(maxsize=4096)
def _original_telemetry(code: str, cycles: int) -> str:
    return json.dumps(Verifier().request(code, cycles=cycles).get("telemetry"), sort_keys=True)


def _preserved(original: str, code: str, telemetry: dict, cycles: int) -> bool:
    """The supplied code is kept verbatim and the program still adds something audible."""
    if original not in code:
        return False
    return json.dumps(telemetry, sort_keys=True) != _original_telemetry(original, cycles)


def _craft_checks(telemetry: dict, contract: RequestContract) -> dict[str, float]:
    """Bands most verified SFT answers pass; they remove junk, not musical choices."""
    sounds = voice_count(telemetry["sound_sources"])
    allowed = max(MAX_SOUNDS, contract.voices.max if contract.voices else 0)
    events = telemetry["events_per_cycle"]
    checks = {
        "craft.density": MIN_EVENTS_PER_CYCLE <= events <= MAX_EVENTS_PER_CYCLE,
        "craft.gain": 0 < telemetry["gain_max"] <= MAX_GAIN,
        "craft.not_stuffed": sounds <= allowed,
    }
    if not contract.loop:
        checks["craft.variation"] = telemetry["structural_variation"] >= MIN_VARIATION
    pitches = telemetry["pitches"]
    if not contract.key and len(pitches) >= 4:
        checks["craft.pitch"] = _scale_fit(pitches) >= MIN_SCALE_FIT
    return {name: float(passed) for name, passed in checks.items()}


def _scale_fit(pitches: list[float]) -> float:
    classes = [round(p) % 12 for p in pitches]
    return max(sum(c in scale for c in classes) / len(classes) for scale in SCALES)


def _honesty_checks(thinking: str, result: dict) -> dict[str, float]:
    """Tempo and key stated in the thinking must be what the program plays."""
    checks = {}
    stated = [float(bpm) for bpm in BPM_CLAIM.findall(thinking)]
    if stated:
        cpm = result.get("cpm")
        checks["honesty.tempo"] = cpm is not None and any(abs(b - cpm * 4) <= 1 for b in stated)
    keys = {f"{root}:{mode}" for root, mode in KEY_CLAIM.findall(thinking)}
    if keys:
        fits = [_harmony(result["events"], key) for key in keys]
        checks["honesty.key"] = max(fits) >= CLAIMED_KEY_SHARE
    return {name: float(passed) for name, passed in checks.items()}


def overlong_penalty(length: int, limit: int) -> float:
    """DAPO-style soft penalty over the last OVERLONG_WINDOW tokens before the hard limit."""
    start = limit - OVERLONG_WINDOW
    return OVERLONG_PENALTY * min(1.0, max(0.0, (length - start) / OVERLONG_WINDOW))
