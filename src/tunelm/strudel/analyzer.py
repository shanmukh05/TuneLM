"""Extract verifier features and perform lightweight Strudel syntax checks.

The Node executor remains authoritative for execution; this module provides the
fast, deterministic analysis shared by data generation and rewards.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from tunelm.data.vocab import load_vocab


SAMPLE_CALLS = ("s", "sound")
LAYER_CALLS = ("stack", "cat", "slowcat", "fastcat")
DRUM_NAMES = {
    "bd",
    "sd",
    "hh",
    "oh",
    "cp",
    "rim",
    "cr",
    "rd",
    "lt",
    "mt",
    "ht",
    "tom",
    "perc",
    "shaker",
    "tambourine",
    "cowbell",
    "conga",
    "drums",
}
DELIMITER_PAIRS = {"(": ")", "[": "]", "{": "}"}
QUOTES = "'\"`"


@lru_cache(maxsize=1)
def _instrument_names() -> tuple[dict[str, str | list[str]], dict[str, str]]:
    vocab = load_vocab()
    sounds = {sound: name for name, sound in vocab.instrument_sounds.items()}
    return vocab.instrument_aliases, sounds


def canonical_instrument(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9_-]", "", name.casefold())
    aliases, sounds = _instrument_names()
    alias = aliases.get(lowered)
    semantic_name = alias if isinstance(alias, str) else lowered
    return sounds.get(semantic_name, semantic_name)


def instruments_satisfy(expected: str, actual_names: set[str] | list[str]) -> bool:
    """Return True when at least one actual sound matches a requested instrument."""
    expected_norm = canonical_instrument(expected)
    aliases, sound_to_name = _instrument_names()
    vocab = load_vocab()
    acceptable = {expected_norm}
    default_sound = vocab.instrument_sounds.get(expected_norm)
    if default_sound:
        acceptable.add(canonical_instrument(default_sound))
    for sound, name in sound_to_name.items():
        if name == expected_norm:
            acceptable.add(canonical_instrument(sound))
    for alias, target in aliases.items():
        resolved = target if isinstance(target, str) else alias
        if canonical_instrument(resolved) == expected_norm:
            acceptable.add(canonical_instrument(alias))
    if expected_norm == "synth":
        acceptable.update({"sawtooth", "sine", "triangle", "square"})
    actual = {canonical_instrument(name) for name in actual_names}
    if acceptable & actual:
        return True
    if expected_norm == "organ":
        return any("organ" in name for name in actual)
    if expected_norm == "saxophone":
        return any("sax" in name for name in actual)
    return False


def _string_arguments(code: str, function_names: tuple[str, ...]) -> list[str]:
    names = "|".join(map(re.escape, function_names))
    pattern = rf"\b(?:{names})\s*\(\s*(['\"`])(.+?)\1\s*\)"
    return [match.group(2) for match in re.finditer(pattern, code, re.DOTALL)]


def _tokens(notation: str) -> list[str]:
    # Keep sample/note names but discard mini-notation operators and multipliers.
    return [
        canonical_instrument(token.split(":", 1)[0])
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*(?::[A-Za-z0-9_-]+)?", notation)
        if token.casefold() not in {"x", "rand"}
    ]


def infer_tempo(code: str) -> float | None:
    code = _strip_comments(code)
    setcpm = re.search(r"\bsetcpm\s*\(\s*([0-9.]+)(?:\s*/\s*([0-9.]+))?\s*\)", code)
    if setcpm:
        numerator = float(setcpm.group(1))
        divisor = float(setcpm.group(2) or 1)
        return numerator / divisor * 4
    cps = re.search(r"\bsetcps\s*\(\s*([0-9.]+)\s*\)", code)
    return float(cps.group(1)) * 240 if cps else None


def _strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", code)


def _call_arity(code: str, open_index: int) -> int:
    """Count top-level arguments of the call whose `(` sits at `open_index`."""
    depth = 0
    commas = 0
    quote: str | None = None
    escaped = False
    last = ""
    for char in code[open_index:]:
        if quote:
            quote, escaped = _advance_quote(char, quote, escaped)
            continue
        if char in QUOTES:
            quote = char
        elif char in DELIMITER_PAIRS:
            depth += 1
        elif char in DELIMITER_PAIRS.values():
            depth -= 1
            if depth == 0:
                return 0 if last == "(" else commas + (last != ",")
        elif char == "," and depth == 1:
            commas += 1
        if not char.isspace():
            last = char
    return commas + 1


def _layer_count(code: str) -> int:
    """Widest `stack(...)` argument count; a program without `stack` is one layer."""
    stripped = _strip_comments(code)
    arities = [
        _call_arity(stripped, match.end() - 1)
        for match in re.finditer(r"\bstack\s*\(", stripped)
    ]
    if arities:
        return max(arities)
    return 1 if code.strip() else 0


def _audible_onsets(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events that start in the queried window and are not muted with gain 0."""
    return [
        event
        for event in events
        if event.get("onset") is not False
        and isinstance(event.get("value"), dict)
        and event["value"].get("gain", 1) != 0
    ]


def _cycle_features(onsets: list[dict[str, Any]]) -> tuple[int, int]:
    """Return (cycles with sound, distinct per-cycle event patterns)."""
    per_cycle: dict[int, list[tuple[float, str]]] = {}
    for event in onsets:
        begin = event.get("begin")
        if begin is None:
            continue
        cycle = math.floor(begin)
        value = event["value"]
        musical_value = {key: value[key] for key in ("s", "note", "n") if key in value}
        signature = (round(begin - cycle, 4), json.dumps(musical_value, sort_keys=True))
        per_cycle.setdefault(cycle, []).append(signature)
    distinct = {tuple(sorted(items)) for items in per_cycle.values()}
    return len(per_cycle), len(distinct)


def analyze_code(code: str, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Extract verifier features from source, preferring executed events when present.

    With events, `instruments` lists only sounds that actually play (dead code and
    `gain(0)` layers do not count). Without events (static backend, edit/repair
    comparisons) it falls back to string arguments in the source.
    """
    sample_strings = _string_arguments(code, SAMPLE_CALLS)
    sample_tokens = [token for notation in sample_strings for token in _tokens(notation)]
    quoted_notes = _string_arguments(code, ("note", "n"))
    note_tokens = [token for notation in quoted_notes for token in _tokens(notation)]
    drum_hits = sum(sample_tokens.count(name) for name in DRUM_NAMES)
    multipliers = [int(value) for value in re.findall(r"\*\s*(\d+)", " ".join(sample_strings))]
    drum_density = drum_hits * math.prod(multipliers[:1]) if drum_hits else 0

    onsets = _audible_onsets(events or [])
    if events:
        instruments = sorted(
            {
                canonical_instrument(str(event["value"]["s"]).split(":", 1)[0])
                for event in onsets
                if event["value"].get("s")
            }
        )
    else:
        instruments = sorted(set(sample_tokens))
    active_cycles, distinct_cycles = _cycle_features(onsets)

    return {
        "tempo": infer_tempo(code),
        "instruments": instruments,
        "sample_counts": dict(Counter(sample_tokens)),
        "notes": note_tokens,
        "layer_count": _layer_count(code),
        "drum_density": drum_density,
        "event_count": len(events or []),
        "event_values": [event.get("value") for event in events or []],
        "active_cycles": active_cycles,
        "distinct_cycles": distinct_cycles,
        "functions": sorted(set(re.findall(r"\.?\b([A-Za-z_$][\w$]*)\s*\(", code))),
    }


def _advance_quote(char: str, quote: str, escaped: bool) -> tuple[str | None, bool]:
    if escaped:
        return quote, False
    if char == "\\":
        return quote, True
    if char == quote:
        return None, False
    return quote, False


def _delimiter_error(code: str) -> str | None:
    code = _strip_comments(code)
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for char in code:
        if quote:
            quote, escaped = _advance_quote(char, quote, escaped)
            continue
        if char in QUOTES:
            quote = char
            continue
        if char in DELIMITER_PAIRS:
            stack.append(char)
            continue
        if char in DELIMITER_PAIRS.values() and (not stack or DELIMITER_PAIRS[stack.pop()] != char):
            return f"unbalanced delimiter {char}"
    if quote:
        return "unterminated string"
    if stack:
        return f"unclosed delimiter {stack[-1]}"
    return None


def statically_valid(code: str) -> tuple[bool, str | None]:
    """Check balanced syntax and require a recognizable pattern constructor."""
    if not code.strip():
        return False, "empty Strudel code"
    if error := _delimiter_error(code):
        return False, error
    if not re.search(r"\b(?:s|sound|note|n|stack|cat|seq|silence)\s*\(", code):
        return False, "no recognizable Strudel pattern constructor"
    return True, None
