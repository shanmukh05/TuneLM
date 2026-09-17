"""Extract verifier features and perform lightweight Strudel syntax checks.

The Node executor remains authoritative for execution; this module provides the
fast, deterministic analysis shared by data generation and rewards.
"""

from __future__ import annotations

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
    setcpm = re.search(r"\bsetcpm\s*\(\s*([0-9.]+)(?:\s*/\s*([0-9.]+))?\s*\)", code)
    if setcpm:
        numerator = float(setcpm.group(1))
        divisor = float(setcpm.group(2) or 1)
        return numerator / divisor * 4
    cps = re.search(r"\bsetcps\s*\(\s*([0-9.]+)\s*\)", code)
    return float(cps.group(1)) * 240 if cps else None


def analyze_code(code: str, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sample_strings = _string_arguments(code, SAMPLE_CALLS)
    sample_tokens = [token for notation in sample_strings for token in _tokens(notation)]
    quoted_notes = _string_arguments(code, ("note", "n"))
    note_tokens = [token for notation in quoted_notes for token in _tokens(notation)]
    instruments = sorted(set(sample_tokens))
    drum_hits = sum(sample_tokens.count(name) for name in DRUM_NAMES)
    multipliers = [int(value) for value in re.findall(r"\*\s*(\d+)", " ".join(sample_strings))]
    drum_density = drum_hits * math.prod(multipliers[:1]) if drum_hits else 0

    layer_count = 1 if code.strip() else 0
    stack = re.search(r"\bstack\s*\(", code)
    if stack:
        # Static, conservative estimate; the executor's event stream remains the
        # source of truth for temporal metrics.
        layer_count = max(2, len(sample_strings), len(quoted_notes))

    event_values = [event.get("value") for event in events or []]
    return {
        "tempo": infer_tempo(code),
        "instruments": instruments,
        "sample_counts": dict(Counter(sample_tokens)),
        "notes": note_tokens,
        "layer_count": layer_count,
        "drum_density": drum_density,
        "event_count": len(events or []),
        "event_values": event_values,
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
