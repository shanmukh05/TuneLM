"""Reward compose outputs that keep sounding and change across the queried cycles.

Execution alone rewards a one-bar loop as much as a developing piece, and short
code is the safest way to execute, so RL drifts toward "too short". This score
uses the `active_cycles` / `distinct_cycles` features from `strudel.analyzer`.
"""

from __future__ import annotations

import math
from typing import Any

TARGET_DISTINCT_CYCLES = 8


def is_short_loop_request(prompt: str) -> bool:
    lowered = prompt.casefold()
    return any(term in lowered for term in ("one-bar", "one bar", "short loop", "simple loop"))


def score_richness(features: dict[str, Any], cycles: float, *, short_loop: bool = False) -> float:
    """Score audible coverage and musical variation over the queried cycles.

    Per-cycle event signatures are a structural proxy; listening evaluation is
    still needed to judge musical quality.
    """
    window = max(1, math.ceil(cycles))
    coverage = min(1.0, (features.get("active_cycles") or 0) / window)
    if short_loop:
        return coverage
    target = min(TARGET_DISTINCT_CYCLES, window)
    if target <= 1:
        return coverage
    distinct = features.get("distinct_cycles") or 0
    variation = min(1.0, max(0.0, (distinct - 1) / (target - 1)))
    return coverage * variation
