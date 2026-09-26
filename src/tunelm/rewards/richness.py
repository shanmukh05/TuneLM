"""Reward compose outputs that keep sounding and change across the queried cycles.

Execution alone rewards a one-bar loop as much as a developing piece, and short
code is the safest way to execute, so RL drifts toward "too short". This score
uses the `active_cycles` / `distinct_cycles` features from `strudel.analyzer`.
"""

from __future__ import annotations

import math
from typing import Any

TARGET_DISTINCT_CYCLES = 4


def score_richness(features: dict[str, Any], cycles: float) -> float:
    """Mean of sound coverage and pattern variation over `cycles` queried cycles.

    ponytail: per-cycle event signatures are a proxy that random functions can
    inflate; keep the weight small and replace with an audio or judge reward once
    rendering exists.
    """
    window = max(1, math.ceil(cycles))
    coverage = min(1.0, (features.get("active_cycles") or 0) / window)
    target = min(TARGET_DISTINCT_CYCLES, window)
    if target <= 1:
        return coverage
    distinct = features.get("distinct_cycles") or 0
    variation = min(1.0, max(0.0, (distinct - 1) / (target - 1)))
    return (coverage + variation) / 2
