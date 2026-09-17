from __future__ import annotations

from tunelm.strudel.analyzer import analyze_code


def _overlap_score(before_values: list[str], after_values: list[str]) -> float:
    before_set = set(before_values)
    after_set = set(after_values)
    if not before_set:
        return 1.0
    return len(before_set & after_set) / len(before_set)


def _layer_count_score(before_count: int, after_count: int) -> float:
    if before_count == after_count:
        return 1.0
    # Valid repair for a blank stack slot often drops one layer while fixing syntax.
    if after_count == before_count - 1:
        return 0.8
    return 0.0


def score_repair_consistency(candidate_code: str, reference_code: str | None) -> float:
    if not reference_code:
        return 1.0
    before = analyze_code(reference_code)
    after = analyze_code(candidate_code)
    scores = [
        float(before["tempo"] == after["tempo"]),
        _overlap_score(before["instruments"], after["instruments"]),
        _overlap_score(before["notes"], after["notes"]),
        _layer_count_score(int(before["layer_count"] or 0), int(after["layer_count"] or 0)),
        float(before["drum_density"] == after["drum_density"]),
    ]
    return sum(scores) / len(scores)
