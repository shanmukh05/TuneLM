"""Combine TuneLM reward components for offline evaluation.

Trainer-facing adapters live in `tunelm.rl.trainer`; this module owns the single
normalized score used by evaluation and local diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tunelm.rewards.constraints import has_verifiable_constraints, score_constraints
from tunelm.rewards.edit_consistency import score_edit_consistency
from tunelm.rewards.format import score_format
from tunelm.rewards.repair_consistency import score_repair_consistency
from tunelm.schemas import EditTask, RepairTask, RLTask
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


@dataclass(slots=True)
class RewardResult:
    """Normalized total reward and the applicable raw components."""

    total: float
    components: dict[str, float]


class RewardEngine:
    """Score one completion using only rewards applicable to its task type."""

    def __init__(self, executor: StrudelExecutor, weights: dict[str, float] | None = None) -> None:
        self.executor = executor
        self.weights = weights or {
            "format": 0.1,
            "execution": 0.4,
            "constraints": 0.5,
            "edit_consistency": 0.5,
            "repair_consistency": 0.5,
        }

    def score(self, completion: Any, task: RLTask) -> RewardResult:
        components = {"format": score_format(completion)}
        constraints_apply = has_verifiable_constraints(task.constraints)
        result = None
        candidate_code = None
        try:
            response = parse_model_response(extract_completion_text(completion))
        except ResponseParseError:
            pass
        else:
            result = self.executor.run(response.strudel_code)
            candidate_code = response.strudel_code if result.valid else None
        components["execution"] = float(bool(result and result.valid))
        if constraints_apply:
            components["constraints"] = (
                score_constraints(result.features, task.constraints)
                if result and result.valid
                else 0.0
            )
        components.update(_consistency_components(task, candidate_code))
        active_weight = sum(self.weights.get(name, 0.0) for name in components)
        total = (
            sum(score * self.weights.get(name, 0.0) for name, score in components.items())
            / active_weight
            if active_weight
            else 0.0
        )
        return RewardResult(total=total, components=components)


def _consistency_components(task: RLTask, candidate_code: str | None) -> dict[str, float]:
    if isinstance(task, EditTask):
        score = score_edit_consistency(candidate_code, task) if candidate_code else 0.0
        return {"edit_consistency": score}
    if isinstance(task, RepairTask):
        score = (
            score_repair_consistency(candidate_code, task.reference_code) if candidate_code else 0.0
        )
        return {"repair_consistency": score}
    return {}


__all__ = ["RewardEngine", "RewardResult"]
