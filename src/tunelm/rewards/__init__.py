"""Combine TuneLM reward components for offline evaluation.

Trainer-facing adapters live in `tunelm.rl.reward_adapter`; this module owns the
single normalized score used by evaluation and local diagnostics. Both apply the
same rules: only prompt-stated constraints count, and downstream components
require a playable (executing, non-silent) program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tunelm.rewards.constraints import (
    has_verifiable_constraints,
    score_constraints,
    stated_constraints,
)
from tunelm.rewards.edit_consistency import score_edit_consistency
from tunelm.rewards.format import score_format
from tunelm.rewards.repair_consistency import score_repair_consistency
from tunelm.rewards.response_consistency import score_response_consistency
from tunelm.rewards.richness import is_short_loop_request, score_richness
from tunelm.schemas import ComposeTask, EditTask, ExecutionResult, RepairTask, RLTask
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


@dataclass(slots=True)
class RewardResult:
    """Normalized total reward and the applicable raw components."""

    total: float
    components: dict[str, float]
    execution: ExecutionResult | None = None


class RewardEngine:
    """Score one completion using only rewards applicable to its task type."""

    def __init__(self, executor: StrudelExecutor, weights: dict[str, float] | None = None) -> None:
        self.executor = executor
        self.weights = weights or {
            "format": 0.1,
            "execution": 0.4,
            "response_consistency": 0.3,
            "constraints": 0.5,
            "edit_consistency": 0.5,
            "repair_consistency": 0.5,
            "richness": 0.2,
        }

    def score(self, completion: Any, task: RLTask) -> RewardResult:
        components = {"format": score_format(completion)}
        constraints = stated_constraints(task.prompt, task.constraints)
        result = None
        try:
            response = parse_model_response(extract_completion_text(completion))
        except ResponseParseError:
            response = None
        else:
            result = self.executor.run(response.strudel_code)
        playable = result if result is not None and result.playable else None
        candidate_code = response.strudel_code if playable else None
        components["execution"] = float(playable is not None)
        components["response_consistency"] = (
            score_response_consistency(response, playable.features) if playable and response else 0.0
        )
        if has_verifiable_constraints(constraints):
            components["constraints"] = (
                score_constraints(playable.features, constraints) if playable else 0.0
            )
        components.update(_consistency_components(task, candidate_code))
        components.update(self._richness_component(task, playable))
        active_weight = sum(self.weights.get(name, 0.0) for name in components)
        total = (
            sum(score * self.weights.get(name, 0.0) for name, score in components.items())
            / active_weight
            if active_weight
            else 0.0
        )
        return RewardResult(total=total, components=components, execution=result)

    def _richness_component(
        self, task: RLTask, playable: ExecutionResult | None
    ) -> dict[str, float]:
        if not isinstance(task, ComposeTask):
            return {}
        if playable is None:
            return {"richness": 0.0}
        if playable.backend == "static":
            return {}
        return {
            "richness": score_richness(
                playable.features,
                self.executor.cycles,
                short_loop=is_short_loop_request(task.prompt),
            )
        }


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
