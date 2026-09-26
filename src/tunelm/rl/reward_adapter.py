"""TRL-compatible TuneLM rewards with a shared Strudel execution cache.

TRL calls each method with the completions plus every dataset column prepared by
`tunelm.rl.trainer.prepare_rl_row`. Scoring rules live in `tunelm.rewards`.
"""

from __future__ import annotations

from collections import OrderedDict

from tunelm.rewards.constraints import has_verifiable_constraints, score_constraints
from tunelm.rewards.edit_consistency import score_edit_consistency
from tunelm.rewards.format import score_format
from tunelm.rewards.repair_consistency import score_repair_consistency
from tunelm.rewards.richness import score_richness
from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


class RewardAdapter:
    """TRL-compatible rewards sharing an execution cache across components.

    Every component except format requires a playable result (executes and
    produces events), so silent programs cannot collect downstream reward.
    """

    def __init__(self, executor, max_cache_size: int = 128) -> None:
        self.executor = executor
        self.cache: OrderedDict[str, object] = OrderedDict()
        self.max_cache_size = max_cache_size

    def _execution(self, completion):
        text = extract_completion_text(completion)
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        try:
            response = parse_model_response(text)
            result = self.executor.run(response.strudel_code)
            result = result.model_copy(update={"events": []})
        except ResponseParseError:
            result = None
        self.cache[text] = result
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return result

    def _playable(self, completion):
        result = self._execution(completion)
        return result if result is not None and result.playable else None

    def format_reward(self, completions, **kwargs):
        return [score_format(value) for value in completions]

    def execution_reward(self, completions, **kwargs):
        return [float(self._playable(value) is not None) for value in completions]

    def constraint_reward(self, completions, constraints=None, **kwargs):
        constraints = constraints or [{}] * len(completions)
        output = []
        for index, value in enumerate(completions):
            task_constraints = constraints[index] if isinstance(constraints, list) else constraints
            if not has_verifiable_constraints(task_constraints):
                output.append(None)
                continue
            result = self._playable(value)
            output.append(score_constraints(result.features, task_constraints) if result else 0.0)
        return output

    def edit_consistency_reward(
        self,
        completions,
        original_code=None,
        expected_changes=None,
        preserve=None,
        **kwargs,
    ):
        output = []
        originals = original_code or [None] * len(completions)
        changes = expected_changes or [None] * len(completions)
        preserve_values = preserve or [[]] * len(completions)
        for index, completion in enumerate(completions):
            if originals[index] is None or changes[index] is None:
                output.append(None)
                continue
            if self._playable(completion) is None:
                output.append(0.0)
                continue
            try:
                response = parse_model_response(extract_completion_text(completion))
                task = {
                    "id": f"reward-edit-{index}",
                    "task_type": "edit",
                    "prompt": "edit",
                    "original_code": originals[index],
                    "instruction": "edit",
                    "expected_changes": changes[index],
                    "preserve": preserve_values[index] or [],
                }
                output.append(score_edit_consistency(response.strudel_code, task))
            except (ResponseParseError, ValueError):
                output.append(0.0)
        return output

    def repair_consistency_reward(self, completions, reference_code=None, **kwargs):
        references = reference_code or [None] * len(completions)
        output = []
        for index, completion in enumerate(completions):
            if references[index] is None:
                output.append(None)
                continue
            if self._playable(completion) is None:
                output.append(0.0)
                continue
            try:
                response = parse_model_response(extract_completion_text(completion))
                output.append(score_repair_consistency(response.strudel_code, references[index]))
            except ResponseParseError:
                output.append(0.0)
        return output

    def richness_reward(self, completions, task_type=None, **kwargs):
        """Compose-only; `None` for edit/repair and for the static backend (no events)."""
        types = task_type or [None] * len(completions)
        output = []
        for index, completion in enumerate(completions):
            if types[index] != "compose":
                output.append(None)
                continue
            result = self._playable(completion)
            if result is None:
                output.append(0.0)
            elif result.backend == "static":
                output.append(None)
            else:
                output.append(score_richness(result.features, self.executor.cycles))
        return output
