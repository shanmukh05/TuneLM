"""Counts-aware GRPO reward metrics for mixed compose/edit/repair batches."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class _Moments:
    count: int = 0
    total: float = 0.0
    squares: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.squares += value * value

    @property
    def mean(self) -> float:
        return self.total / self.count

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.squares / self.count - self.mean**2))


@dataclass
class _Window:
    samples: int = 0
    task_counts: Counter[str] = field(default_factory=Counter)
    rewards: dict[str, _Moments] = field(default_factory=lambda: defaultdict(_Moments))
    task_rewards: dict[tuple[str, str], _Moments] = field(
        default_factory=lambda: defaultdict(_Moments)
    )
    weighted_total: float = 0.0
    unweighted_total: float = 0.0
    task_weighted_total: Counter[str] = field(default_factory=Counter)


class RewardTelemetry:
    """Accumulate raw reward outputs without averaging batches containing only None."""

    def __init__(self, specs: list[tuple[str, str, float]]) -> None:
        if not specs:
            raise ValueError("GRPO needs at least one enabled reward")
        self.specs = specs
        self.first_name = specs[0][0]
        self.mode = "train"
        self.windows = {"train": _Window(), "eval": _Window()}

    def wrap(self, name: str, weight: float, function):
        @wraps(function)
        def monitored(completions, **kwargs):
            scores = function(completions, **kwargs)
            self.record(name, weight, scores, kwargs.get("task_type"))
            return scores

        return monitored

    def record(self, name: str, weight: float, scores, task_types) -> None:
        if not isinstance(task_types, list) or len(task_types) != len(scores):
            raise ValueError("GRPO reward calls must include one task_type per completion")
        window = self.windows[self.mode]
        if name == self.first_name:
            window.samples += len(scores)
            window.task_counts.update(task_types)
        for task_type, score in zip(task_types, scores, strict=True):
            if score is None:
                continue
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"non-finite {name} reward for {task_type}")
            window.rewards[name].add(value)
            window.task_rewards[task_type, name].add(value)
            window.weighted_total += weight * value
            window.unweighted_total += value
            window.task_weighted_total[task_type] += weight * value

    def add_metrics(self, metrics: dict[str, list[float]], mode: str) -> None:
        window = self.windows[mode]
        if not window.samples:
            return
        metrics["objective/weighted_reward"] = [window.weighted_total / window.samples]
        metrics["objective/unweighted_reward"] = [window.unweighted_total / window.samples]
        metrics["objective/sample_count"] = [float(window.samples)]
        for name, function_name, _ in self.specs:
            moments = window.rewards[name]
            mean_key = f"rewards/{function_name}/mean"
            std_key = f"rewards/{function_name}/std"
            if moments.count:
                metrics[mean_key] = [moments.mean]
                metrics[std_key] = [moments.std]
            else:
                metrics.pop(mean_key, None)
                metrics.pop(std_key, None)
            metrics[f"reward_terms/{name}/applicable_count"] = [float(moments.count)]
        for task_type, count in window.task_counts.items():
            prefix = f"task/{task_type}"
            metrics[f"{prefix}/count"] = [float(count)]
            metrics[f"{prefix}/weighted_reward"] = [
                window.task_weighted_total[task_type] / count
            ]
            for name, _, _ in self.specs:
                moments = window.task_rewards[task_type, name]
                if moments.count:
                    metrics[f"{prefix}/rewards/{name}/mean"] = [moments.mean]
                    metrics[f"{prefix}/rewards/{name}/applicable_count"] = [
                        float(moments.count)
                    ]
        self.windows[mode] = _Window()
