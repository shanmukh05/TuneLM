from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def aggregate(records: Iterable[dict]) -> dict:
    rows = list(records)
    components: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row["rewards"].items():
            components[name].append(float(value))
        by_type[row["task_type"]].append(float(row["total_reward"]))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "count": len(rows),
        "mean_total_reward": mean([row["total_reward"] for row in rows]),
        "metrics": {name: mean(values) for name, values in sorted(components.items())},
        "by_task_type": {
            name: {"count": len(values), "mean_total_reward": mean(values)}
            for name, values in sorted(by_type.items())
        },
    }
