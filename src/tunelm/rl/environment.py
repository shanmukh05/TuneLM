from __future__ import annotations

from tunelm.schemas import RLTask
from tunelm.strudel.executor import StrudelExecutor


class StrudelEnvironment:
    """Small stateful facade for future tool-using RL experiments."""

    def __init__(self, executor: StrudelExecutor) -> None:
        self.executor = executor
        self.task: RLTask | None = None
        self.last_result = None

    def reset(self, task: RLTask) -> str:
        self.task = task
        self.last_result = None
        return task.prompt

    def execute_strudel(self, code: str) -> dict:
        """Execute Strudel and return validity, error, events, and features."""
        self.last_result = self.executor.run(code)
        return self.last_result.model_dump()

    def get_reward(self) -> float:
        return float(bool(self.last_result and self.last_result.valid))
