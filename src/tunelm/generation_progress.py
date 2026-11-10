"""Share live terminal progress and operational settings across generation stages.

Progress writes to stderr so CLI results remain valid JSON on stdout. Display and
export cadence can change on resume without invalidating frozen generation inputs.
"""

import shutil
import sys

from pydantic import Field

from tunelm.schemas import Record

RUNTIME_FIELDS = {"progress", "save_every"}


class RunOptions(Record):
    """Accepted journals always persist immediately; save_every controls dataset exports."""

    progress: bool = True
    save_every: int = Field(default=10, ge=1)

    def identity_config(self) -> dict:
        return self.model_dump(exclude=RUNTIME_FIELDS)

    def runtime_settings(self) -> dict:
        return self.model_dump(include=RUNTIME_FIELDS)


class GenerationProgress:
    """Display completed jobs separately from accepted records, including resumed work."""

    def __init__(
        self, label: str, total: int, *, accepted: int, completed: int, attempts: int, enabled: bool
    ):
        self.label, self.total = label, total
        self.accepted, self.completed, self.attempts = accepted, completed, attempts
        self.enabled = enabled
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.detail = ""
        self.width = 0

    def __enter__(self):
        self._render()
        return self

    def __exit__(self, *_):
        if self.enabled and self.interactive:
            print(file=self.stream, flush=True)

    def attempt(self, model_id: str, retry: int, limit: int) -> None:
        """Show the active model and correction attempt before waiting for the API."""
        self.attempts += 1
        self.detail = f"{model_id} try {retry + 1}/{limit}"
        self._render()

    def advance(self, accepted: int) -> None:
        self.completed += 1
        self.accepted = accepted
        self.detail = ""
        self._render()

    def finish(self, status: str) -> None:
        self.detail = status
        self._render()

    def snapshot(self, status: str) -> dict:
        return {
            "status": status,
            "completed_jobs": self.completed,
            "total_jobs": self.total,
            "accepted": self.accepted,
            "attempts": self.attempts,
        }

    def _render(self):
        if not self.enabled:
            return
        filled = min(16, 16 * self.completed // max(1, self.total))
        bar = "#" * filled + "-" * (16 - filled)
        line = (
            f"{self.label} [{bar}] {self.completed}/{self.total} jobs | "
            f"accepted {self.accepted} | failed {self.completed - self.accepted} | "
            f"attempts {self.attempts}"
        )
        if self.detail:
            line += f" | {self.detail}"
        if self.interactive:
            columns = max(20, shutil.get_terminal_size().columns - 1)
            line = line[:columns]
            print("\r" + line.ljust(min(self.width, columns)), end="", file=self.stream, flush=True)
            self.width = len(line)
        else:
            print(line, file=self.stream, flush=True)
