"""Terminal progress for dataset generation with per-model counters."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field


@dataclass
class ModelStats:
    ok: int = 0
    failed: int = 0
    active: int = 0


@dataclass
class GenerationProgress:
    """Render one progress line to stderr; safe for parallel teacher workers."""

    label: str
    total: int
    width: int = 32
    _done: int = 0
    _models: dict[str, ModelStats] = field(default_factory=dict)
    _failures: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = False

    def start(self, model_id: str) -> None:
        with self._lock:
            self._models.setdefault(model_id, ModelStats()).active += 1
            self._render()

    def record(self, model_id: str, *, ok: bool, reason: str | None = None) -> None:
        with self._lock:
            stats = self._models.setdefault(model_id, ModelStats())
            stats.active = max(0, stats.active - 1)
            if ok:
                stats.ok += 1
            else:
                stats.failed += 1
                if reason:
                    self._failures[reason] = self._failures.get(reason, 0) + 1
            self._done += 1
            self._render()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _bar(self) -> str:
        if self.total <= 0:
            return "-" * self.width
        filled = min(self.width, int(self.width * self._done / self.total))
        return "=" * filled + "-" * (self.width - filled)

    def _model_summary(self) -> str:
        if not self._models:
            return ""
        parts: list[str] = []
        for model_id in sorted(self._models):
            stats = self._models[model_id]
            active = f" ~{stats.active}" if stats.active else ""
            parts.append(f"{model_id}: {stats.ok} ok/{stats.failed} fail{active}")
        return " | ".join(parts)

    def _failure_summary(self) -> str:
        if not self._failures:
            return ""
        ranked = sorted(self._failures.items(), key=lambda item: item[1], reverse=True)
        return "fails " + ", ".join(f"{kind}={count}" for kind, count in ranked[:5])

    def _render(self) -> None:
        if self._closed:
            return
        summary = self._model_summary()
        failures = self._failure_summary()
        line = f"{self.label} [{self._bar()}] {self._done}/{self.total}"
        if summary:
            line = f"{line}  {summary}"
        if failures:
            line = f"{line} | {failures}"
        sys.stderr.write("\r" + line[: max(160, len(line))])
        sys.stderr.flush()
