from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal

from tunelm.schemas import ExecutionResult
from tunelm.strudel.analyzer import analyze_code, statically_valid


Backend = Literal["auto", "node", "static"]


class StrudelExecutor:
    def __init__(
        self,
        *,
        backend: Backend = "node",
        timeout_seconds: float = 3.0,
        cycles: float = 2,
        max_events: int = 2048,
        bridge_path: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        self.cycles = cycles
        self.max_events = max_events
        self.bridge_path = Path(bridge_path or Path(__file__).with_name("bridge.js")).resolve()

    def run(self, code: str) -> ExecutionResult:
        started = time.perf_counter()
        if self.backend == "static":
            return self._static(code, started)
        try:
            result = self._node(code, started)
            if self.backend == "auto" and result.error and "Cannot find package" in result.error:
                return self._static(code, started, prefix="Node bridge unavailable; ")
            return result
        except FileNotFoundError:
            if self.backend == "node":
                return self._result(False, "Node.js is not installed", [], code, "node", started)
            return self._static(code, started, prefix="Node.js unavailable; ")
        except subprocess.TimeoutExpired:
            return self._result(False, "execution timed out", [], code, "node", started)

    def _node(self, code: str, started: float) -> ExecutionResult:
        request = json.dumps({"code": code, "cycles": self.cycles, "max_events": self.max_events})
        with tempfile.TemporaryDirectory(prefix="tunelm-strudel-") as cwd:
            completed = subprocess.run(
                ["node", "--disable-proto=delete", str(self.bridge_path)],
                input=request,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                cwd=cwd,
                env={"PATH": os.environ.get("PATH", "")},
                check=False,
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            error = (
                completed.stderr.strip() or completed.stdout.strip() or "bridge returned no JSON"
            )
            return self._result(False, error, [], code, "node", started)
        events = payload.get("events", [])
        return self._result(
            bool(payload.get("valid")), payload.get("error"), events, code, "node", started
        )

    def _static(self, code: str, started: float, prefix: str = "") -> ExecutionResult:
        valid, error = statically_valid(code)
        return self._result(
            valid, f"{prefix}{error}" if error else None, [], code, "static", started
        )

    @staticmethod
    def _result(valid, error, events, code, backend, started) -> ExecutionResult:
        return ExecutionResult(
            valid=valid,
            error=error,
            events=events,
            features=analyze_code(code, events),
            backend=backend,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def executor_from_config(config: dict) -> StrudelExecutor:
    values = config.get("strudel", config)
    return StrudelExecutor(
        backend=values.get("backend", "node"),
        timeout_seconds=float(values.get("timeout_seconds", 3)),
        cycles=float(values.get("cycles", 2)),
        max_events=int(values.get("max_events", 2048)),
    )
