"""Launch credential-free bounded Node executions and validate their results.

There is no static fallback. This process boundary is not an OS security sandbox.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import time
from dataclasses import dataclass
from uuid import uuid4

from tunelm.io import ROOT, load_config
from tunelm.telemetry import summarize


@dataclass(frozen=True)
class Verifier:
    timeout: float = 5
    max_events: int = 20000
    heap_mb: int = 256

    def __post_init__(self):
        if not 0 < self.timeout <= 60 or not 64 <= self.heap_mb <= 4096:
            raise ValueError("Invalid verifier wall/heap limit")
        if not 1 <= self.max_events <= 50000:
            raise ValueError("Invalid verifier event limit")

    def request(
        self, code: str, *, operation: str = "verify", cycles: int = 16, key: str | None = None
    ) -> dict:
        """Run all requested cycles under a parent-enforced wall deadline."""
        if len(code.encode()) > 100000:
            return _failure("request", "CODE_LIMIT", "Code exceeds 100 KB")
        payload = {
            "schema_version": 2,
            "request_id": uuid4().hex,
            "operation": operation,
            "code": code,
            "cycles": cycles,
            "max_events": self.max_events,
            "key": key,
        }
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="tunelm-verifier-") as cwd:
            try:
                raw, returncode = _run(payload, cwd, self.timeout, self.heap_mb)
            except FileNotFoundError:
                return _failure("runtime_setup", "NODE_MISSING", "Node.js is not installed")
            except subprocess.TimeoutExpired:
                return _failure("timeout", "TIMEOUT", "Verifier wall deadline exceeded")
            except OutputLimit:
                return _failure("protocol", "OUTPUT_LIMIT", "Verifier output exceeds 32 MiB")
        response = _decode(raw, returncode)
        response["duration_ms"] = round((time.monotonic() - start) * 1000, 2)
        if response.get("valid") and operation == "verify":
            response["telemetry"] = summarize(response["events"], response["cycles"]).model_dump()
            _check_sounds(response)
        return response


class OutputLimit(RuntimeError):
    """Raised while reading pipes, before unbounded worker output is buffered."""


def _run(payload: dict, cwd: str, timeout: float, heap: int) -> tuple[str, int]:
    started = time.monotonic()
    with tempfile.TemporaryFile() as request:
        request.write(json.dumps(payload).encode())
        request.seek(0)
        with subprocess.Popen(
            [
                "node",
                f"--max-old-space-size={heap}",
                "--disable-proto=delete",
                str(ROOT / "verifier/runtime.mjs"),
            ],
            stdin=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        ) as child:
            try:
                raw = _collect(child, timeout)
                return raw, child.wait(timeout=max(0.01, timeout - (time.monotonic() - started)))
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


def _collect(child, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    result, total = bytearray(), 0
    with selectors.DefaultSelector() as selector:
        selector.register(child.stdout, selectors.EVENT_READ)
        selector.register(child.stderr, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(child.args, timeout)
            for key, _ in selector.select(remaining):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    raise OutputLimit()
                if key.fileobj is child.stdout:
                    result.extend(chunk)
    return result.decode("utf-8", errors="replace")


def _failure(stage: str, code: str, message: str) -> dict:
    return {
        "schema_version": 2,
        "valid": False,
        "stage": stage,
        "error_code": code,
        "error": message,
    }


def _decode(raw: str, returncode: int) -> dict:
    if len(raw.encode()) > 32 * 1024 * 1024:
        return _failure("protocol", "OUTPUT_LIMIT", "Verifier response exceeds 32 MiB")
    try:
        value = json.loads(raw)
    except ValueError:
        return _failure("runtime_setup", "WORKER_EXIT", f"Node exited without JSON ({returncode})")
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        return _failure("protocol", "INVALID_RESPONSE", "Invalid verifier response")
    return value


def _check_sounds(response: dict) -> None:
    catalog = load_config("configs/registry/sounds.yaml")
    unknown = sorted(set(response["telemetry"]["sound_sources"]) - set(catalog["sounds"]))
    response["sample_catalog_version"] = catalog["version"]
    response["unknown_sounds"] = unknown
    if unknown:
        response.update(
            valid=False,
            stage="sound_catalog",
            error_code="UNKNOWN_SOUND",
            error="Unknown sounds: " + ", ".join(unknown),
        )
