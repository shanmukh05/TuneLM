"""Safe YAML configuration and strict JSON/JSONL artifact persistence.

Repository-relative writes reject archive destinations. Generation journals use
durable appends and an OS lock; canonical records remain human-readable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: str | Path = ".env.local") -> None:
    """Load literal KEY=value settings without shell execution or overriding the environment."""
    source = project_path(path)
    if not source.exists():
        return
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"Invalid environment assignment at {source.name}:{number}")
        try:
            value = _env_value(match[2])
        except ValueError:
            raise ValueError(f"Invalid environment quoting at {source.name}:{number}") from None
        os.environ.setdefault(match[1], value)


def _env_value(raw: str) -> str:
    # A # inside an unquoted key/URL is literal; only whitespace starts an inline comment.
    if not raw.startswith(("'", '"')):
        return re.split(r"\s+#", raw, maxsplit=1)[0].rstrip()
    parts = shlex.split(raw, comments=True)
    if len(parts) != 1:
        raise ValueError("Expected one quoted value")
    return parts[0]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_path(value: str | Path) -> Path:
    """Resolve output paths and keep archived assets immutable."""
    path = project_path(value)
    if path.is_relative_to(ROOT / "contents_v1"):
        raise ValueError("Writing into contents_v1/ is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path):
    return json.loads(project_path(path).read_text(), parse_constant=_bad_constant)


def load_config(path: str | Path) -> dict:
    """Load safe YAML mappings; generated artifacts continue to use JSON/JSONL."""
    import yaml

    try:
        value = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Configuration contains unsupported or non-finite values: {path}"
        ) from error
    return value


@contextmanager
def generation_lock(directory: str | Path):
    """Allow one writer per generation run; OS locks release after interruptions."""
    import fcntl

    with output_path(Path(directory) / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This generation run is already active") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read complete JSONL records, reporting the exact invalid line."""
    result = []
    with project_path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=_bad_constant)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                result.append(value)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return result


def recover_journal(path: str | Path) -> list[dict]:
    """Repair only an interrupted final JSONL line, then perform strict parsing."""
    dest = project_path(path)
    if not dest.exists():
        return []
    raw = dest.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        try:
            json.loads(raw[boundary:], parse_constant=_bad_constant)
            _atomic_text(dest, raw.decode("utf-8") + "\n")
        except ValueError:
            _atomic_text(dest, raw[:boundary].decode("utf-8"))
    return read_jsonl(dest)


def write_json(path: str | Path, value) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def write_jsonl(path: str | Path, rows) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    _atomic_text(path, text)


def _atomic_text(path: str | Path, text: str) -> None:
    dest = output_path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dest.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, value: dict) -> None:
    """Append a durable complete line; only an interrupted final line is recoverable."""
    dest = output_path(path)
    if dest.exists() and dest.stat().st_size:
        with dest.open("rb") as handle:
            handle.seek(-1, 2)
            complete = handle.read(1) == b"\n"
        if not complete:
            recover_journal(dest)
    with dest.open("ab+") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
