from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def configure_wandb(config: dict[str, Any]) -> None:
    """Apply W&B settings from training YAML when report_to=wandb.

    Shell env vars win when already set so JarvisLabs/.env can override per run.
    """
    training = config.get("training", {})
    if training.get("report_to") != "wandb":
        return
    for env_key, config_key, default in (
        ("WANDB_PROJECT", "wandb_project", "tunelm"),
        ("WANDB_ENTITY", "wandb_entity", None),
        ("WANDB_RUN_GROUP", "wandb_group", None),
    ):
        value = training.get(config_key, default)
        if value and not os.environ.get(env_key):
            os.environ[env_key] = str(value)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_experiment_metadata(
    output_dir: str | Path, config: dict[str, Any], extra: dict[str, Any] | None = None
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        **(extra or {}),
    }
    path = output / "experiment.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
