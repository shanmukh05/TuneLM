from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from tunelm.config import load_config, project_path, training_script_for_config


def require_dataset(path: str | Path) -> Path:
    resolved = project_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {resolved}. Generate it first or point training.data.train_file "
            "to an existing JSONL file."
        )
    return resolved


def require_cuda_when_requested(config: dict) -> None:
    training = config.get("training", {})
    if not training.get("bf16", True):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "CUDA is not available but bf16 training is enabled. Use a GPU machine or set training.bf16=false."
    )


def require_node_for_strudel(config: dict) -> None:
    backend = str(config.get("strudel", {}).get("backend", "auto")).lower()
    if backend not in {"node", "auto"}:
        return
    if shutil.which("node") is None:
        raise RuntimeError(
            "Node.js is required for Strudel backend=node/auto but was not found on PATH."
        )


def infer_pipeline(config: dict) -> str:
    if config.get("rewards"):
        return "grpo"
    return "sft"


def validate_training_config(
    config: dict,
    *,
    config_path: str | Path | None = None,
    strict_datasets: bool = False,
) -> dict[str, Any]:
    """Validate a training config without loading any model weights."""
    path = Path(config_path or config.get("_config_path", ""))
    if path.name:
        training_script_for_config(path)
        strict_datasets = strict_datasets or "smoke" in path.name

    if "model" not in config or "name" not in config.get("model", {}):
        raise ValueError("training config must define model.name")
    if "data" not in config or "train_file" not in config.get("data", {}):
        raise ValueError("training config must define data.train_file")
    if "training" not in config or "output_dir" not in config.get("training", {}):
        raise ValueError("training config must define training.output_dir")

    pipeline = infer_pipeline(config)
    warnings: list[str] = []
    train_file = config["data"]["train_file"]
    try:
        dataset = require_dataset(train_file)
    except FileNotFoundError as error:
        if strict_datasets:
            raise
        warnings.append(str(error))
        dataset = project_path(train_file)

    if pipeline == "grpo":
        require_node_for_strudel(config)
        adapter = config.get("model", {}).get("adapter")
        if adapter and not project_path(adapter).exists():
            warnings.append(f"SFT adapter not found yet: {adapter}")
    elif not config.get("lora", {}).get("enabled", True):
        warnings.append("SFT config has lora.enabled=false; no new adapter will be trained")

    return {
        "pipeline": pipeline,
        "config_path": str(path) if path else None,
        "model": config["model"]["name"],
        "dataset": str(dataset),
        "output_dir": config["training"]["output_dir"],
        "warnings": warnings,
    }


def dry_run_training(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a training config. Does not import torch models or start training."""
    config = load_config(config_path)
    report = validate_training_config(config, config_path=config_path)
    report["status"] = "ok"
    return report
