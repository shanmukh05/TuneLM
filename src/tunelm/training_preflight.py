from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from tunelm.config import load_config, project_path, repo_root


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
    backend = str(config.get("strudel", {}).get("backend", "node")).lower()
    if backend not in {"node", "auto"}:
        return
    if shutil.which("node") is None:
        raise RuntimeError(
            "Node.js is required for Strudel backend=node/auto but was not found on PATH."
        )
    package = repo_root() / "node_modules" / "@strudel" / "core" / "package.json"
    if backend == "node" and not package.is_file():
        raise RuntimeError(
            "Strudel packages are missing. Run `npm install` from the repository root."
        )


def infer_pipeline(config: dict) -> str:
    if config.get("rewards"):
        return "grpo"
    return "sft"


def _base_config_errors(config: dict) -> list[str]:
    errors = []
    if "name" not in config.get("model", {}):
        errors.append("training config must define model.name")
    if "train_file" not in config.get("data", {}):
        errors.append("training config must define data.train_file")
    if "output_dir" not in config.get("training", {}):
        errors.append("training config must define training.output_dir")
    backend = config.get("training", {}).get("backend", "transformers")
    if backend != "transformers":
        errors.append(f"unsupported training backend: {backend}")
    return errors


def _validate_grpo(config: dict, *, strict: bool) -> list[str]:
    require_node_for_strudel(config)
    training = config["training"]
    effective_batch = (
        int(training.get("per_device_train_batch_size", 1))
        * int(training.get("gradient_accumulation_steps", 1))
        * int(training.get("num_processes", 1))
    )
    generations = int(training.get("num_generations", 8))
    if effective_batch % generations:
        raise ValueError(
            "GRPO effective batch size "
            f"({effective_batch}) must be divisible by num_generations ({generations})"
        )
    adapter = config.get("model", {}).get("adapter")
    if not adapter:
        raise ValueError("GRPO config must define model.adapter with the SFT checkpoint")
    if project_path(adapter).exists():
        return []
    message = f"SFT adapter not found yet: {adapter}"
    if strict:
        raise FileNotFoundError(message)
    return [message]


def require_training_runtime(config: dict) -> None:
    """Fail before model loading when required training integrations are unavailable."""
    require_cuda_when_requested(config)
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is not installed. Install a CUDA-matched PyTorch build.")
    if (
        config.get("model", {}).get("load_in_4bit")
        and importlib.util.find_spec("bitsandbytes") is None
    ):
        raise RuntimeError("bitsandbytes is required when model.load_in_4bit=true")
    report_to = config.get("training", {}).get("report_to", "none")
    if report_to == "wandb" and importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is required when training.report_to=wandb")


def validate_training_config(
    config: dict,
    *,
    config_path: str | Path | None = None,
    strict_datasets: bool = False,
    strict_adapter: bool = False,
) -> dict[str, Any]:
    """Validate a training config without loading any model weights."""
    path = Path(config_path or config.get("_config_path", ""))
    errors = _base_config_errors(config)
    if errors:
        raise ValueError("; ".join(errors))

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
        warnings.extend(_validate_grpo(config, strict=strict_adapter))
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
    report = validate_training_config(config, config_path=config_path, strict_datasets=True)
    report["status"] = "ok"
    return report
