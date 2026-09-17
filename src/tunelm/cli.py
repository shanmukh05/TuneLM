"""Unified CLI for dataset generation and training/evaluation.

`scripts/generate.py` and `scripts/train.py` delegate here. Routing is inferred
from the loaded YAML so callers only pass `--config`.
"""

from __future__ import annotations

import argparse
import json

from tunelm.config import load_config, project_path
from tunelm.training_preflight import (
    dry_run_training,
    infer_pipeline,
    require_cuda_when_requested,
    validate_training_config,
)


def infer_generate_kind(config: dict) -> str:
    if config.get("providers") or config.get("models"):
        return "sft"
    return "rl"


def infer_train_kind(config: dict) -> str:
    if "tasks" in config and "predictions" in config:
        return "evaluate"
    return infer_pipeline(config)


def run_generate(config_path: str) -> None:
    config = load_config(config_path)
    kind = infer_generate_kind(config)
    if kind == "sft":
        from tunelm.sft_data.dataset import generate_dataset

        counts = generate_dataset(config)
        print(" ".join(f"{key}={value}" for key, value in counts.items()))
        return
    from tunelm.rl_data.dataset import generate_task_bank

    counts = generate_task_bank(config)
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


def run_train(
    config_path: str,
    *,
    dry_run: bool = False,
    checkpoint: str | None = None,
    limit: int | None = None,
) -> None:
    if dry_run and checkpoint:
        raise ValueError("--dry-run cannot be combined with --checkpoint")

    config = load_config(config_path)
    kind = infer_train_kind(config)

    if dry_run:
        if kind == "evaluate":
            tasks = project_path(config["tasks"])
            if not tasks.is_file():
                raise FileNotFoundError(f"evaluation task file not found: {tasks}")
            predictions = project_path(config["predictions"])
            line = (
                f"dry-run ok: evaluate tasks={config['tasks']} "
                f"predictions={config['predictions']} "
                f"output={config.get('output', 'results/evaluation.jsonl')}"
            )
            if not predictions.is_file():
                print(f"{line} (predictions missing; pass --checkpoint to generate)")
            else:
                print(line)
            return
        report = dry_run_training(config_path)
        print(
            f"dry-run ok: {report['pipeline']} model={report['model']} "
            f"dataset={report['dataset']} output={report['output_dir']}"
        )
        for warning in report.get("warnings", []):
            print(f"warning: {warning}")
        return
    if kind == "evaluate":
        if checkpoint:
            from tunelm.eval.predictor import generate_predictions

            count = generate_predictions(
                project_path(config["tasks"]),
                project_path(config["predictions"]),
                checkpoint,
                temperature=float(config.get("generation", {}).get("temperature", 0.0)),
                max_new_tokens=int(config.get("generation", {}).get("max_new_tokens", 1536)),
                limit=limit,
            )
            print(f"generated {count} predictions")
        from tunelm.eval.evaluator import evaluate_predictions

        print(json.dumps(evaluate_predictions(config), indent=2))
        return

    validate_training_config(config, config_path=config_path, strict_datasets=True)
    require_cuda_when_requested(config)
    if kind == "grpo":
        from tunelm.rl.trainer import train_rl

        train_rl(config)
        return
    from tunelm.sft.trainer import train_sft

    train_sft(config)


def generate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate TuneLM RL tasks or SFT data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_generate(args.config)


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate TuneLM checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and dataset paths without loading model weights",
    )
    parser.add_argument("--checkpoint", help="evaluate: generate predictions before scoring")
    parser.add_argument("--limit", type=int, help="evaluate: only generate this many predictions")
    args = parser.parse_args(argv)
    run_train(
        args.config,
        dry_run=args.dry_run,
        checkpoint=args.checkpoint,
        limit=args.limit,
    )
