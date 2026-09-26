from __future__ import annotations

import json
from functools import partial

from tunelm.config import project_path
from tunelm.data.datasets import load_sft_datasets
from tunelm.experiment import configure_wandb, write_experiment_metadata
from tunelm.models.templates import training_record


def _training_record_from_json(row: dict, *, tokenizer) -> dict:
    return training_record(json.loads(row["json"]), tokenizer)


def _map_sft_dataset(dataset, *, tokenizer):
    return dataset.map(
        partial(_training_record_from_json, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
    )


def train_sft(config: dict):
    from trl import SFTConfig, SFTTrainer

    from tunelm.models.adapters import attach_lora
    from tunelm.models.loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    train_raw, eval_raw = load_sft_datasets(config)
    train_dataset = _map_sft_dataset(train_raw, tokenizer=tokenizer)
    eval_dataset = _map_sft_dataset(eval_raw, tokenizer=tokenizer) if eval_raw is not None else None

    training = config.get("training", {})
    configure_wandb(config)
    output_dir = str(project_path(training.get("output_dir", "checkpoints/sft")))
    per_device_batch = int(training.get("per_device_train_batch_size", 2))
    grad_accum = int(training.get("gradient_accumulation_steps", 8))
    num_epochs = float(training.get("num_train_epochs", 2))
    steps_per_epoch = max(1, len(train_dataset) // (per_device_batch * grad_accum))
    total_steps = max(1, int(num_epochs * steps_per_epoch))
    warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    eval_count = int(training.get("eval_count", 10))
    eval_steps = max(1, total_steps // eval_count)
    save_steps = int(training.get("save_steps", 100))
    if eval_dataset is not None and save_steps % eval_steps != 0:
        save_steps = eval_steps * max(1, save_steps // eval_steps)
        if save_steps < eval_steps:
            save_steps = eval_steps
    args_kwargs: dict = {
        "output_dir": output_dir,
        "max_length": int(config.get("model", {}).get("max_seq_length", 4096)),
        "completion_only_loss": True,
        "packing": bool(training.get("packing", False)),
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "warmup_steps": warmup_steps,
        "logging_steps": int(training.get("logging_steps", 10)),
        "save_steps": save_steps,
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "bf16": bool(training.get("bf16", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "report_to": training.get("report_to", "none"),
        "run_name": training.get("run_name", "tunelm-sft"),
        "seed": int(training.get("seed", 42)),
    }
    if eval_dataset is not None:
        args_kwargs.update(
            {
                "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
                "eval_strategy": "steps",
                "eval_steps": eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    args = SFTConfig(**args_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    write_experiment_metadata(
        output_dir,
        config,
        {
            "pipeline": "sft",
            "examples": len(train_dataset),
            "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_steps": eval_steps if eval_dataset is not None else None,
        },
    )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
