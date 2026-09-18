from __future__ import annotations

from functools import partial

from tunelm.config import project_path
from tunelm.experiment import write_experiment_metadata
from tunelm.models.adapters import attach_lora
from tunelm.models.loader import load_model_and_tokenizer
from tunelm.models.templates import training_record


def train_sft(config: dict):
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    data_path = str(project_path(config["data"]["train_file"]))
    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        partial(training_record, tokenizer=tokenizer), remove_columns=dataset.column_names
    )
    training = config.get("training", {})
    output_dir = str(project_path(training.get("output_dir", "checkpoints/sft")))
    args = SFTConfig(
        output_dir=output_dir,
        max_length=int(config.get("model", {}).get("max_seq_length", 4096)),
        completion_only_loss=True,
        packing=bool(training.get("packing", False)),
        num_train_epochs=float(training.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        warmup_ratio=float(training.get("warmup_ratio", 0.05)),
        logging_steps=int(training.get("logging_steps", 10)),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        report_to=training.get("report_to", "none"),
        run_name=training.get("run_name", "tunelm-sft"),
        seed=int(training.get("seed", 42)),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    write_experiment_metadata(output_dir, config, {"pipeline": "sft", "examples": len(dataset)})
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
