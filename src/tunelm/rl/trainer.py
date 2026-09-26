"""Configure GRPO training for TuneLM task-bank rows.

Each task-bank row becomes a chat prompt plus the fixed reward columns TRL
forwards to `RewardAdapter`. Reward logic lives in `tunelm.rewards`.
"""

from __future__ import annotations

import json

from tunelm.config import project_path
from tunelm.data.datasets import load_jsonl_dataset
from tunelm.experiment import configure_wandb, write_experiment_metadata
from tunelm.models.templates import messages_for_task
from tunelm.rewards.constraints import stated_constraints
from tunelm.schemas import RLTask, parse_task

# Every row carries every column (None when not applicable); Arrow cannot write
# a mixed task bank whose rows have different keys.
REWARD_COLUMNS = (
    "task_type",
    "original_code",
    "expected_changes",
    "preserve",
    "reference_code",
)


def _parse_rl_task(row: dict) -> RLTask:
    payload = dict(row)
    if "task_prompt" in payload:
        payload["prompt"] = payload.pop("task_prompt")
    common = {"id", "task_type", "prompt", "constraints", "difficulty", "tags"}
    specific = {
        "compose": set(),
        "edit": {"original_code", "instruction", "expected_changes", "preserve"},
        "repair": {"broken_code", "reference_code", "corruption"},
    }[payload["task_type"]]
    return parse_task(
        {
            key: value
            for key, value in payload.items()
            if key in common | specific and value is not None
        }
    )


def prepare_rl_row(row: dict) -> dict:
    """Convert a task-bank row to a GRPO prompt plus uniform reward columns.

    Constraints the prompt never states are dropped so the policy is only
    rewarded for targets it can see.
    """
    task = _parse_rl_task(row)
    dumped = task.model_dump(mode="json")
    columns = {name: dumped.get(name) for name in REWARD_COLUMNS}
    constraints = stated_constraints(task.prompt, task.constraints)
    return {
        **columns,
        "constraints": constraints.model_dump(mode="json"),
        "prompt": messages_for_task(task),
    }


def _prepare_rl_json_row(row: dict) -> dict:
    return prepare_rl_row(json.loads(row["json"]))


def train_rl(config: dict):
    from trl import GRPOConfig, GRPOTrainer

    from tunelm.models.adapters import attach_lora
    from tunelm.models.loader import load_model_and_tokenizer
    from tunelm.rl.reward_adapter import RewardAdapter
    from tunelm.strudel.executor import executor_from_config

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    dataset = load_jsonl_dataset(project_path(config["data"]["train_file"]))
    dataset = dataset.map(_prepare_rl_json_row, remove_columns=["json"])
    rewards_config = config.get("rewards", {})
    adapter = RewardAdapter(executor_from_config(config))
    reward_funcs = []
    reward_weights = []
    for name, function in (
        ("format", adapter.format_reward),
        ("execution", adapter.execution_reward),
        ("constraints", adapter.constraint_reward),
        ("edit_consistency", adapter.edit_consistency_reward),
        ("repair_consistency", adapter.repair_consistency_reward),
        ("richness", adapter.richness_reward),
    ):
        values = rewards_config.get(name, {})
        if values.get("enabled", True):
            reward_funcs.append(function)
            reward_weights.append(float(values.get("weight", 1.0)))

    training = config.get("training", {})
    configure_wandb(config)
    output_dir = str(project_path(training.get("output_dir", "checkpoints/rl")))
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(training.get("learning_rate", 5e-6)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_generations=int(training.get("num_generations", 4)),
        max_completion_length=int(training.get("max_completion_length", 1536)),
        max_steps=int(training.get("max_steps", 500)),
        logging_steps=int(training.get("logging_steps", 5)),
        save_steps=int(training.get("save_steps", 50)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        bf16=bool(training.get("bf16", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        report_to=training.get("report_to", "none"),
        run_name=training.get("run_name", "tunelm-grpo"),
        seed=int(training.get("seed", 42)),
        reward_weights=reward_weights,
        chat_template_kwargs=training.get("chat_template_kwargs", {"enable_thinking": False}),
        mask_truncated_completions=bool(training.get("mask_truncated_completions", True)),
        log_completions=bool(training.get("log_completions", True)),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    write_experiment_metadata(output_dir, config, {"pipeline": "grpo", "tasks": len(dataset)})
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
