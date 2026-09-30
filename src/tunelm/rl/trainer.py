"""Configure GRPO training for TuneLM task-bank rows.

Each task-bank row becomes a chat prompt plus the fixed reward columns TRL
forwards to `RewardAdapter`. Reward logic lives in `tunelm.rewards`.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

from datasets import concatenate_datasets

from tunelm.config import project_path
from tunelm.data.datasets import load_jsonl_dataset
from tunelm.experiment import configure_wandb, write_experiment_metadata
from tunelm.models.templates import messages_for_task
from tunelm.rewards.constraints import stated_constraints
from tunelm.rewards.richness import is_short_loop_request
from tunelm.rl.reward_telemetry import RewardTelemetry
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
        "short_loop": is_short_loop_request(task.prompt),
        "prompt": messages_for_task(task),
    }


def _prepare_rl_json_row(row: dict) -> dict:
    return prepare_rl_row(json.loads(row["json"]))


def split_rl_dataset(dataset, validation_size: int, seed: int):
    """Hold out a reproducible, task-stratified subset before prompt mapping."""
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError("data.validation_size must be between 1 and the task count minus 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, line in enumerate(dataset["json"]):
        groups[json.loads(line)["task_type"]].append(index)
    targets = {name: validation_size * len(indices) / len(dataset) for name, indices in groups.items()}
    minimum = 1 if validation_size >= len(groups) else 0
    counts = {name: max(minimum, int(target)) for name, target in targets.items()}
    while sum(counts.values()) < validation_size:
        name = max(groups, key=lambda item: (targets[item] - counts[item], item))
        counts[name] += 1
    while sum(counts.values()) > validation_size:
        name = max(
            (item for item in groups if counts[item] > minimum),
            key=lambda item: (counts[item] - targets[item], item),
        )
        counts[name] -= 1
    rng = random.Random(seed)
    held_out: set[int] = set()
    for name in sorted(groups):
        rng.shuffle(groups[name])
        held_out.update(groups[name][: counts[name]])
    return (
        dataset.select([index for index in range(len(dataset)) if index not in held_out]),
        dataset.select(sorted(held_out)),
    )


def repeat_compose_rows(dataset, factor: int, seed: int):
    """Increase compose exposure without duplicating held-out tasks."""
    if factor < 1:
        raise ValueError("data.compose_repeat_factor must be at least 1")
    if factor == 1:
        return dataset
    indices = [
        index
        for index, line in enumerate(dataset["json"])
        if json.loads(line)["task_type"] == "compose"
    ]
    if not indices:
        raise ValueError("compose oversampling requires compose tasks")
    compose = dataset.select(indices)
    return concatenate_datasets([dataset, *([compose] * (factor - 1))]).shuffle(seed=seed)


def train_rl(config: dict):
    from trl import GRPOConfig, GRPOTrainer

    from tunelm.models.adapters import attach_lora
    from tunelm.models.loader import load_model_and_tokenizer
    from tunelm.rl.reward_adapter import RewardAdapter
    from tunelm.strudel.executor import executor_from_config

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    dataset = load_jsonl_dataset(project_path(config["data"]["train_file"]))
    training = config.get("training", {})
    validation_size = int(config.get("data", {}).get("validation_size", 0))
    if validation_size:
        dataset, eval_dataset = split_rl_dataset(
            dataset, validation_size, int(training.get("seed", 42))
        )
    else:
        eval_dataset = None
    dataset = repeat_compose_rows(
        dataset,
        int(config.get("data", {}).get("compose_repeat_factor", 1)),
        int(training.get("seed", 42)),
    )
    dataset = dataset.map(_prepare_rl_json_row, remove_columns=["json"])
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_prepare_rl_json_row, remove_columns=["json"])
    rewards_config = config.get("rewards", {})
    adapter = RewardAdapter(executor_from_config(config))
    reward_specs = []
    for name, function in (
        ("format", adapter.format_reward),
        ("execution", adapter.execution_reward),
        ("response_consistency", adapter.response_consistency_reward),
        ("constraints", adapter.constraint_reward),
        ("edit_consistency", adapter.edit_consistency_reward),
        ("repair_consistency", adapter.repair_consistency_reward),
        ("richness", adapter.richness_reward),
    ):
        values = rewards_config.get(name, {})
        if values.get("enabled", True):
            reward_specs.append((name, function, float(values.get("weight", 1.0))))
    telemetry = RewardTelemetry(
        [(name, function.__name__, weight) for name, function, weight in reward_specs]
    )
    reward_funcs = [
        telemetry.wrap(name, weight, function) for name, function, weight in reward_specs
    ]
    reward_weights = [weight for _, _, weight in reward_specs]

    class MonitoredGRPOTrainer(GRPOTrainer):
        def evaluate(self, *args, **kwargs):
            previous_mode = telemetry.mode
            telemetry.mode = "eval"
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                telemetry.mode = previous_mode

        def log(self, logs, *args, **kwargs):
            mode = "train" if self.model.training else "eval"
            telemetry.add_metrics(self._metrics[mode], mode)
            return super().log(logs, *args, **kwargs)

    configure_wandb(config)
    output_dir = str(project_path(training.get("output_dir", "checkpoints/rl")))
    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(training.get("learning_rate", 5e-6)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        num_generations=int(training.get("num_generations", 4)),
        num_generations_eval=int(training.get("num_generations_eval", 2)) if eval_dataset is not None else None,
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 2)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(training.get("eval_steps", 50)) if eval_dataset is not None else None,
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
        beta=float(training.get("beta", 0.0)),
        chat_template_kwargs=training.get("chat_template_kwargs", {"enable_thinking": False}),
        mask_truncated_completions=bool(training.get("mask_truncated_completions", True)),
        log_completions=bool(training.get("log_completions", True)),
    )
    trainer = MonitoredGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    write_experiment_metadata(
        output_dir,
        config,
        {
            "pipeline": "grpo",
            "tasks": len(dataset),
            "eval_tasks": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_steps": args.eval_steps if eval_dataset is not None else None,
        },
    )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
