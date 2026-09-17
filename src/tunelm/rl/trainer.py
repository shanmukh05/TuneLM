"""Configure GRPO training and expose TRL-compatible TuneLM rewards.

`RewardAdapter` shares execution results across reward components so each model
completion is evaluated by Strudel at most once per cache lifetime.
"""

from __future__ import annotations

from collections import OrderedDict

from tunelm.config import project_path
from tunelm.experiment import write_experiment_metadata
from tunelm.models.adapters import attach_lora
from tunelm.models.loader import load_model_and_tokenizer
from tunelm.models.templates import messages_for_task
from tunelm.rewards.constraints import has_verifiable_constraints, score_constraints
from tunelm.rewards.edit_consistency import score_edit_consistency
from tunelm.rewards.format import score_format
from tunelm.rewards.repair_consistency import score_repair_consistency
from tunelm.schemas import parse_task
from tunelm.strudel.executor import executor_from_config
from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


class RewardAdapter:
    """TRL-compatible rewards sharing an execution cache across components."""

    def __init__(self, executor, max_cache_size: int = 4096) -> None:
        self.executor = executor
        self.cache: OrderedDict[str, object] = OrderedDict()
        self.max_cache_size = max_cache_size

    def _execution(self, completion):
        text = extract_completion_text(completion)
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        try:
            response = parse_model_response(text)
            result = self.executor.run(response.strudel_code)
        except ResponseParseError:
            result = None
        self.cache[text] = result
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return result

    def format_reward(self, completions, **kwargs):
        return [score_format(value) for value in completions]

    def execution_reward(self, completions, **kwargs):
        return [
            float(bool((result := self._execution(value)) and result.valid))
            for value in completions
        ]

    def constraint_reward(self, completions, constraints=None, **kwargs):
        constraints = constraints or [{}] * len(completions)
        output = []
        for index, value in enumerate(completions):
            task_constraints = constraints[index] if isinstance(constraints, list) else constraints
            if not has_verifiable_constraints(task_constraints):
                output.append(None)
                continue
            result = self._execution(value)
            if not result or not result.valid:
                output.append(0.0)
                continue
            score = score_constraints(result.features, task_constraints)
            output.append(score)
        return output

    def edit_consistency_reward(
        self,
        completions,
        original_code=None,
        expected_changes=None,
        preserve=None,
        **kwargs,
    ):
        output = []
        originals = original_code or [None] * len(completions)
        changes = expected_changes or [None] * len(completions)
        preserve_values = preserve or [[]] * len(completions)
        for index, completion in enumerate(completions):
            if originals[index] is None or changes[index] is None:
                output.append(None)
                continue
            execution = self._execution(completion)
            if not execution or not execution.valid:
                output.append(0.0)
                continue
            try:
                response = parse_model_response(extract_completion_text(completion))
                task = {
                    "id": f"reward-edit-{index}",
                    "task_type": "edit",
                    "prompt": "edit",
                    "original_code": originals[index],
                    "instruction": "edit",
                    "expected_changes": changes[index],
                    "preserve": preserve_values[index],
                }
                output.append(score_edit_consistency(response.strudel_code, task))
            except (ResponseParseError, ValueError):
                output.append(0.0)
        return output

    def repair_consistency_reward(self, completions, reference_code=None, **kwargs):
        references = reference_code or [None] * len(completions)
        output = []
        for index, completion in enumerate(completions):
            if references[index] is None:
                output.append(None)
                continue
            execution = self._execution(completion)
            if not execution or not execution.valid:
                output.append(0.0)
                continue
            try:
                response = parse_model_response(extract_completion_text(completion))
                output.append(score_repair_consistency(response.strudel_code, references[index]))
            except ResponseParseError:
                output.append(0.0)
        return output


def prepare_rl_row(row: dict) -> dict:
    """Convert a task-bank row, including Arrow's nullable union columns, to a prompt."""
    payload = dict(row)
    if "task_prompt" in payload:
        payload["prompt"] = payload.pop("task_prompt")
    common = {"id", "task_type", "prompt", "constraints", "difficulty", "tags"}
    specific = {
        "compose": set(),
        "edit": {"original_code", "instruction", "expected_changes", "preserve"},
        "repair": {"broken_code", "reference_code", "corruption"},
    }[payload["task_type"]]
    task = parse_task(
        {
            key: value
            for key, value in payload.items()
            if key in common | specific and value is not None
        }
    )
    return {"prompt": messages_for_task(task)}


def train_rl(config: dict):
    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc

    model, tokenizer = load_model_and_tokenizer(config)
    model = attach_lora(model, config)
    dataset = load_dataset(
        "json", data_files=str(project_path(config["data"]["train_file"])), split="train"
    )

    dataset = dataset.rename_column("prompt", "task_prompt")

    dataset = dataset.map(prepare_rl_row)
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
    ):
        values = rewards_config.get(name, {})
        if values.get("enabled", True):
            reward_funcs.append(function)
            reward_weights.append(float(values.get("weight", 1.0)))

    training = config.get("training", {})
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
        bf16=bool(training.get("bf16", True)),
        report_to=training.get("report_to", "none"),
        run_name=training.get("run_name", "tunelm-grpo"),
        seed=int(training.get("seed", 42)),
        reward_weights=reward_weights,
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
