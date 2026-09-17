"""Batch Gemini workflow for RL task prompt authoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tunelm.config import project_path
from tunelm.data.batch_common import batch_settings, refuse_overwrite, require_single_gemini_model, run_batch
from tunelm.data.diversity import load_prompt_hashes
from tunelm.data.resume import resume_remaining
from tunelm.data.task_factory import generate_tasks, task_signature
from tunelm.io import append_jsonl, write_jsonl
from tunelm.rl_data.prompt_author import prompt_author_config
from tunelm.rl_data.task_planner import plan_rl_styled_prompts, tasks_from_styled_prompt_results
from tunelm.schemas import ModelProvenance


def _prompts_path(config: dict[str, Any]) -> Path:
    return project_path(config.get("prompts_file", "datasets/rl/prompts.jsonl"))


def _output_path(config: dict[str, Any]) -> Path:
    return project_path(config["output"])


def _forbidden_signatures(config: dict[str, Any], existing_rows: list[dict]) -> set[str]:
    forbidden = load_prompt_hashes(config.get("disjoint_from")) if config.get("disjoint_from") else set()
    for row in existing_rows:
        forbidden.add(task_signature(row))
    return forbidden


def _procedural_skeletons(
    config: dict[str, Any],
    *,
    count: int,
    existing_rows: list[dict],
) -> list:
    seed = int(config.get("seed", 42))
    mix = config.get("mix", {"compose": 0.6, "repair": 0.25, "edit": 0.15})
    forbidden = _forbidden_signatures(config, existing_rows)
    tasks = generate_tasks(
        count,
        seed,
        mix,
        mode="procedural",
        id_prefix=str(config.get("id_prefix", "rl")),
        disjoint_from=None,
        min_unique_prompts=config.get("min_unique_prompts"),
    )
    if forbidden:
        tasks = [task for task in tasks if task_signature(task) not in forbidden][:count]
    return tasks


def _prompt_author_model(config: dict[str, Any]) -> tuple[str, str, str]:
    return require_single_gemini_model(prompt_author_config(config), label="RL prompt authoring")


def _style_tasks_batch(config: dict[str, Any], tasks: list) -> list:
    if not tasks:
        return []
    seed = int(config.get("seed", 42))
    specs = plan_rl_styled_prompts(tasks, config, seed=seed)
    model_id, provider, model = _prompt_author_model(config)
    texts = run_batch(
        config,
        [spec.request for spec in specs],
        display_name="tunelm-rl-prompts",
        model=model,
    )
    author = ModelProvenance(model_id=model_id, provider=provider, model=model)
    return tasks_from_styled_prompt_results(specs, texts, prompt_author=author)


def _write_tasks(path: Path, tasks, config: dict[str, Any], existing_rows: list[dict]) -> int:
    rows = [task.model_dump(mode="json") for task in tasks]
    if config.get("append", False) and existing_rows:
        for row in rows:
            append_jsonl(path, row)
        return len(rows)
    return write_jsonl(path, rows)


def batch_generate_prompts(config: dict[str, Any]) -> dict[str, int]:
    """Batch-style procedural RL tasks and write prompts_file."""
    prompts_path = _prompts_path(config)
    refuse_overwrite(prompts_path, config)
    remaining, existing = resume_remaining(config, prompts_path)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    skeletons = _procedural_skeletons(config, count=remaining, existing_rows=existing)
    tasks = _style_tasks_batch(config, skeletons)
    written = _write_tasks(prompts_path, tasks, config, existing)
    return {"written": written, "skipped": len(existing)}


def batch_generate_tasks(config: dict[str, Any]) -> dict[str, int]:
    """Batch-style RL tasks and write output directly."""
    output = _output_path(config)
    refuse_overwrite(output, config)
    remaining, existing = resume_remaining(config, output)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    skeletons = _procedural_skeletons(config, count=remaining, existing_rows=existing)
    tasks = _style_tasks_batch(config, skeletons)
    written = _write_tasks(output, tasks, config, existing)
    return {"written": written, "skipped": len(existing)}
