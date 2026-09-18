"""Generate RL task banks in one pass or staged prompt-review passes.

When ``batch.enabled: true`` and ``prompt_author`` is configured, styled prompts
are authored through the Gemini Batch API. Legacy sync generation remains when
batch is disabled or when hybrid ``llm_fraction`` generation is used without
styled prompt authoring.

Set ``append: true`` to resume into an existing prompts or train JSONL file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tunelm.config import load_config, project_path
from tunelm.data.batch_common import batch_enabled
from tunelm.data.diversity import load_prompt_hashes
from tunelm.data.generation_progress import GenerationProgress
from tunelm.data.resume import existing_ids, resume_remaining
from tunelm.data.task_factory import generate_tasks, task_signature
from tunelm.io import append_jsonl, read_jsonl, write_jsonl
from tunelm.providers import require_prompt_router_from_config
from tunelm.rl_data.batch_workflow import batch_generate_prompts, batch_generate_tasks
from tunelm.rl_data.prompt_author import apply_rl_prompt_styles, prompt_author_config
from tunelm.schemas import parse_task

STAGES = {"prompts", "tasks", "full"}


def _generation_stage(config: dict) -> str:
    stage = str(config.get("stage", "full"))
    if stage not in STAGES:
        raise ValueError(f"Unknown RL stage: {stage!r}; expected one of {sorted(STAGES)}")
    return stage


def _prompts_path(config: dict) -> Path:
    return project_path(config.get("prompts_file", "datasets/rl/prompts.jsonl"))


def _refuse_overwrite(path: Path, config: dict) -> None:
    if config.get("append", False):
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; set append: true or remove it")


def _styled_generation_enabled(config: dict) -> bool:
    return bool(config.get("prompt_author")) or _generation_stage(config) in {"prompts", "tasks"}


def _batch_styled_generation_enabled(config: dict) -> bool:
    return batch_enabled(config) and _styled_generation_enabled(config)


def _forbidden_signatures(config: dict, existing_rows: list[dict]) -> set[str]:
    forbidden = (
        load_prompt_hashes(config.get("disjoint_from")) if config.get("disjoint_from") else set()
    )
    for row in existing_rows:
        forbidden.add(task_signature(row))
    return forbidden


def _generate_task_specs(
    config: dict,
    *,
    count: int,
    existing_rows: list[dict],
    progress: GenerationProgress | None = None,
) -> list:
    seed = int(config.get("seed", 42))
    mix = config.get("mix", {"compose": 0.6, "repair": 0.25, "edit": 0.15})
    forbidden = _forbidden_signatures(config, existing_rows)
    if _styled_generation_enabled(config):
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
        prompt_router = require_prompt_router_from_config(prompt_author_config(config))
        return apply_rl_prompt_styles(
            tasks,
            config,
            prompt_router=prompt_router,
            seed=seed,
            progress=progress,
        )
    tasks = generate_tasks(
        count,
        seed,
        mix,
        mode=str(config.get("mode", "procedural")),
        llm_fraction=float(config.get("llm_fraction", 0.0)),
        llm_config=config.get("llm"),
        id_prefix=str(config.get("id_prefix", "rl")),
        disjoint_from=config.get("disjoint_from"),
        min_unique_prompts=config.get("min_unique_prompts"),
    )
    random.Random(seed).shuffle(tasks)
    return tasks[:count]


def _write_tasks(path: Path, tasks, config: dict, existing_rows: list[dict]) -> int:
    rows = [task.model_dump(mode="json") for task in tasks]
    if config.get("append", False) and existing_rows:
        for row in rows:
            append_jsonl(path, row)
        return len(rows)
    return write_jsonl(path, rows)


def generate_prompt_bank(config: dict) -> dict[str, int]:
    """Write reviewable RL task rows without changing the final train path."""
    if _batch_styled_generation_enabled(config):
        return batch_generate_prompts(config)
    prompts_path = _prompts_path(config)
    _refuse_overwrite(prompts_path, config)
    remaining, existing = resume_remaining(config, prompts_path)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    progress = GenerationProgress("RL prompts", remaining)
    tasks = _generate_task_specs(config, count=remaining, existing_rows=existing, progress=progress)
    progress.close()
    written = _write_tasks(prompts_path, tasks, config, existing)
    return {"written": written, "skipped": len(existing)}


def finalize_task_bank(config: dict) -> dict[str, int]:
    """Copy a reviewed prompt bank into the final RL task output path."""
    prompts_path = _prompts_path(config)
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt bank not found: {prompts_path}")
    output = project_path(config["output"])
    _refuse_overwrite(output, config)
    remaining, existing = resume_remaining(config, output)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    seen = existing_ids(existing)
    rows: list[dict] = []
    for row in read_jsonl(prompts_path):
        task_id = row.get("id")
        if task_id in seen:
            continue
        rows.append(parse_task(row).model_dump(mode="json"))
        seen.add(str(task_id))
        if len(rows) >= remaining:
            break
    if config.get("append", False) and existing:
        for row in rows:
            append_jsonl(output, row)
        return {"written": len(rows), "skipped": len(existing)}
    return {"written": write_jsonl(output, rows), "skipped": 0}


def generate_task_bank(config: dict) -> dict[str, int]:
    stage = _generation_stage(config)
    if stage == "prompts":
        return generate_prompt_bank(config)
    if stage == "tasks":
        return finalize_task_bank(config)

    if _batch_styled_generation_enabled(config):
        return batch_generate_tasks(config)

    output = project_path(config["output"])
    _refuse_overwrite(output, config)
    remaining, existing = resume_remaining(config, output)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    progress = GenerationProgress("RL tasks", remaining)
    tasks = _generate_task_specs(config, count=remaining, existing_rows=existing, progress=progress)
    progress.close()
    written = _write_tasks(output, tasks, config, existing)
    return {"written": written, "skipped": len(existing)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a TuneLM RL task bank")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    counts = generate_task_bank(load_config(args.config))
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
