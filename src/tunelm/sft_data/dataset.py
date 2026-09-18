"""Generate validated SFT datasets via batch Gemini jobs or legacy sync paths.

Batch workflow (default when ``batch.enabled: true``):

1. ``stage: prompts`` — batch prompt authoring → ``prompts_file``
2. ``stage: samples`` — batch teacher generation → ``samples_file``
3. ``stage: validate`` — Strudel validation, set ``metadata.valid``, export valid rows
4. ``stage: regenerate`` — batch retry failed samples, then re-validate
5. ``stage: full`` — run the full batch loop above

Legacy sync stages ``responses`` and inline ``full`` remain when ``batch.enabled: false``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tunelm.config import load_config, project_path
from tunelm.data.generation_progress import GenerationProgress
from tunelm.data.mix import effective_mix, filter_tasks_by_type
from tunelm.data.resume import existing_ids, next_index, resume_remaining
from tunelm.io import append_jsonl, read_jsonl, write_jsonl
from tunelm.providers import require_prompt_router_from_config, router_from_config
from tunelm.schemas import parse_task
from tunelm.data.batch_common import batch_enabled
from tunelm.sft_data.batch_workflow import (
    batch_generate_full,
    batch_generate_prompts,
    batch_generate_samples,
    batch_regenerate_failed,
    batch_validate_samples,
)
from tunelm.sft_data.example_builder import build_sft_example
from tunelm.sft_data.parallel import generate_teacher_examples_parallel, parallel_models_enabled
from tunelm.sft_data.solution_generator import generate_solution
from tunelm.sft_data.task_generator import generate_sft_tasks
from tunelm.sft_data.validator import (
    classify_failure,
    minimum_task_consistency_score,
    validate_solution,
)
from tunelm.strudel.executor import executor_from_config

STAGES = {"prompts", "samples", "validate", "regenerate", "responses", "full"}


def _generation_stage(config: dict) -> str:
    stage = str(config.get("stage", "full"))
    if stage == "responses":
        return "samples"
    if stage not in STAGES:
        raise ValueError(f"Unknown SFT stage: {stage!r}; expected one of {sorted(STAGES)}")
    return stage


def _prompts_path(config: dict) -> Path:
    return project_path(config.get("prompts_file", "datasets/sft/prompts.jsonl"))


def _compose_config(config: dict) -> dict:
    return {**config, **config.get("compose", {})}


def _refuse_overwrite(path: Path, config: dict) -> None:
    if config.get("append", False):
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; set append: true or remove it")


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return existing_ids(read_jsonl(path))


def _generate_sft_task_specs(
    config: dict,
    *,
    count: int,
    seed: int,
    compose_start_index: int,
    progress: GenerationProgress | None,
):
    prompt_router = require_prompt_router_from_config(_compose_config(config))
    return generate_sft_tasks(
        count,
        seed,
        effective_mix(config),
        compose_config=_compose_config(config),
        prompt_router=prompt_router,
        compose_start_index=compose_start_index,
        progress=progress,
    )


def _generate_teacher_rows(
    tasks,
    config: dict,
    executor,
    output: Path,
    *,
    completed_ids: set[str],
    target: int | None,
    label: str,
) -> dict[str, int]:
    remaining = target
    if target is not None:
        remaining = max(0, target - len(completed_ids)) if config.get("append", False) else target
    progress = GenerationProgress(label, remaining or len(tasks))
    if parallel_models_enabled(config):
        return generate_teacher_examples_parallel(
            tasks,
            config,
            executor,
            output_path=output,
            completed_ids=completed_ids,
            target=remaining,
            progress=progress,
        )

    teacher_router = router_from_config(config)
    accepted = rejected = skipped = 0
    for task in tasks:
        if task.id in completed_ids:
            skipped += 1
            continue
        if remaining is not None and accepted >= remaining:
            break
        error = None
        for _attempt in range(int(config.get("max_attempts", 2))):
            teacher_id = "unknown"
            try:
                response, teacher = generate_solution(
                    task,
                    teacher_router,
                    temperature=float(config.get("temperature", 0.7)),
                    max_tokens=int(config.get("max_tokens", 2048)),
                    repair_context=error,
                )
                teacher_id = teacher.model_id
                progress.start(teacher_id)
                report = validate_solution(
                    task,
                    response,
                    executor,
                    minimum_constraint_score=float(config.get("minimum_constraint_score", 0.75)),
                    minimum_task_consistency_score=minimum_task_consistency_score(config, task),
                )
            except Exception as exc:
                error = str(exc)
                progress.record(
                    teacher_id,
                    ok=False,
                    reason=classify_failure(exc=exc),
                )
                continue
            if report.valid:
                example = build_sft_example(task, response, teacher, report)
                append_jsonl(output, example.model_dump(mode="json"))
                progress.record(teacher.model_id, ok=True)
                accepted += 1
                break
            error = f"{report.error}\nPrevious answer:\n{response.model_dump_json()}"
            progress.record(
                teacher.model_id,
                ok=False,
                reason=classify_failure(report=report),
            )
        else:
            rejected += 1
    progress.close()
    return {
        "accepted": accepted,
        "rejected": rejected,
        "skipped": skipped,
        "attempted": accepted + rejected,
    }


def generate_prompt_bank(config: dict) -> dict[str, int]:
    """Write reviewable SFT task rows without teacher responses."""
    if batch_enabled(config):
        return batch_generate_prompts(config)
    prompts_path = _prompts_path(config)
    _refuse_overwrite(prompts_path, config)
    remaining, existing = resume_remaining(config, prompts_path)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    seed = int(config.get("seed", 42))
    compose_prefix = f"sft-compose-{seed}-"
    compose_start = next_index(existing_ids(existing), compose_prefix)
    progress = GenerationProgress("SFT prompts", remaining)
    tasks = _generate_sft_task_specs(
        config,
        count=remaining,
        seed=seed,
        compose_start_index=compose_start,
        progress=progress,
    )
    progress.close()
    rows = [task.model_dump(mode="json") for task in tasks]
    if config.get("append", False) and existing:
        for row in rows:
            append_jsonl(prompts_path, row)
        return {"written": len(rows), "skipped": len(existing)}
    return {"written": write_jsonl(prompts_path, rows), "skipped": 0}


def generate_samples(config: dict) -> dict[str, int]:
    """Batch or sync teacher generation into samples/output files."""
    if batch_enabled(config):
        return batch_generate_samples(config)
    return generate_responses(config)


def generate_responses(config: dict) -> dict[str, int]:
    """Legacy sync path: validate teacher responses while generating output."""
    prompts_path = _prompts_path(config)
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt bank not found: {prompts_path}")
    output = project_path(config["output"])
    _refuse_overwrite(output, config)
    executor = executor_from_config(config)
    completed_ids = _load_completed_ids(output) if config.get("append", False) else set()
    tasks = filter_tasks_by_type(
        [parse_task(row) for row in read_jsonl(prompts_path)],
        config,
    )
    target = int(config.get("count", len(tasks)))
    return _generate_teacher_rows(
        tasks,
        config,
        executor,
        output,
        completed_ids=completed_ids,
        target=target,
        label="SFT responses",
    )


def generate_dataset(config: dict) -> dict[str, int]:
    stage = _generation_stage(config)
    if stage == "prompts":
        return generate_prompt_bank(config)
    if stage == "samples":
        return generate_samples(config)
    if stage == "validate":
        if not batch_enabled(config):
            raise RuntimeError("stage: validate requires batch.enabled: true")
        return batch_validate_samples(config)
    if stage == "regenerate":
        if not batch_enabled(config):
            raise RuntimeError("stage: regenerate requires batch.enabled: true")
        return batch_regenerate_failed(config)
    if batch_enabled(config):
        return batch_generate_full(config)

    output = project_path(config["output"])
    _refuse_overwrite(output, config)
    executor = executor_from_config(config)
    target = int(config.get("count", 100))
    remaining, existing = resume_remaining(config, output)
    completed_ids = existing_ids(existing)
    if remaining <= 0:
        return {"accepted": 0, "skipped": len(existing), "rejected": 0, "attempted": 0}

    seed = int(config.get("seed", 42))
    compose_prefix = f"sft-compose-{seed}-"
    compose_start = next_index(completed_ids, compose_prefix)
    progress = GenerationProgress("SFT prompts", remaining)
    tasks = _generate_sft_task_specs(
        config,
        count=remaining,
        seed=seed,
        compose_start_index=compose_start,
        progress=progress,
    )
    progress.close()
    return _generate_teacher_rows(
        tasks,
        config,
        executor,
        output,
        completed_ids=completed_ids,
        target=target,
        label="SFT responses",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated TuneLM SFT data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    counts = generate_dataset(load_config(args.config))
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
