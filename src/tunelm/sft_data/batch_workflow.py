"""Batch Gemini workflow for SFT data: prompts, samples, validate, regenerate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tunelm.config import project_path
from tunelm.data.batch_common import (
    batch_settings,
    refuse_overwrite,
    require_single_gemini_model,
    run_batch,
)
from tunelm.data.mix import configured_task_types, effective_mix, filter_tasks_by_type
from tunelm.data.resume import existing_ids, next_index, resume_remaining
from tunelm.io import append_jsonl, read_jsonl, write_jsonl
from tunelm.models.templates import messages_for_task
from tunelm.providers.gemini_batch import BatchRequest
from tunelm.schemas import ModelProvenance, ModelResponse, SFTExample, parse_task
from tunelm.sft_data.example_builder import build_sft_sample
from tunelm.sft_data.task_planner import plan_sft_tasks, tasks_from_prompt_results
from tunelm.sft_data.validator import minimum_task_consistency_score, validate_solution
from tunelm.strudel.executor import executor_from_config
from tunelm.strudel.parser import parse_model_response


def _compose_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, **config.get("compose", {})}


def _prompts_path(config: dict[str, Any]) -> Path:
    return project_path(config.get("prompts_file", "datasets/sft/prompts.jsonl"))


def _samples_path(config: dict[str, Any]) -> Path:
    return project_path(config.get("samples_file", "datasets/sft/samples.jsonl"))


def _output_path(config: dict[str, Any]) -> Path:
    return project_path(config["output"])


def _prompt_author_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = _compose_config(config)
    author = merged.get("prompt_author")
    if author:
        return {**merged, **author}
    return merged


def _prompt_author_model(config: dict[str, Any]) -> tuple[str, str, str]:
    return require_single_gemini_model(_prompt_author_config(config), label="SFT prompt authoring")


def _teacher_provenance(model_id: str, provider: str, model: str) -> ModelProvenance:
    return ModelProvenance(model_id=model_id, provider=provider, model=model)


def _task_from_row(row: dict[str, Any]):
    from tunelm.schemas import ComposeTask, EditTask, RepairTask

    payload = dict(row)
    payload.pop("response", None)
    payload.pop("metadata", None)
    task_models = {
        "compose": ComposeTask,
        "edit": EditTask,
        "repair": RepairTask,
    }
    task_model = task_models[str(payload["task_type"])]
    return task_model.model_validate(
        {key: payload[key] for key in task_model.model_fields if key in payload}
    )


def _sample_from_row(row: dict[str, Any]) -> SFTExample:
    return SFTExample.model_validate(row)


def _teacher_request(
    task,
    *,
    temperature: float,
    max_tokens: int,
    repair_context: str | None = None,
) -> BatchRequest:
    messages = messages_for_task(task)
    if repair_context:
        messages.append(
            {
                "role": "user",
                "content": "The previous answer was invalid. Fix it. Validation error:\n"
                + repair_context,
            }
        )
    return BatchRequest(
        key=task.id,
        messages=messages,
        response_schema=ModelResponse.model_json_schema(),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def batch_generate_prompts(config: dict[str, Any]) -> dict[str, int]:
    """Batch-author prompts and write datasets/sft/prompts.jsonl."""
    prompts_path = _prompts_path(config)
    refuse_overwrite(prompts_path, config)
    remaining, existing = resume_remaining(config, prompts_path)
    if remaining <= 0:
        return {"written": 0, "skipped": len(existing)}
    seed = int(config.get("seed", 42))
    compose_prefix = f"sft-compose-{seed}-"
    compose_start = next_index(existing_ids(existing), compose_prefix)
    plan = plan_sft_tasks(
        remaining,
        seed,
        effective_mix(config),
        compose_config=_compose_config(config),
        compose_start_index=compose_start,
    )
    requests = [spec.request for spec in plan.compose_specs] + [
        spec.request for spec in plan.styled_specs
    ]
    model_id, provider, model = _prompt_author_model(config)
    texts = run_batch(
        config,
        requests,
        display_name="tunelm-sft-prompts",
        model=model,
    )
    author = _teacher_provenance(model_id, provider, model)
    tasks = tasks_from_prompt_results(plan, texts, prompt_author=author, seed=seed)
    rows = [task.model_dump(mode="json") for task in tasks]
    if config.get("append", False) and existing:
        for row in rows:
            append_jsonl(prompts_path, row)
        return {"written": len(rows), "skipped": len(existing)}
    return {"written": write_jsonl(prompts_path, rows), "skipped": 0}


def batch_generate_samples(config: dict[str, Any]) -> dict[str, int]:
    """Batch-generate teacher responses into datasets/sft/samples.jsonl."""
    prompts_path = _prompts_path(config)
    if not prompts_path.is_file():
        raise FileNotFoundError(f"Prompt bank not found: {prompts_path}")
    samples_path = _samples_path(config)
    refuse_overwrite(samples_path, config)
    existing_rows = list(read_jsonl(samples_path)) if samples_path.exists() else []
    completed_ids = existing_ids(existing_rows)
    tasks = filter_tasks_by_type(
        [parse_task(row) for row in read_jsonl(prompts_path) if row["id"] not in completed_ids],
        config,
    )
    if not tasks:
        return {"written": 0, "skipped": len(completed_ids)}
    temperature = float(config.get("temperature", 0.7))
    max_tokens = int(config.get("max_tokens", 2048))
    model_id, provider, model = require_single_gemini_model(config, label="SFT teacher generation")
    requests = [
        _teacher_request(task, temperature=temperature, max_tokens=max_tokens) for task in tasks
    ]
    texts = run_batch(
        config,
        requests,
        display_name="tunelm-sft-samples",
        model=model,
    )
    teacher = _teacher_provenance(model_id, provider, model)
    written = 0
    for task in tasks:
        text = texts.get(task.id)
        if not text:
            continue
        response = parse_model_response(text)
        sample = build_sft_sample(task, response, teacher, valid=None, generation_attempts=1)
        append_jsonl(samples_path, sample.model_dump(mode="json"))
        written += 1
    return {"written": written, "skipped": len(completed_ids)}


def batch_validate_samples(config: dict[str, Any]) -> dict[str, int]:
    """Execute Strudel validation for every sample and export valid rows to output."""
    samples_path = _samples_path(config)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Sample bank not found: {samples_path}")
    output = _output_path(config)
    executor = executor_from_config(config)
    rows = list(read_jsonl(samples_path))
    valid_count = invalid_count = 0
    updated_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        sample = _sample_from_row(row)
        task = _task_from_row(row)
        report = validate_solution(
            task,
            sample.response,
            executor,
            minimum_constraint_score=float(config.get("minimum_constraint_score", 0.75)),
            minimum_task_consistency_score=minimum_task_consistency_score(config, task),
        )
        metadata = sample.metadata.model_copy(
            update={"valid": report.valid, "validation": report.as_dict()}
        )
        sample = sample.model_copy(update={"metadata": metadata})
        row = sample.model_dump(mode="json")
        updated_rows.append(row)
        if report.valid:
            valid_count += 1
            valid_rows.append(row)
        else:
            invalid_count += 1
    write_jsonl(samples_path, updated_rows)
    write_jsonl(output, valid_rows)
    return {
        "valid": valid_count,
        "invalid": invalid_count,
        "exported": len(valid_rows),
    }


def batch_regenerate_failed(config: dict[str, Any]) -> dict[str, int]:
    """Batch-regenerate teacher responses for samples that failed validation."""
    samples_path = _samples_path(config)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Sample bank not found: {samples_path}")
    max_attempts = int(config.get("max_attempts", 3))
    temperature = float(config.get("temperature", 0.7))
    max_tokens = int(config.get("max_tokens", 2048))
    rows = list(read_jsonl(samples_path))
    retry_rows: list[tuple[Any, str]] = []
    for row in rows:
        sample = _sample_from_row(row)
        if sample.metadata.valid is not False:
            continue
        if sample.metadata.generation_attempts >= max_attempts:
            continue
        task = _task_from_row(row)
        if configured_task_types(config) and task.task_type.value not in configured_task_types(config):
            continue
        error = sample.metadata.validation.get("error") or "validation failed"
        repair_context = f"{error}\nPrevious answer:\n{sample.response.model_dump_json()}"
        retry_rows.append((task, repair_context))

    if not retry_rows:
        return {"regenerated": 0, "skipped": len(rows)}

    model_id, provider, model = require_single_gemini_model(config, label="SFT teacher generation")
    requests = [
        _teacher_request(
            task,
            temperature=temperature,
            max_tokens=max_tokens,
            repair_context=repair_context,
        )
        for task, repair_context in retry_rows
    ]
    texts = run_batch(
        config,
        requests,
        display_name="tunelm-sft-regenerate",
        model=model,
    )
    teacher = _teacher_provenance(model_id, provider, model)
    regenerated_map: dict[str, dict[str, Any]] = {}
    regenerated = 0
    for task, repair_context in retry_rows:
        text = texts.get(task.id)
        if not text:
            continue
        prior = _sample_from_row(next(row for row in rows if row["id"] == task.id))
        response = parse_model_response(text)
        sample = build_sft_sample(
            task,
            response,
            teacher,
            valid=None,
            generation_attempts=prior.metadata.generation_attempts + 1,
        )
        regenerated_map[task.id] = sample.model_dump(mode="json")
        regenerated += 1
    write_jsonl(
        samples_path,
        [regenerated_map.get(row["id"], row) for row in rows],
    )
    counts = {"regenerated": regenerated, "skipped": len(rows) - regenerated}
    if batch_settings(config)["regenerate_validate"] and regenerated:
        counts.update(batch_validate_samples(config))
    return counts


def batch_generate_full(config: dict[str, Any]) -> dict[str, int]:
    """Run prompts → samples → validate, then regenerate failed rows until done."""
    counts: dict[str, int] = {}
    counts.update(batch_generate_prompts(config))
    counts.update(batch_generate_samples(config))
    counts.update(batch_validate_samples(config))
    max_attempts = int(config.get("max_attempts", 3))
    for _round in range(max_attempts - 1):
        rows = list(read_jsonl(_samples_path(config)))
        failed = [
            row
            for row in rows
            if _sample_from_row(row).metadata.valid is False
            and _sample_from_row(row).metadata.generation_attempts < max_attempts
        ]
        if not failed:
            break
        round_counts = batch_regenerate_failed(config)
        for key, value in round_counts.items():
            counts[key] = counts.get(key, 0) + value
    return counts
