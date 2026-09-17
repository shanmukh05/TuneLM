"""Shard SFT teacher work across multiple configured models in parallel."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tunelm.data.generation_progress import GenerationProgress
from tunelm.io import append_jsonl
from tunelm.providers.registry import single_model_routers
from tunelm.schemas import RLTask
from tunelm.sft_data.example_builder import build_sft_example
from tunelm.sft_data.solution_generator import generate_solution
from tunelm.sft_data.validator import (
    classify_failure,
    minimum_task_consistency_score,
    validate_solution,
)
from tunelm.strudel.executor import StrudelExecutor


def parallel_models_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("parallel_models", False))


def _shard_tasks(tasks: list[RLTask], shard_count: int) -> list[list[RLTask]]:
    shards: list[list[RLTask]] = [[] for _ in range(shard_count)]
    for index, task in enumerate(tasks):
        shards[index % shard_count].append(task)
    return shards


def _process_task(
    task: RLTask,
    router,
    executor: StrudelExecutor,
    config: dict[str, Any],
    *,
    model_id: str,
    progress: GenerationProgress | None = None,
) -> tuple[Any | None, bool]:
    error = None
    for _attempt in range(int(config.get("max_attempts", 2))):
        if progress is not None:
            progress.start(model_id)
        try:
            response, teacher = generate_solution(
                task,
                router,
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
                repair_context=error,
            )
            report = validate_solution(
                task,
                response,
                executor,
                minimum_constraint_score=float(config.get("minimum_constraint_score", 0.75)),
                minimum_task_consistency_score=minimum_task_consistency_score(config, task),
            )
        except Exception as exc:
            error = str(exc)
            if progress is not None:
                progress.record(
                    model_id,
                    ok=False,
                    reason=classify_failure(exc=exc),
                )
            continue
        if report.valid:
            if progress is not None:
                progress.record(teacher.model_id, ok=True)
            return build_sft_example(task, response, teacher, report), True
        error = f"{report.error}\nPrevious answer:\n{response.model_dump_json()}"
        if progress is not None:
            progress.record(
                model_id,
                ok=False,
                reason=classify_failure(report=report),
            )
    return None, False


def generate_teacher_examples_parallel(
    tasks: list[RLTask],
    config: dict[str, Any],
    executor: StrudelExecutor,
    *,
    output_path,
    completed_ids: set[str],
    target: int | None = None,
    progress: GenerationProgress | None = None,
) -> dict[str, int]:
    """Generate validated SFT rows using one thread per enabled model."""
    pending = [task for task in tasks if task.id not in completed_ids]
    routers = single_model_routers(config)
    shards = _shard_tasks(pending, len(routers))
    lock = threading.Lock()
    stats = {"accepted": 0, "rejected": 0, "skipped": len(completed_ids), "attempted": 0}
    stop = threading.Event()

    def worker(model_id: str, router, shard: list[RLTask]) -> None:
        for task in shard:
            if stop.is_set():
                return
            example, accepted = _process_task(
                task,
                router,
                executor,
                config,
                model_id=model_id,
                progress=progress,
            )
            with lock:
                stats["attempted"] += 1
                if target is not None and stats["accepted"] >= target:
                    stop.set()
                    return
                if not accepted or example is None:
                    stats["rejected"] += 1
                    continue
                append_jsonl(output_path, example.model_dump(mode="json"))
                stats["accepted"] += 1
                if target is not None and stats["accepted"] >= target:
                    stop.set()

    max_workers = int(config.get("parallel_workers", len(routers)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(worker, model_id, router, shard)
            for (model_id, router), shard in zip(routers, shards, strict=False)
        ]
        for future in as_completed(futures):
            future.result()
    if progress is not None:
        progress.close()
    return stats
