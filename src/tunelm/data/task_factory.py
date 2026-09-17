"""Build deterministic SFT and RL task collections from configured vocabularies."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from tunelm.data.diversity import load_prompt_hashes
from tunelm.data.mix import task_counts
from tunelm.rl_data.corruption import repair_tasks
from tunelm.rl_data.editing import editing_tasks
from tunelm.rl_data.procedural import composition_tasks
from tunelm.data.generation_progress import GenerationProgress
from tunelm.rl_data.prompt_author import apply_rl_prompt_styles
from tunelm.schemas import ComposeTask, RLTask
from tunelm.providers.base import ProviderRouter
from tunelm.sft_data.prompt_author import (
    author_compose_prompt,
    brief_to_constraints,
    sample_compose_brief,
    sample_compose_level,
)


def generate_sft_compose_tasks(
    count: int,
    seed: int = 42,
    *,
    start_index: int = 0,
    compose_config: dict[str, Any] | None = None,
    prompt_router: ProviderRouter | None = None,
    progress: GenerationProgress | None = None,
) -> Iterator[ComposeTask]:
    if prompt_router is None:
        raise RuntimeError("SFT compose tasks require a configured prompt-author router")
    compose_config = compose_config or {}
    prompt_temperature = float(compose_config.get("prompt_temperature", 0.9))
    prompt_max_tokens = int(compose_config.get("prompt_max_tokens", 256))
    rng = random.Random(seed)
    for offset in range(count):
        index = start_index + offset
        level = sample_compose_level(rng, compose_config)
        brief = sample_compose_brief(rng, level)
        try:
            prompt, provenance = author_compose_prompt(
                brief,
                prompt_router,
                temperature=prompt_temperature,
                max_tokens=prompt_max_tokens,
            )
            if progress is not None:
                progress.record(provenance.model_id, ok=True)
        except Exception:
            if progress is not None:
                progress.record("unknown", ok=False)
            raise
        yield ComposeTask(
            id=f"sft-compose-{seed}-{index:06d}",
            prompt=prompt,
            constraints=brief_to_constraints(brief),
            difficulty=level,
            tags=[f"level-{level}", brief.mood, "prompt-llm", "generated"],
            prompt_author=provenance,
        )


def _procedural_tasks(
    compose_count: int,
    edit_count: int,
    repair_count: int,
    seed: int,
    *,
    id_prefix: str,
) -> list[RLTask]:
    tasks: list[RLTask] = [
        *composition_tasks(compose_count, seed),
        *editing_tasks(edit_count, seed),
        *repair_tasks(repair_count, seed),
    ]
    if id_prefix != "rl":
        for index, task in enumerate(tasks):
            tasks[index] = task.model_copy(
                update={"id": task.id.replace("rl-", f"{id_prefix}-", 1)}
            )
    return tasks


def _split_llm_counts(total: int, llm_fraction: float) -> tuple[int, int]:
    llm_count = round(total * llm_fraction)
    return llm_count, total - llm_count


def task_signature(task: RLTask | dict[str, Any]) -> str:
    from tunelm.data.diversity import task_signature as _task_signature

    payload = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    return _task_signature(payload)


def _collect_unique_tasks(
    factory: Callable[[int, int], Iterator[RLTask]],
    target: int,
    seed: int,
    *,
    forbidden: set[str],
    max_rounds: int,
) -> list[RLTask]:
    accepted: list[RLTask] = []
    seen = set(forbidden)
    for offset in range(max_rounds):
        if len(accepted) >= target:
            break
        for task in factory(max(target * 5, 32), seed + offset * 9973):
            signature = task_signature(task)
            if signature in seen:
                continue
            seen.add(signature)
            accepted.append(task)
            if len(accepted) >= target:
                break
    return accepted


def _generate_disjoint_tasks(
    count: int,
    mix: dict[str, float],
    seed: int,
    *,
    id_prefix: str,
    forbidden: set[str],
) -> list[RLTask]:
    compose_count, edit_count, repair_count = task_counts(count, mix)
    tasks: list[RLTask] = []
    if compose_count:
        tasks.extend(
            _collect_unique_tasks(
                composition_tasks,
                compose_count,
                seed,
                forbidden=forbidden | {task_signature(task) for task in tasks},
                max_rounds=compose_count,
            )
        )
    if edit_count:
        tasks.extend(
            _collect_unique_tasks(
                editing_tasks,
                edit_count,
                seed,
                forbidden=forbidden | {task_signature(task) for task in tasks},
                max_rounds=edit_count,
            )
        )
    if repair_count:
        tasks.extend(
            _collect_unique_tasks(
                repair_tasks,
                repair_count,
                seed,
                forbidden=forbidden | {task_signature(task) for task in tasks},
                max_rounds=repair_count,
            )
        )
    return tasks


def generate_tasks(
    count: int,
    seed: int = 42,
    mix: dict[str, float] | None = None,
    *,
    mode: str = "procedural",
    llm_fraction: float = 0.0,
    llm_config: dict[str, Any] | None = None,
    id_prefix: str = "rl",
    disjoint_from: str | Path | None = None,
    min_unique_prompts: float | None = None,
) -> list[RLTask]:
    from tunelm.providers import llm_providers_available
    from tunelm.rl_data.llm_generator import llm_tasks

    mix = mix or {"compose": 0.6, "repair": 0.25, "edit": 0.15}
    forbidden = load_prompt_hashes(disjoint_from) if disjoint_from else set()
    llm_config = llm_config or {}
    use_llm = mode == "llm" or (
        mode == "hybrid" and llm_fraction > 0 and llm_providers_available(llm_config)
    )

    if forbidden:
        tasks = _generate_disjoint_tasks(
            count,
            mix,
            seed,
            id_prefix=id_prefix,
            forbidden=forbidden,
        )
    elif use_llm and mode == "llm":
        tasks = list(llm_tasks(count, seed, mix=mix, config=llm_config))
    elif use_llm and mode == "hybrid":
        llm_total, procedural_total = _split_llm_counts(count, llm_fraction)
        llm_compose, llm_edit, llm_repair = task_counts(llm_total, mix)
        proc_compose, proc_edit, proc_repair = task_counts(procedural_total, mix)
        tasks = [
            *llm_tasks(
                llm_total,
                seed,
                mix={
                    "compose": llm_compose / llm_total if llm_total else 0,
                    "edit": llm_edit / llm_total if llm_total else 0,
                    "repair": llm_repair / llm_total if llm_total else 0,
                },
                config=llm_config or {},
            ),
            *_procedural_tasks(
                proc_compose,
                proc_edit,
                proc_repair,
                seed + 1,
                id_prefix=id_prefix,
            ),
        ]
    else:
        compose_count, edit_count, repair_count = task_counts(count, mix)
        tasks = _procedural_tasks(
            compose_count,
            edit_count,
            repair_count,
            seed,
            id_prefix=id_prefix,
        )

    if min_unique_prompts is not None and tasks:
        from tunelm.data.diversity import benchmark_disjoint_ratio, compute_diversity_metrics

        payload = [task.model_dump(mode="json") for task in tasks]
        ratio = benchmark_disjoint_ratio(payload, forbidden)
        if ratio < float(min_unique_prompts):
            raise ValueError(
                f"Generated tasks are not disjoint enough: {ratio:.2%} < {min_unique_prompts:.2%}"
            )
        metrics = compute_diversity_metrics(payload)
        if metrics["by_task_type"].get("compose", {}).get("unique_normalized_prompts", 0) < 1:
            raise ValueError("Generated compose tasks lack prompt diversity")

    random.Random(seed).shuffle(tasks)
    return tasks[:count]


def generate_sft_tasks(
    count: int,
    seed: int = 42,
    mix: dict[str, float] | None = None,
    *,
    compose_config: dict[str, Any] | None = None,
    prompt_router: ProviderRouter | None = None,
    compose_start_index: int = 0,
    progress: GenerationProgress | None = None,
) -> list[RLTask]:
    if prompt_router is None:
        raise RuntimeError("SFT tasks require a configured prompt-author router")
    mix = mix or {"compose": 0.7, "edit": 0.2, "repair": 0.1}
    compose_config = compose_config or {}
    compose_count, edit_count, repair_count = task_counts(count, mix)
    compose_tasks = list(
        generate_sft_compose_tasks(
            compose_count,
            seed,
            start_index=compose_start_index,
            compose_config=compose_config,
            prompt_router=prompt_router,
            progress=progress,
        )
    )
    edit_repair: list[RLTask] = [
        *editing_tasks(edit_count, seed),
        *repair_tasks(repair_count, seed),
    ]
    for index, task in enumerate(edit_repair):
        edit_repair[index] = task.model_copy(update={"id": task.id.replace("rl-", "sft-", 1)})
    if edit_repair:
        edit_repair = apply_rl_prompt_styles(
            edit_repair,
            compose_config,
            prompt_router=prompt_router,
            seed=seed,
            progress=progress,
        )
    tasks = [*compose_tasks, *edit_repair]
    random.Random(seed).shuffle(tasks)
    return tasks
