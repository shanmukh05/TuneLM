"""Plan SFT task specs and batch prompt-author requests without LLM calls."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from tunelm.schemas import ModelProvenance

from tunelm.data.mix import task_counts
from tunelm.rl_data.corruption import repair_tasks
from tunelm.rl_data.editing import editing_tasks
from tunelm.rl_data.prompt_author import sample_style, task_author_payload, task_style_system_prompt
from tunelm.schemas import ComposeTask, EditTask, RepairTask, RLTask, TaskType
from tunelm.sft_data.prompt_author import (
    ComposeBrief,
    brief_to_constraints,
    compose_level_system_prompt,
    sample_compose_brief,
    sample_compose_level,
)
from tunelm.providers.gemini_batch import BatchRequest


@dataclass(slots=True)
class ComposePromptSpec:
    task_id: str
    brief: ComposeBrief
    request: BatchRequest


@dataclass(slots=True)
class StyledPromptSpec:
    task: RLTask
    style: str
    request: BatchRequest


@dataclass(slots=True)
class SFTTaskPlan:
    compose_specs: list[ComposePromptSpec]
    styled_specs: list[StyledPromptSpec]


def _compose_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, **config.get("compose", {})}


def plan_sft_tasks(
    count: int,
    seed: int = 42,
    mix: dict[str, float] | None = None,
    *,
    compose_config: dict[str, Any] | None = None,
    compose_start_index: int = 0,
) -> SFTTaskPlan:
    """Sample procedural specs and build prompt-author batch requests."""
    compose_config = compose_config or {}
    mix = mix or {"compose": 0.7, "edit": 0.2, "repair": 0.1}
    compose_count, edit_count, repair_count = task_counts(count, mix)
    prompt_temperature = float(compose_config.get("prompt_temperature", 0.9))
    prompt_max_tokens = int(compose_config.get("prompt_max_tokens", 1024))
    rng = random.Random(seed)

    compose_specs: list[ComposePromptSpec] = []
    for offset in range(compose_count):
        index = compose_start_index + offset
        level = sample_compose_level(rng, compose_config)
        brief = sample_compose_brief(rng, level)
        task_id = f"sft-compose-{seed}-{index:06d}"
        compose_specs.append(
            ComposePromptSpec(
                task_id=task_id,
                brief=brief,
                request=BatchRequest(
                    key=task_id,
                    messages=[
                        {"role": "system", "content": compose_level_system_prompt(level)},
                        {
                            "role": "user",
                            "content": json.dumps(brief.to_author_payload(), ensure_ascii=False),
                        },
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                    temperature=prompt_temperature,
                    max_tokens=prompt_max_tokens,
                ),
            )
        )

    edit_repair: list[RLTask] = [
        *editing_tasks(edit_count, seed),
        *repair_tasks(repair_count, seed),
    ]
    styled_specs: list[StyledPromptSpec] = []
    for index, task in enumerate(edit_repair):
        task = task.model_copy(update={"id": task.id.replace("rl-", "sft-", 1)})
        style = sample_style(rng, compose_config, task.task_type)
        styled_specs.append(
            StyledPromptSpec(
                task=task,
                style=style,
                request=BatchRequest(
                    key=task.id,
                    messages=[
                        {"role": "system", "content": task_style_system_prompt(task.task_type, style)},
                        {
                            "role": "user",
                            "content": json.dumps(task_author_payload(task), ensure_ascii=False),
                        },
                    ],
                    response_schema={
                        "type": "object",
                        "properties": {"prompt": {"type": "string"}},
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                    temperature=prompt_temperature,
                    max_tokens=prompt_max_tokens,
                ),
            )
        )
    return SFTTaskPlan(compose_specs=compose_specs, styled_specs=styled_specs)


def tasks_from_prompt_results(
    plan: SFTTaskPlan,
    results: dict[str, str],
    *,
    prompt_author: ModelProvenance,
    seed: int = 42,
) -> list[RLTask]:
    """Merge batch prompt-author text back into RLTask rows."""
    from tunelm.strudel.parser import parse_json_object

    provenance = prompt_author
    tasks: list[RLTask] = []
    for spec in plan.compose_specs:
        text = results.get(spec.task_id)
        if not text:
            raise RuntimeError(f"missing batch prompt result for {spec.task_id}")
        prompt = str(parse_json_object(text).get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError(f"empty compose prompt for {spec.task_id}")
        brief = spec.brief
        tasks.append(
            ComposeTask(
                id=spec.task_id,
                prompt=prompt,
                constraints=brief_to_constraints(brief),
                difficulty=brief.level,
                tags=[f"level-{brief.level}", brief.mood, "prompt-llm", "generated"],
                prompt_author=provenance,
            )
        )
    for spec in plan.styled_specs:
        text = results.get(spec.task.id)
        if not text:
            raise RuntimeError(f"missing batch prompt result for {spec.task.id}")
        prompt = str(parse_json_object(text).get("prompt", "")).strip()
        if not prompt:
            raise RuntimeError(f"empty styled prompt for {spec.task.id}")
        tags = [tag for tag in spec.task.tags if not tag.startswith(("style-", "prompt-"))]
        tags.extend([f"style-{spec.style}", "prompt-llm"])
        tasks.append(
            spec.task.model_copy(
                update={
                    "prompt": prompt,
                    "tags": tags,
                    "prompt_author": provenance.model_copy(update={"style": spec.style}),
                }
            )
        )
    random.Random(seed).shuffle(tasks)
    return tasks
