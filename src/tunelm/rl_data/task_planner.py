"""Plan RL styled prompt batch requests from procedural task skeletons."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from tunelm.providers.gemini_batch import BatchRequest
from tunelm.rl_data.prompt_author import (
    PROMPT_RESPONSE_SCHEMA,
    prompt_author_config,
    sample_style,
    task_author_payload,
)
from tunelm.models.prompt_author_templates import task_style_system_prompt
from tunelm.schemas import ModelProvenance, RLTask
from tunelm.strudel.parser import parse_json_object


@dataclass(slots=True)
class StyledPromptSpec:
    task: RLTask
    style: str
    request: BatchRequest


def plan_rl_styled_prompts(
    tasks: list[RLTask],
    config: dict[str, Any],
    *,
    seed: int = 42,
) -> list[StyledPromptSpec]:
    """Build batch prompt-author requests for procedural RL task skeletons."""
    author = prompt_author_config(config)
    temperature = float(author.get("prompt_temperature", 0.9))
    max_tokens = int(author.get("prompt_max_tokens", 1024))
    rng = random.Random(seed)
    specs: list[StyledPromptSpec] = []
    for task in tasks:
        style = sample_style(rng, author, task.task_type)
        specs.append(
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
                    response_schema=PROMPT_RESPONSE_SCHEMA,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )
        )
    return specs


def tasks_from_styled_prompt_results(
    specs: list[StyledPromptSpec],
    results: dict[str, str],
    *,
    prompt_author: ModelProvenance,
) -> list[RLTask]:
    """Merge batch prompt-author text back into RL task rows."""
    tasks: list[RLTask] = []
    for spec in specs:
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
                    "prompt_author": prompt_author.model_copy(update={"style": spec.style}),
                }
            )
        )
    return tasks
