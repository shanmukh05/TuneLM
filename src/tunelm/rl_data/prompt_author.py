"""Author natural-language RL task prompts from procedural task specs.

Procedural generation owns constraints, code, and verifier metadata. This module
rewrites the musician-facing `prompt` field using a randomly chosen style and a
configured prompt-author model router.
"""

from __future__ import annotations

import json
import random
from typing import Any

from tunelm.data.generation_progress import GenerationProgress
from tunelm.data.vocab import load_vocab
from tunelm.providers.base import ProviderRouter
from tunelm.providers.provenance import provider_provenance
from tunelm.schemas import (
    ComposeTask,
    EditTask,
    ModelProvenance,
    RepairTask,
    RLTask,
    TaskType,
    TempoRange,
)
from tunelm.models.prompt_author_templates import task_style_system_prompt
from tunelm.strudel.parser import ResponseParseError, parse_json_object

PROMPT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}

DEFAULT_STYLES = {
    TaskType.COMPOSE: ("scene", "musician", "brief"),
    TaskType.EDIT: ("imperative", "conversational"),
    TaskType.REPAIR: ("direct", "collaborative", "diagnostic"),
}


def prompt_author_config(config: dict[str, Any]) -> dict[str, Any]:
    author = config.get("prompt_author")
    if author:
        return {**config, **author}
    return config


def available_styles(config: dict[str, Any], task_type: TaskType) -> tuple[str, ...]:
    author = prompt_author_config(config)
    key = {
        TaskType.COMPOSE: "compose_styles",
        TaskType.EDIT: "edit_styles",
        TaskType.REPAIR: "repair_styles",
    }[task_type]
    values = author.get(key, DEFAULT_STYLES[task_type])
    return tuple(str(item) for item in values)


def sample_style(rng: random.Random, config: dict[str, Any], task_type: TaskType) -> str:
    return rng.choice(available_styles(config, task_type))


def _compose_scene_hints(task: ComposeTask) -> list[str]:
    skip = {"procedural", "generated", "rich-compose", "prompt-llm"}
    return [
        tag for tag in task.tags if tag not in skip and not tag.startswith(("style-", "level-"))
    ]


def _friendly_instrument(name: str) -> str:
    """Musician word for a canonical sound id (`cp` -> `clap`) so prompts avoid Strudel ids."""
    for alias, target in load_vocab().instrument_aliases.items():
        if target == name and not alias.startswith("gm_"):
            return alias
    return name


def _brief_constraint_hints(task: ComposeTask) -> dict[str, Any]:
    """Explicit targets for the `brief` style; `stated_constraints` keeps what it mentions."""
    constraints = task.constraints
    hints: dict[str, Any] = {}
    if isinstance(constraints.tempo, TempoRange):
        hints["tempo_bpm"] = round((constraints.tempo.min + constraints.tempo.max) / 2)
    elif constraints.tempo is not None:
        hints["tempo_bpm"] = round(constraints.tempo)
    if constraints.required_instruments:
        hints["instruments"] = [_friendly_instrument(n) for n in constraints.required_instruments]
    if constraints.forbidden_instruments:
        hints["avoid_instruments"] = [
            _friendly_instrument(n) for n in constraints.forbidden_instruments
        ]
    if constraints.min_layers is not None:
        hints["min_layers"] = constraints.min_layers
    return hints


def task_author_payload(task: RLTask, style: str | None = None) -> dict[str, Any]:
    """Hints for prompt authoring; verifier targets are exposed only for the `brief` style."""
    payload: dict[str, Any] = {"task_type": task.task_type.value}
    if isinstance(task, ComposeTask):
        payload["difficulty"] = task.difficulty
        if task.constraints.mood:
            payload["mood"] = task.constraints.mood
        scene_hints = _compose_scene_hints(task)
        if scene_hints:
            payload["scene_hints"] = scene_hints[:5]
        if style == "brief":
            payload.update(_brief_constraint_hints(task))
        return payload
    if isinstance(task, EditTask):
        payload["instruction"] = task.instruction
        payload["preserve"] = task.preserve
        return payload
    if isinstance(task, RepairTask):
        payload["corruption"] = task.corruption
    return payload


def author_task_prompt(
    task: RLTask,
    style: str,
    router: ProviderRouter,
    *,
    temperature: float = 0.9,
    max_tokens: int = 256,
    max_attempts: int = 3,
) -> tuple[str, ModelProvenance]:
    system_prompt = task_style_system_prompt(task.task_type, style)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(task_author_payload(task, style), ensure_ascii=False),
        },
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            provider = router.choose()
            text = provider.generate(
                messages,
                response_schema=PROMPT_RESPONSE_SCHEMA,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not text.strip():
                raise ValueError("RL prompt author returned empty response")
            payload = parse_json_object(text)
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("RL prompt author returned empty prompt")
            return prompt, provider_provenance(provider, style=style)
        except (ResponseParseError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"RL prompt author failed after {max_attempts} attempts: {last_error}")


def apply_rl_prompt_styles(
    tasks: list[RLTask],
    config: dict[str, Any],
    *,
    prompt_router: ProviderRouter,
    seed: int = 42,
    progress: GenerationProgress | None = None,
) -> list[RLTask]:
    """Rewrite musician-facing prompts while keeping verifier metadata intact."""
    author = prompt_author_config(config)
    temperature = float(author.get("prompt_temperature", 0.9))
    max_tokens = int(author.get("prompt_max_tokens", 256))
    rng = random.Random(seed)
    styled: list[RLTask] = []
    for task in tasks:
        style = sample_style(rng, author, task.task_type)
        try:
            prompt, provenance = author_task_prompt(
                task,
                style,
                prompt_router,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if progress is not None:
                progress.start(provenance.model_id)
                progress.record(provenance.model_id, ok=True)
        except Exception:
            if progress is not None:
                progress.record("unknown", ok=False)
            raise
        tags = [tag for tag in task.tags if not tag.startswith(("style-", "prompt-"))]
        tags.extend([f"style-{style}", "prompt-llm"])
        styled.append(
            task.model_copy(update={"prompt": prompt, "tags": tags, "prompt_author": provenance})
        )
    if progress is not None:
        progress.close()
    return styled
