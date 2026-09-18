"""Generate verifier-friendly RL tasks through configured LLM providers.

Provider, JSON, and task-validation failures are fatal so hybrid datasets cannot
silently change their requested procedural/LLM composition.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from tunelm.data.mix import task_counts
from tunelm.data.vocab import load_vocab
from tunelm.providers import router_from_config
from tunelm.schemas import (
    ComposeTask,
    EditTask,
    RepairTask,
    RLTask,
)
from tunelm.strudel.executor import StrudelExecutor


RL_LLM_TASK_SYSTEM_PROMPT = "You create verifier-friendly TuneLM training tasks as JSON."

TASK_SCHEMA_HINT = """Return one JSON object for a TuneLM task with fields:
- task_type: compose | edit | repair
- prompt: varied natural-language request from the musician
- constraints: {tempo, required_instruments, forbidden_instruments, mood, min_layers, max_layers}
- for edit: original_code, instruction, expected_changes, preserve
- for repair: broken_code, reference_code, corruption
"""


def _validate_compose(task: ComposeTask, executor: StrudelExecutor) -> bool:
    constraints = task.constraints
    if not constraints.required_instruments and constraints.tempo is None:
        return False
    return True


def _validate_edit(task: EditTask, executor: StrudelExecutor) -> bool:
    result = executor.run(task.original_code)
    return result.valid


def _validate_repair(task: RepairTask, executor: StrudelExecutor) -> bool:
    if not task.reference_code:
        return False
    return executor.run(task.reference_code).valid


def _parse_llm_task(payload: dict[str, Any], task_type: str, seed: int, index: int) -> RLTask:
    payload = dict(payload)
    payload["task_type"] = task_type
    payload["id"] = payload.get("id") or f"llm-{task_type}-{seed}-{index:07d}"
    if task_type == "compose":
        return ComposeTask.model_validate(payload)
    if task_type == "edit":
        return EditTask.model_validate(payload)
    return RepairTask.model_validate(payload)


def _generate_one(
    task_type: str,
    router,
    executor: StrudelExecutor,
    seed: int,
    index: int,
    *,
    temperature: float,
    max_tokens: int,
) -> RLTask:
    vocab = load_vocab()
    sound_ids = ", ".join(f"{name}={sound}" for name, sound in vocab.instrument_sounds.items())
    prompt = (
        f"{TASK_SCHEMA_HINT}\n"
        f"Generate one {task_type} task using varied musical vocabulary.\n"
        f"Canonical instruments: {', '.join(vocab.instruments)}\n"
        f"Strudel sound identifiers: {sound_ids}\n"
        f"Available moods: {', '.join(vocab.moods)}\n"
        f"Available genres: {', '.join(vocab.genres)}"
    )
    messages = [
        {"role": "system", "content": RL_LLM_TASK_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    provider = router.choose()
    text = provider.generate(messages, temperature=temperature, max_tokens=max_tokens)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider returned invalid JSON for {task_type} task {index}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"provider returned non-object JSON for {task_type} task {index}")
    task = _parse_llm_task(payload, task_type, seed, index)

    validators = {
        "compose": _validate_compose,
        "edit": _validate_edit,
        "repair": _validate_repair,
    }
    if not validators[task_type](task, executor):
        raise ValueError(f"provider generated invalid {task_type} task {task.id}")
    return task


def llm_tasks(
    count: int,
    seed: int = 42,
    *,
    mix: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> Iterator[RLTask]:
    config = config or {}
    compose_count, edit_count, repair_count = task_counts(
        count, mix or {"compose": 0.6, "edit": 0.2, "repair": 0.2}
    )
    router = router_from_config(config)
    executor = StrudelExecutor(backend=config.get("strudel", {}).get("backend", "static"))
    temperature = float(config.get("temperature", 0.9))
    max_tokens = int(config.get("max_tokens", 2048))
    index = 0
    for task_type, task_count in (
        ("compose", compose_count),
        ("edit", edit_count),
        ("repair", repair_count),
    ):
        for _ in range(task_count):
            yield _generate_one(
                task_type,
                router,
                executor,
                seed,
                index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            index += 1
