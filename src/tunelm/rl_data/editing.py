from __future__ import annotations

from collections.abc import Iterator

from tunelm.data.vocab import load_vocab
from tunelm.rl_data.corruption import reference_program
from tunelm.schemas import EditTask, ExpectedChanges


def _expected_changes(payload: dict) -> ExpectedChanges:
    return ExpectedChanges(
        tempo=payload.get("tempo"),
        drum_density=payload.get("drum_density"),
        layer_count=payload.get("layer_count"),
        add_instruments=list(payload.get("add_instruments", [])),
        remove_instruments=list(payload.get("remove_instruments", [])),
    )


def editing_tasks(count: int, seed: int = 42) -> Iterator[EditTask]:
    vocab = load_vocab()
    operations = vocab.edit_operations
    for index in range(count):
        operation = operations[index % len(operations)]
        templates = operation.get("instruction_templates", [])
        instruction = (
            templates[(index // len(operations)) % len(templates)] if templates else "Edit"
        )
        original = reference_program(index + seed * 100_000)
        yield EditTask(
            id=f"rl-edit-{seed}-{index:07d}",
            prompt=instruction,
            original_code=original,
            instruction=instruction,
            expected_changes=_expected_changes(operation.get("expected_changes", {})),
            preserve=list(operation.get("preserve", [])),
            difficulty=3 + index % 3,
            tags=["procedural", "editing"],
        )
