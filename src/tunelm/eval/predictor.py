from __future__ import annotations

from tunelm.inference import generate_messages, load_checkpoint
from tunelm.io import read_jsonl, write_jsonl
from tunelm.models.templates import messages_for_task
from tunelm.schemas import parse_task


def generate_predictions(
    tasks_path,
    output_path,
    checkpoint: str,
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
    limit: int | None = None,
) -> int:
    model, tokenizer = load_checkpoint(checkpoint)

    def rows():
        for index, raw_task in enumerate(read_jsonl(tasks_path)):
            if limit is not None and index >= limit:
                break
            task = parse_task(raw_task)
            completion = generate_messages(
                messages_for_task(task),
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            yield {"task_id": task.id, "completion": completion}

    return write_jsonl(output_path, rows())
