from __future__ import annotations

import json

from tunelm.inference import generate_messages, load_checkpoint, response_from_code_completion
from tunelm.io import read_jsonl, write_jsonl
from tunelm.models.templates import CODE_ONLY_SYSTEM_PROMPT, messages_for_task
from tunelm.schemas import parse_task


def generate_predictions(
    tasks_path,
    output_path,
    checkpoint: str,
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 1536,
    limit: int | None = None,
    mode: str = "combined",
    revision: str | None = None,
) -> int:
    model, tokenizer = load_checkpoint(checkpoint, revision=revision)

    def rows():
        for index, raw_task in enumerate(read_jsonl(tasks_path)):
            if limit is not None and index >= limit:
                break
            task = parse_task(raw_task)
            messages = messages_for_task(task)
            if mode == "code-first":
                messages[0] = {"role": "system", "content": CODE_ONLY_SYSTEM_PROMPT}
            completion = generate_messages(
                messages,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if mode == "code-first":
                try:
                    completion = json.dumps(
                        response_from_code_completion(completion).model_dump(), ensure_ascii=False
                    )
                except ValueError:
                    pass
            yield {"task_id": task.id, "completion": completion}

    return write_jsonl(output_path, rows())
