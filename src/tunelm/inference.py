"""Load TuneLM checkpoints and generate validated Strudel responses.

Message rendering uses a minimal local format only when the checkpoint tokenizer
cannot supply a chat template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tunelm.models.templates import SYSTEM_PROMPT
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import parse_model_response


def _render_messages(messages: list[dict[str, str]], tokenizer) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except (TypeError, ValueError):
            # Transformers exposes apply_chat_template even when a tokenizer has
            # no configured template. Fall back to TuneLM's portable rendering.
            pass
    return "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages) + "\n<assistant>\n"


def load_checkpoint(checkpoint: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=False)
    if (Path(checkpoint) / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", torch_dtype="auto", trust_remote_code=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", torch_dtype="auto", trust_remote_code=False
        )
    return model, tokenizer


def generate_messages(
    messages: list[dict[str, str]],
    model,
    tokenizer,
    *,
    max_new_tokens: int = 1536,
    temperature: float = 0.7,
) -> str:
    import torch

    text = _render_messages(messages, tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation["temperature"] = temperature
    with torch.inference_mode():
        ids = model.generate(**inputs, **generation)
    return tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def generate(
    prompt: str,
    checkpoint: str,
    *,
    max_new_tokens: int = 1536,
    temperature: float = 0.7,
    validate: bool = True,
) -> dict[str, Any]:
    model, tokenizer = load_checkpoint(checkpoint)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    completion = generate_messages(
        messages,
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    response = parse_model_response(completion)
    output = response.model_dump()
    if validate:
        output["validation"] = StrudelExecutor().run(response.strudel_code).model_dump()
    return output


def dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
