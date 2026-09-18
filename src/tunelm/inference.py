"""Load TuneLM checkpoints and generate validated Strudel responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tunelm.models.templates import SYSTEM_PROMPT, render_generation_prompt
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import parse_model_response


def load_checkpoint(checkpoint: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (Path(checkpoint) / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", dtype="auto", trust_remote_code=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", dtype="auto", trust_remote_code=False
        )
    model.eval()
    return model, tokenizer


def _input_device(model):
    """Return the embedding device, including for Accelerate-dispatched models."""
    return model.get_input_embeddings().weight.device


def generate_messages(
    messages: list[dict[str, str]],
    model,
    tokenizer,
    *,
    max_new_tokens: int = 1536,
    temperature: float = 0.7,
) -> str:
    import torch

    text = render_generation_prompt(messages, tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(_input_device(model))
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
