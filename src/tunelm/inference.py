"""Load TuneLM checkpoints and generate validated Strudel responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from tunelm.inference_limits import DEFAULT_MAX_NEW_TOKENS
from tunelm.models.templates import CODE_ONLY_SYSTEM_PROMPT, SYSTEM_PROMPT, render_generation_prompt
from tunelm.schemas import ModelResponse
from tunelm.strudel.controls import infer_controls_from_code
from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import ResponseParseError, parse_model_response


def response_from_code_completion(completion: str) -> ModelResponse:
    try:
        return parse_model_response(completion)
    except ResponseParseError:
        pass
    code = completion.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            code = "\n".join(lines[1:-1]).strip()
    controls = infer_controls_from_code(code)
    instruments = sorted({sound for layer in controls.layers for sound in layer.sounds})
    return ModelResponse.model_validate(
        {
            "plan": {
                "tempo": controls.bpm,
                "instruments": instruments or ["pattern"],
                "structure": "Generated Strudel arrangement",
            },
            "controls": controls.model_dump(),
            "strudel_code": code,
        }
    )


def load_checkpoint(checkpoint: str, *, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if (Path(checkpoint) / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, device_map="auto", dtype="auto", trust_remote_code=False
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, device_map="auto", dtype="auto", trust_remote_code=False
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
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
) -> str:
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
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.7,
    validate: bool = True,
    compile_retries: int = 2,
    mode: str = "combined",
) -> dict[str, Any]:
    model, tokenizer = load_checkpoint(checkpoint)
    if mode not in {"combined", "code-first"}:
        raise ValueError(f"Unknown generation mode: {mode}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT if mode == "combined" else CODE_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    executor = StrudelExecutor(backend="node", cycles=8, max_events=8192)
    last_error = "no completion"
    for attempt in range(compile_retries + 1):
        completion = generate_messages(
            messages,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature if attempt == 0 else min(temperature, 0.2),
        )
        try:
            response = (
                parse_model_response(completion)
                if mode == "combined"
                else response_from_code_completion(completion)
            )
        except (ResponseParseError, ValueError) as exc:
            last_error = str(exc)
            response = None
        else:
            execution = executor.run(response.strudel_code) if validate else None
            if execution is None or execution.playable:
                output = response.model_dump()
                if execution is not None:
                    output["validation"] = execution.model_dump()
                output["generation_attempts"] = attempt + 1
                return output
            last_error = execution.error or "program produced no audible events"
        messages = [
            messages[0],
            messages[1],
            {"role": "assistant", "content": completion},
            {
                "role": "user",
                "content": (
                    f"The previous response failed Strudel validation: {last_error[:600]}. "
                    "Return the full corrected response in the requested format. Preserve the original request."
                ),
            },
        ]
    raise RuntimeError(f"Strudel generation failed after {compile_retries + 1} attempts: {last_error}")


def dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)
