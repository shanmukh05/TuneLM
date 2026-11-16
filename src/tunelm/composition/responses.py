"""Render the shared model contract and strictly parse generated completions.

The think block is musical reasoning in prose, followed by one executable program.
"""

from __future__ import annotations

import re

from tunelm.core.schemas import Task

STRUDEL_RULES = """\
Use setcpm(BPM/4), immutable const declarations, and a final stack/arrange/pattern.
Compose original parts. Preserve code only when the user message supplies it.
When tempo, key, or orchestration are left open, choose them yourself and state
those choices in the thinking. The listener need not specify code or theory.
Use Strudel JS, never Tidal Haskell operators. No imports, network, samples(),
browser APIs, loops, or external side effects. Use only registered sounds.
"""

SYSTEM = (
    """You are TuneLM, an algorithmic Strudel composer.
The user message is the listener's request and nothing else. Choose the musical
details they leave open.
Return <think> with your musical reasoning in ordinary prose. Think out loud
about tempo, key, rhythm, harmony, melody, and the mix as you settle them.
Write continuous reasoning, not a numbered checklist.
Then close </think> and return one ```strudel code block with the entire program.
"""
    + STRUDEL_RULES
)


def messages(task: Task) -> list[dict[str, str]]:
    """LFM receives the listener request. Teacher reference context stays off this prompt."""
    content = task.prompt
    if task.plan.original_code:
        content += "\n\nPreserve this existing voice:\n" + task.plan.original_code
    return listener_messages(content)


def listener_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def completion(summary: str, code: str) -> str:
    value = f"<think>\n{summary.strip()}\n</think>\n```strudel\n{code.strip()}\n```"
    parse_completion(value)
    return value


def restore_thinking(text: str) -> str:
    """Generation continues after the checkpoint template's opening think tag."""
    return text if text.lstrip().startswith("<think>") else "<think>" + text


def parse_completion(text: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*```(?:strudel|javascript|js)\s*\n(.*?)\n```\s*", text, re.S
    )
    if not match:
        raise ValueError("Expected a <think> block and one Strudel code block")
    summary, code = (part.strip() for part in match.groups())
    if not summary or not code:
        raise ValueError("The think block and the Strudel code must both be nonempty")
    if "```" in code or "<think>" in summary or "</think>" in code:
        raise ValueError("Nested response delimiters are forbidden")
    return summary.strip(), code.strip()


def render_prompt(task: Task, tokenizer) -> str:
    return render_chat(messages(task), tokenizer)


def render_chat(chat: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


def strudel_excerpt(code: str, limit: int) -> str:
    if len(code) <= limit:
        return code
    end = limit // 3
    beginning = code[: limit - end].rpartition("\n")[0] or code[: limit - end]
    ending = code[-end:].partition("\n")[2] or code[-end:]
    return f"{beginning}\n// … middle of previous composition omitted …\n{ending}"


def render_conversation(prompt: str, history: list[dict[str, str]], tokenizer, budget: int) -> str:
    """Keep the newest complete turns within the checkpoint's input token budget."""
    system, user = listener_messages(prompt)
    if history:
        system["content"] += (
            "\nFor a revision, use the most recent composition unless the listener refers to an "
            "earlier one. Keep musical details they did not ask to change."
        )
    rendered = render_chat([system, user], tokenizer)
    accepted: list[dict[str, str]] = []
    for turn in reversed(history):
        previous = {"role": "user", "content": turn["prompt"]}
        code = turn["code"]
        # The latest score matters most for requests such as “keep the drums.”
        versions = [code]
        if not accepted:
            versions.extend(
                strudel_excerpt(code, limit)
                for limit in (4500, 2200, 1000, 350)
                if len(code) > limit
            )
        included = False
        for version in versions:
            assistant = {"role": "assistant", "content": f"```strudel\n{version}\n```"}
            candidate = render_chat([system, previous, assistant, *accepted, user], tokenizer)
            if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
                accepted[:0] = [previous, assistant]
                rendered = candidate
                included = True
                break
        if not included:
            break
    return rendered


def generate_completion(
    prompt: str, model, tokenizer, *, max_new_tokens: int, temperature: float, max_length: int
) -> tuple[str, dict[str, int]]:
    """Generate the assistant continuation and count tokens before decoding."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    size = inputs["input_ids"].shape[1]
    if size + max_new_tokens > max_length:
        raise ValueError(
            "Prompt plus completion exceeds configured context; reduce input or output"
        )
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.inference_mode():
        tokens = model.generate(**inputs, **kwargs)
    output = tokens[0, size:]
    text = tokenizer.decode(output, skip_special_tokens=False)
    if tokenizer.eos_token is not None:
        text = text.split(tokenizer.eos_token, 1)[0]
    usage = {
        "prompt_tokens": size,
        "completion_tokens": len(output),
        "total_tokens": tokens.shape[1],
    }
    return restore_thinking(text.rstrip()), usage


def tokenize_example(example, tokenizer, max_length: int) -> dict:
    """Mask exact prompt tokens and reject overlength or template-mismatched rows."""
    prompt = render_prompt(example, tokenizer)
    full = tokenizer.apply_chat_template(
        messages(example)
        + [{"role": "assistant", "content": completion(example.summary, example.code)}],
        tokenize=False,
        preserve_thinking=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = tokenizer.encode(full, add_special_tokens=False)
    if token_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Template prompt is not an exact token prefix: {example.id}")
    if len(token_ids) > max_length or len(token_ids) <= len(prompt_ids):
        raise ValueError(f"Example does not fit max_length or has no completion: {example.id}")
    labels = [-100] * len(prompt_ids) + token_ids[len(prompt_ids) :]
    return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids), "labels": labels}
