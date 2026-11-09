"""Render the shared model contract and strictly parse generated completions.

Composition summaries state five concise musical choices before executable code.
"""

from __future__ import annotations

import re

from tunelm.schemas import Task

SYSTEM = """You are TuneLM, an algorithmic Strudel composer.
Use the request and memory references to create executable Strudel JavaScript.
Return <think> with exactly five concise numbered musical decisions:
1. Tempo, key, and structure.
2. Rhythm and percussion.
3. Bass and harmony.
4. Melody, timbre, and modulation.
5. Mixing and arrangement.
Then close </think> and return one ```strudel code block with the entire program.
Use setcpm(BPM/4), immutable const declarations, and a final stack/arrange/pattern.
Adapt retrieved examples to the request. Preserve supplied code when requested.
For vague, mood, genre, or activity requests, choose fitting musical details yourself
and state those choices in the summary. The user need not specify code or theory.
Use Strudel JS, never Tidal Haskell operators. No imports, network, samples(),
browser APIs, loops, or external side effects. Use only registered sounds.
"""


def messages(task: Task) -> list[dict[str, str]]:
    content = task.prompt + "\n\n" + task.context
    if task.plan.original_code:
        content += "\n\nPreserve this existing voice:\n" + task.plan.original_code
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


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
        raise ValueError("Expected a five-part <think> summary and one Strudel code block")
    summary, code = match.groups()
    numbers = re.findall(r"^\s*([1-5])\.\s+\S", summary, re.M)
    if numbers != ["1", "2", "3", "4", "5"] or not code.strip():
        raise ValueError("Summary must contain exactly stages 1–5 in order and nonempty code")
    if "```" in code or "<think>" in summary or "</think>" in code:
        raise ValueError("Nested response delimiters are forbidden")
    return summary.strip(), code.strip()


def render_prompt(task: Task, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages(task), tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


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
