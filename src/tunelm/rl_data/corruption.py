"""Create deterministic valid reference programs and broken repair tasks.

Corruption handlers must change a reference into code rejected by the intended
verifier; repair targets always retain the original valid program.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

from tunelm.data.prompt_builder import build_repair_prompt
from tunelm.data.vocab import load_vocab
from tunelm.schemas import RepairTask


def _pick_param(rng: random.Random, values: list) -> object:
    return rng.choice(values)


def reference_program(index: int, seed: int = 42) -> str:
    """Create a deterministic, valid, and diverse verifier reference program."""
    vocab = load_vocab()
    skeletons = vocab.repair_templates.get("skeletons", [])
    rng = random.Random(seed + index)
    skeleton = skeletons[index % len(skeletons)]
    template = skeleton["template"]
    params = skeleton.get("params", {})
    slots: dict[str, object] = {"bpm": 60 + index % 91}
    for key, values in params.items():
        slots[key] = _pick_param(rng, values)
    if "gain" not in slots:
        slots["gain"] = 25 + index % 51
    return template.format(**slots)


def _corrupt_missing_closing_parenthesis(code: str) -> tuple[str, str]:
    if ")" not in code:
        raise ValueError("no closing parenthesis")
    return code.replace(")", "", 1), "missing_closing_parenthesis"


def _corrupt_invalid_function(code: str, rng: random.Random) -> tuple[str, str]:
    calls = list(re.finditer(r"\b(?:s|note|stack)\b", code))
    if not calls:
        raise ValueError("no call site")
    match = rng.choice(calls)
    broken = code[: match.start()] + "invalidFunction" + code[match.end() :]
    return broken, "invalid_function"


def _corrupt_unterminated_string(code: str, rng: random.Random) -> tuple[str, str]:
    strings = list(re.finditer(r'"[^"\n]+"', code))
    if not strings:
        raise ValueError("no string literal")
    match = rng.choice(strings)
    broken = code[: match.end() - 1] + code[match.end() :]
    return broken, "unterminated_string"


def _corrupt_missing_opening_parenthesis(code: str) -> tuple[str, str]:
    match = re.search(r"\b(?:stack|s|note)\s*\(", code)
    if not match:
        raise ValueError("no opening parenthesis")
    start = match.end() - 1
    return code[:start] + code[start + 1 :], "missing_opening_parenthesis"


def _corrupt_extra_closing_bracket(code: str) -> tuple[str, str]:
    return code + ")", "extra_closing_bracket"


def _corrupt_typo_in_sample(code: str, rng: random.Random) -> tuple[str, str]:
    samples = list(re.finditer(r's\("([^"]+)"\)', code))
    if not samples:
        raise ValueError("no sample call")
    match = rng.choice(samples)
    # A misspelled sample name is still valid Strudel because sample names are
    # open-ended. Misspell the call syntax instead so this is a genuine repair.
    opening_paren = match.start() + 1
    broken = code[:opening_paren] + "[" + code[opening_paren + 1 :]
    return broken, "typo_in_sample"


def _corrupt_broken_setcpm(code: str) -> tuple[str, str]:
    broken = re.sub(r"setcpm\(([^)]+)\)", "setcpm(broken", code, count=1)
    if broken == code:
        raise ValueError("no setcpm")
    return broken, "broken_setcpm"


def _corrupt_missing_comma(code: str) -> tuple[str, str]:
    match = re.search(r"\),\s*\n\s*(s\(|note\()", code)
    if not match:
        raise ValueError("no comma between layers")
    broken = code[: match.start()] + ")\n  " + code[match.end() - len(match.group(1)) :]
    return broken, "missing_comma"


def _corrupt_double_dot(code: str) -> tuple[str, str]:
    match = re.search(r"\.([a-zA-Z_]\w*)\(", code)
    if not match:
        raise ValueError("no method call")
    broken = code[: match.start()] + ".." + code[match.start() + 1 :]
    return broken, "double_dot"


def _corrupt_unclosed_stack(code: str) -> tuple[str, str]:
    if not code.rstrip().endswith(")"):
        raise ValueError("stack already unclosed")
    return code.rstrip()[:-1], "unclosed_stack"


def _corrupt_invalid_note_syntax(code: str) -> tuple[str, str]:
    match = re.search(r'note\("([^"]+)"\)', code)
    if not match:
        raise ValueError("no note call")
    broken = code.replace(match.group(0), 'note("c4 e4', 1)
    return broken, "invalid_note_syntax"


def _corrupt_wrong_gain_paren(code: str) -> tuple[str, str]:
    match = re.search(r"\.gain\([^)]+\)", code)
    if not match:
        raise ValueError("no gain call")
    broken = code.replace(match.group(0), ".gain(0.8", 1)
    return broken, "wrong_gain_paren"


def _corrupt_missing_method_dot(code: str) -> tuple[str, str]:
    match = re.search(r"\.(?:gain|slow)\(", code)
    if not match:
        raise ValueError("no gain or slow method")
    return code[: match.start()] + code[match.start() + 1 :], "missing_method_dot"


def _corrupt_misspelled_method(code: str) -> tuple[str, str]:
    match = re.search(r"\.(gain|slow)\(", code)
    if not match:
        raise ValueError("no gain or slow method")
    method = match.group(1)
    typo = "gian" if method == "gain" else "solw"
    return code[: match.start(1)] + typo + code[match.end(1) :], "misspelled_method"


def _corrupt_invalid_slow_argument(code: str) -> tuple[str, str]:
    match = re.search(r"\.slow\([^)]+\)", code)
    if not match:
        raise ValueError("no slow method")
    return code[: match.start()] + ".slow(broken)" + code[match.end() :], "invalid_slow_argument"


def _corrupt_missing_layer_expression(code: str) -> tuple[str, str]:
    match = re.search(r"(?m)^(\s+)(?:s|note)\(.+\),\s*$", code)
    if not match:
        raise ValueError("no removable stack layer")
    return code[: match.start()] + f"{match.group(1)}," + code[
        match.end() :
    ], "missing_layer_expression"


CORRUPTION_HANDLERS = {
    "missing_closing_parenthesis": lambda code, rng: _corrupt_missing_closing_parenthesis(code),
    "invalid_function": _corrupt_invalid_function,
    "unterminated_string": _corrupt_unterminated_string,
    "missing_opening_parenthesis": lambda code, rng: _corrupt_missing_opening_parenthesis(code),
    "extra_closing_bracket": lambda code, rng: _corrupt_extra_closing_bracket(code),
    "typo_in_sample": _corrupt_typo_in_sample,
    "broken_setcpm": lambda code, rng: _corrupt_broken_setcpm(code),
    "missing_comma": lambda code, rng: _corrupt_missing_comma(code),
    "double_dot": lambda code, rng: _corrupt_double_dot(code),
    "unclosed_stack": lambda code, rng: _corrupt_unclosed_stack(code),
    "invalid_note_syntax": lambda code, rng: _corrupt_invalid_note_syntax(code),
    "wrong_gain_paren": lambda code, rng: _corrupt_wrong_gain_paren(code),
    "missing_method_dot": lambda code, rng: _corrupt_missing_method_dot(code),
    "misspelled_method": lambda code, rng: _corrupt_misspelled_method(code),
    "invalid_slow_argument": lambda code, rng: _corrupt_invalid_slow_argument(code),
    "missing_layer_expression": lambda code, rng: _corrupt_missing_layer_expression(code),
}


def corrupt(code: str, rng: random.Random, preferred_kind: str | None = None) -> tuple[str, str]:
    kinds = [preferred_kind] if preferred_kind else list(CORRUPTION_HANDLERS)
    if preferred_kind and preferred_kind not in CORRUPTION_HANDLERS:
        raise ValueError(f"unknown corruption kind: {preferred_kind}")
    if not preferred_kind:
        kinds = list(CORRUPTION_HANDLERS)
    errors: list[str] = []
    for kind in kinds:
        handler = CORRUPTION_HANDLERS[kind]
        try:
            return handler(code, rng)
        except ValueError as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError(f"unable to corrupt program: {errors}")


def repair_tasks(count: int, seed: int = 42) -> Iterator[RepairTask]:
    rng = random.Random(seed)
    kinds = list(CORRUPTION_HANDLERS)
    for index in range(count):
        reference = reference_program(index + seed * 10_000, seed)
        broken = None
        corruption = None
        for offset in range(len(kinds)):
            kind = kinds[(index + offset) % len(kinds)]
            try:
                broken, corruption = corrupt(reference, rng, preferred_kind=kind)
                break
            except RuntimeError:
                continue
        if broken is None or corruption is None:
            broken, corruption = corrupt(reference, rng)
        yield RepairTask(
            id=f"rl-repair-{seed}-{index:07d}",
            prompt=build_repair_prompt(rng),
            broken_code=broken,
            reference_code=reference,
            corruption=corruption,
            difficulty=1 + index % 3,
            tags=["procedural", "repair", corruption],
        )
