from __future__ import annotations

from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


def score_format(completion) -> float:
    try:
        parsed = parse_model_response(extract_completion_text(completion))
    except ResponseParseError:
        return 0.0
    # Give partial credit only through other reward functions. This score means
    # the exact public contract was met.
    return 1.0 if parsed.strudel_code.strip() else 0.0


def format_reward(completions, **kwargs) -> list[float]:
    return [score_format(completion) for completion in completions]
