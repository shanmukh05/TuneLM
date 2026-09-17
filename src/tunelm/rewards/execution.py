from __future__ import annotations

from tunelm.strudel.executor import StrudelExecutor
from tunelm.strudel.parser import ResponseParseError, extract_completion_text, parse_model_response


def score_execution(completion, executor: StrudelExecutor) -> float:
    try:
        response = parse_model_response(extract_completion_text(completion))
    except ResponseParseError:
        return 0.0
    return float(executor.run(response.strudel_code).valid)


def make_execution_reward(executor: StrudelExecutor):
    def execution_reward(completions, **kwargs) -> list[float]:
        return [score_execution(completion, executor) for completion in completions]

    execution_reward.__name__ = "execution_reward"
    return execution_reward
