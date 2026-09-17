from __future__ import annotations

from tunelm.models.templates import messages_for_task
from tunelm.providers.base import ProviderRouter
from tunelm.providers.provenance import provider_provenance
from tunelm.schemas import ModelProvenance, ModelResponse, RLTask


def generate_solution(
    task: RLTask,
    router: ProviderRouter,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    repair_context: str | None = None,
) -> tuple[ModelResponse, ModelProvenance]:
    messages = messages_for_task(task)
    if repair_context:
        messages.append(
            {
                "role": "user",
                "content": "The previous answer was invalid. Fix it. Validation error:\n"
                + repair_context,
            }
        )
    provider = router.choose()
    text = provider.generate(
        messages,
        response_schema=ModelResponse.model_json_schema(),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    from tunelm.strudel.parser import parse_model_response

    return parse_model_response(text), provider_provenance(provider)
