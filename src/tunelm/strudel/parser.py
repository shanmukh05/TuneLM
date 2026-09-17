from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from tunelm.schemas import ModelResponse


class ResponseParseError(ValueError):
    pass


def _candidate_json(text: str) -> str:
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.I)
    if fence:
        stripped = fence.group(1)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return stripped[index : index + end]
    raise ResponseParseError("No JSON object found in model response")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response, tolerating fences and preamble."""
    data = json.loads(_candidate_json(text))
    if not isinstance(data, dict):
        raise ResponseParseError("Expected JSON object in model response")
    return data


def _normalize_model_response_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Accept occasional flat plan fields when models omit the nested plan object."""
    if "strudel_code" in data or "plan" in data:
        return data
    plan_keys = ("tempo", "instruments", "structure", "musical_intent")
    if not any(key in data for key in plan_keys):
        return data
    payload = dict(data)
    payload["plan"] = {key: payload.pop(key) for key in plan_keys if key in payload}
    return payload


def parse_model_response(value: str | dict[str, Any]) -> ModelResponse:
    try:
        data = json.loads(_candidate_json(value)) if isinstance(value, str) else value
        if isinstance(data, dict):
            data = _normalize_model_response_payload(data)
        return ModelResponse.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ResponseParseError(f"Invalid TuneLM response: {exc}") from exc


def extract_completion_text(completion: Any) -> str:
    """Normalize text and chat-shaped completions emitted by TRL."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[-1]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    return str(completion)
