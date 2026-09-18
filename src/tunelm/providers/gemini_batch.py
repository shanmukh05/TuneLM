"""Gemini Batch API helpers for offline TuneLM data generation."""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tunelm.providers.base import Message

_COMPLETED_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


@dataclass(slots=True)
class BatchRequest:
    """One keyed generateContent call for a Gemini batch job."""

    key: str
    messages: Sequence[Message]
    response_schema: dict[str, Any] | None = None
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass(slots=True)
class BatchResult:
    """Parsed text (or error) for one batch request key."""

    key: str
    text: str | None
    error: str | None = None


def _generation_config(
    *,
    response_schema: dict[str, Any] | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if response_schema:
        config["response_mime_type"] = "application/json"
        # Batch inline + file paths reject response_schema fields like
        # additionalProperties; response_json_schema accepts full JSON Schema.
        config["response_json_schema"] = response_schema
    return config


def _contents_from_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in messages
        if message["role"] != "system"
    ]


def _system_instruction(messages: Sequence[Message]) -> dict[str, Any] | None:
    system = "\n".join(message["content"] for message in messages if message["role"] == "system")
    if not system:
        return None
    return {"parts": [{"text": system}]}


def to_inline_request(request: BatchRequest) -> dict[str, Any]:
    """Build one inline batch item understood by google.genai Client.batches.create."""
    config = _generation_config(
        response_schema=request.response_schema,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    system = _system_instruction(request.messages)
    if system:
        config["system_instruction"] = system
    return {
        "contents": _contents_from_messages(request.messages),
        "config": config,
        "metadata": {"key": request.key},
    }


def to_file_request_line(request: BatchRequest) -> dict[str, Any]:
    """Build one JSONL line for file-based Gemini batch jobs."""
    generation_config = _generation_config(
        response_schema=request.response_schema,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    body: dict[str, Any] = {
        "contents": _contents_from_messages(request.messages),
        "generation_config": generation_config,
    }
    system = _system_instruction(request.messages)
    if system:
        body["system_instruction"] = system
    return {"key": request.key, "request": body}


def _response_text(payload: Any) -> str | None:
    if payload is None:
        return None
    text = getattr(payload, "text", None)
    if text:
        return text
    if isinstance(payload, Mapping):
        candidates = payload.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts") or []
            texts = [part.get("text", "") for part in parts if part.get("text")]
            joined = "".join(texts).strip()
            return joined or None
    return None


def _parse_result_line(line: str) -> BatchResult | None:
    if not line.strip():
        return None
    payload = json.loads(line)
    key = payload.get("key") or (payload.get("metadata") or {}).get("key")
    if not key:
        return None
    if payload.get("error"):
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        return BatchResult(key=str(key), text=None, error=message or "batch request failed")
    response = payload.get("response")
    text = _response_text(response)
    if text:
        return BatchResult(key=str(key), text=text)
    return BatchResult(key=str(key), text=None, error="empty batch response")


class GeminiBatchClient:
    """Submit, poll, and collect Gemini batch generateContent jobs."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install provider extras: pip install -e '.[providers]'") from exc
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def run(
        self,
        requests: Sequence[BatchRequest],
        *,
        display_name: str,
        poll_seconds: int = 30,
        inline_max_requests: int = 0,
    ) -> dict[str, BatchResult]:
        if not requests:
            return {}
        if inline_max_requests and len(requests) <= inline_max_requests:
            job = self.client.batches.create(
                model=self.model,
                src=[to_inline_request(request) for request in requests],
                config={"display_name": display_name},
            )
        else:
            job = self._create_file_job(requests, display_name=display_name)
        job = self._poll(job.name, poll_seconds=poll_seconds)
        if job.state and job.state.name != "JOB_STATE_SUCCEEDED":
            error = getattr(job, "error", None)
            message = getattr(error, "message", None) if error else None
            raise RuntimeError(
                f"Gemini batch job {job.name} finished with {job.state.name}: {message or 'unknown error'}"
            )
        return self._collect(job)

    def _create_file_job(self, requests: Sequence[BatchRequest], *, display_name: str):
        from google.genai import types

        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as handle:
            path = Path(handle.name)
            for request in requests:
                handle.write(json.dumps(to_file_request_line(request), ensure_ascii=False) + "\n")
        uploaded = self.client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        path.unlink(missing_ok=True)
        return self.client.batches.create(
            model=self.model,
            src=uploaded.name,
            config={"display_name": display_name},
        )

    def _poll(self, job_name: str, *, poll_seconds: int):
        job = self.client.batches.get(name=job_name)
        while job.state and job.state.name not in _COMPLETED_STATES:
            time.sleep(poll_seconds)
            job = self.client.batches.get(name=job_name)
        return job

    def _collect(self, job) -> dict[str, BatchResult]:
        results: dict[str, BatchResult] = {}
        dest = getattr(job, "dest", None)
        if dest and dest.inlined_responses:
            for item in dest.inlined_responses:
                key = (item.metadata or {}).get("key") if item.metadata else None
                if not key:
                    continue
                if item.error:
                    message = getattr(item.error, "message", None) or str(item.error)
                    results[str(key)] = BatchResult(key=str(key), text=None, error=message)
                    continue
                text = _response_text(item.response)
                results[str(key)] = BatchResult(
                    key=str(key),
                    text=text,
                    error=None if text else "empty batch response",
                )
            return results
        if dest and dest.file_name:
            content = self.client.files.download(file=dest.file_name)
            text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
            for line in text.splitlines():
                parsed = _parse_result_line(line)
                if parsed:
                    results[parsed.key] = parsed
            return results
        raise RuntimeError(f"Gemini batch job {job.name} succeeded without downloadable results")
