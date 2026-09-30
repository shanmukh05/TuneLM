#!/usr/bin/env python3
"""Call a deployed TuneLM Modal endpoint and save train.jsonl-compatible rows."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tunelm.inference_limits import DEFAULT_MAX_NEW_TOKENS
from tunelm.io import read_jsonl, write_jsonl
from tunelm.eval.manual_prompts import MANUAL_PROMPTS
from tunelm.models.templates import task_from_example, task_user_prompt
from tunelm.strudel.executor import StrudelExecutor

DEFAULT_SAMPLES = "datasets/sft/sample.jsonl"
DEFAULT_OUTPUT = "results/modal_inference_best_new.jsonl"

TRAIN_NULL_FIELDS = (
    "original_code",
    "instruction",
    "expected_changes",
    "preserve",
    "broken_code",
    "reference_code",
    "corruption",
)


def load_sample_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def manual_compose_row(prompt: str, *, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"infer-compose-manual-{index:04d}",
        "task_type": "compose",
        "prompt": prompt,
        "constraints": {},
        "metadata": {},
    }
    for field in TRAIN_NULL_FIELDS:
        row[field] = None
    return row


def model_response(model_output: dict[str, Any]) -> dict[str, Any] | None:
    if not all(key in model_output for key in ("plan", "controls", "strudel_code")):
        return None
    return {
        "plan": model_output["plan"],
        "controls": model_output["controls"],
        "strudel_code": model_output["strudel_code"],
    }


def train_row_from_inference(
    source: dict[str, Any] | None,
    *,
    prompt: str,
    model_output: dict[str, Any],
    index: int,
    endpoint: str,
) -> dict[str, Any]:
    row = copy.deepcopy(source) if source else manual_compose_row(prompt, index=index)
    response = model_response(model_output)
    row["response"] = response if response is not None else model_output

    metadata = dict(row.get("metadata") or {})
    for key in (
        "teacher", "teacher_provider", "teacher_model", "valid", "validation",
        "generation_attempts", "created_at",
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "inference_source": "modal",
            "endpoint": endpoint,
            "inferred_at": datetime.now(UTC).isoformat(),
        }
    )
    if "validation" in model_output:
        metadata["validation"] = model_output["validation"]
    if model_output.get("detail"):
        metadata["parse_error"] = str(model_output["detail"])
    if model_output.get("raw_completion"):
        metadata["raw_completion"] = model_output["raw_completion"]
    if response is None and "parse_error" not in metadata:
        metadata["parse_error"] = "model output missing plan/controls/strudel_code"
    row["metadata"] = metadata
    return row


def call_endpoint(
    url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    validate: bool,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "run_validation": validate,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            parsed = {"detail": detail or str(error)}
        if error.code == 422:
            if isinstance(parsed.get("detail"), dict):
                return parsed["detail"]
            return parsed
        raise
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}
    if isinstance(parsed, dict):
        return parsed
    return {"raw": parsed}


def resolve_jobs(args: argparse.Namespace) -> list[tuple[dict[str, Any] | None, str]]:
    if args.samples is not None:
        samples_path = Path(args.samples)
        if not samples_path.is_file():
            raise FileNotFoundError(f"samples file not found: {samples_path}")
        rows = load_sample_rows(samples_path)
        if args.limit is not None:
            rows = rows[: args.limit]
        return [(row, task_user_prompt(task_from_example(row))) for row in rows]
    if args.prompt:
        return [(None, args.prompt)]
    if MANUAL_PROMPTS:
        prompts = MANUAL_PROMPTS[: args.limit] if args.limit is not None else MANUAL_PROMPTS
        return [(None, prompt) for prompt in prompts]
    raise RuntimeError("pass a prompt, use --samples, or fill MANUAL_PROMPTS in the script")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="POST prompts to a TuneLM Modal endpoint")
    parser.add_argument("prompt", nargs="?", help="single user prompt")
    parser.add_argument("--url", required=True, help="Modal web endpoint URL")
    parser.add_argument(
        "--samples",
        nargs="?",
        const=DEFAULT_SAMPLES,
        help=f"read tasks from a TuneLM JSONL file (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"write train.jsonl-compatible rows (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--limit", type=int, help="only run the first N prompts")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compile-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)

    try:
        jobs = resolve_jobs(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    executor = StrudelExecutor(backend="node", cycles=8, timeout_seconds=5, max_events=8192)
    for index, (source, user_prompt) in enumerate(jobs, start=1):
        label = (source or {}).get("id") or f"prompt-{index}"
        print(f"[{index}/{len(jobs)}] {label}", file=sys.stderr)
        try:
            model_output = call_endpoint(
                args.url,
                user_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                validate=args.validate,
                timeout=args.timeout,
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(detail or str(error), file=sys.stderr)
            return 1
        except urllib.error.URLError as error:
            print(error, file=sys.stderr)
            return 1

        attempts = 1
        while True:
            response = model_response(model_output)
            execution = executor.run(response["strudel_code"]) if response is not None else None
            if (execution is not None and execution.playable) or attempts > args.compile_retries:
                break
            error = execution.error if execution is not None else model_output.get("detail", "invalid JSON")
            repair_prompt = (
                f"{user_prompt}\n\nYour previous response failed Strudel validation: "
                f"{str(error)[:600]}\nReturn the complete corrected JSON response."
            )
            try:
                model_output = call_endpoint(
                    args.url,
                    repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=min(args.temperature, 0.2),
                    validate=False,
                    timeout=args.timeout,
                )
            except urllib.error.URLError as error:
                model_output = {"detail": str(error)}
            attempts += 1

        row = train_row_from_inference(
            source,
            prompt=user_prompt,
            model_output=model_output,
            index=index,
            endpoint=args.url,
        )
        row["metadata"]["inference_attempts"] = attempts
        row["metadata"]["valid"] = bool(execution is not None and execution.playable)
        if execution is not None:
            row["metadata"]["validation"] = execution.model_dump()
        results.append(row)
        code = ""
        response = row.get("response")
        if isinstance(response, dict):
            code = str(response.get("strudel_code") or "")[:80].replace("\n", " ")
        print(f"  saved {row['id']}: {code}", file=sys.stderr)

    output_path = Path(args.output)
    write_jsonl(output_path, results)
    print(f"wrote {len(results)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
