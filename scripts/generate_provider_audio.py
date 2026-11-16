"""Compose Strudel with an LLM provider and save MusicPrefs benchmark WAVs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

from pydantic import Field

from tunelm.composition.responses import STRUDEL_RULES, completion, parse_completion
from tunelm.core.io import (
    ROOT,
    content_hash,
    generation_lock,
    load_config,
    load_env_file,
    project_path,
    read_json,
    write_json,
)
from tunelm.core.schemas import Record
from tunelm.data.teachers import DEFAULT_ENV, Teacher, TeacherConfig
from tunelm.evaluation.audio_eval import DEFAULT_SECONDS, SAFE_NAME, model_audio, read_benchmark
from tunelm.evaluation.render import RENDER_SECONDS, render_audio

SYSTEM = (
    """You are an algorithmic Strudel composer. The request field is the
listener's prompt. Choose the musical details they leave open.
Return JSON fields schema_version (2), thinking, and code. In thinking, describe
the musical choices in ordinary prose. In code, return one complete Strudel
JavaScript program without Markdown fences.
Use exact sound IDs and drum-bank roles from the runtime reference. Strudel
method names are case-sensitive. Use ~ for rests in mini-notation. If feedback
is present, correct the failure and return the full program for the same prompt.
"""
    + STRUDEL_RULES
)


class Composition(Record):
    thinking: str = Field(min_length=1)
    code: str = Field(min_length=1)


def _runtime_reference() -> dict:
    methods = load_config("configs/registry/api.yaml")["methods"]
    sounds = load_config("configs/registry/sounds.yaml")["sounds"]
    banks = {}
    for sound in sounds:
        if "_" in sound and not sound.startswith("gm_"):
            bank, role = sound.rsplit("_", 1)
            if role:
                banks.setdefault(bank, []).append(role)
    return {
        "common_strudel_api": [entry["signature"] for entry in methods.values()],
        "sound_ids": [name for name in sounds if name.startswith("gm_") or "_" not in name],
        "drum_banks_and_available_roles": banks,
    }


def _provider_config(provider: str, model: str, max_tokens: int | None) -> TeacherConfig:
    config = TeacherConfig(provider=provider, model=model)
    guide = ROOT / "configs" / "samples-song-describer.yaml"
    profiles = load_config(guide)["models"] if guide.is_file() else []
    profile = next(
        (
            row
            for row in profiles
            if row["provider"] == config.provider and row["model"] == config.model
        ),
        {},
    )
    return TeacherConfig(
        provider=config.provider,
        model=config.model,
        max_tokens=profile.get("max_tokens", config.max_tokens)
        if max_tokens is None
        else max_tokens,
        thinking_level=profile.get("thinking_level"),
    )


def _request(
    teacher: Teacher, prompt_id: str, prompt: str, attempt: int, reference, feedback
) -> dict:
    start = perf_counter()
    record = {"prompt_id": prompt_id, "prompt": prompt, "attempt": attempt}
    try:
        reply = teacher.generate({"request": prompt, "runtime_reference": reference}, feedback)
        record["completion"] = completion(reply.thinking, reply.code)
    except ValueError as error:
        record["error"] = str(error)
    except Exception as error:
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        detail = f"HTTP {status}" if isinstance(status, int) else type(error).__name__
        raise RuntimeError(f"{teacher.identity} request failed ({detail})") from None
    return {**record, "usage": teacher.last_usage, "api_seconds": perf_counter() - start}


def _feedback(error: str) -> str:
    if "AudioWorkletGlobalScope.js" in error:
        return (
            "The offline AudioWorklet crashed. Replace worklet-heavy instruments or effects "
            "with simpler synthesis and effects for this composition."
        )
    lines = [line for line in error.splitlines() if line.strip()]
    if lines and "validation error" in lines[0]:
        return "Return valid JSON with schema_version (2), thinking, and complete Strudel code."
    return (lines[-1] if lines else "The composition could not be rendered.")[:600]


def _save_prompt(pid, prompt, audio, cache, attempts, request):
    feedback = []
    for attempt in range(1, attempts + 1):
        suffix = "" if attempt == 1 else f"-attempt-{attempt}"
        path = audio / "completions" / f"{pid}{suffix}.json"
        if path.is_file():
            saved = read_json(path)
        else:
            saved = request(pid, prompt, attempt, feedback)
            write_json(path, saved)
        if (saved["prompt_id"], saved["prompt"], saved["attempt"]) != (pid, prompt, attempt):
            raise ValueError(f"Saved completion does not match the prompt roster: {path}")
        try:
            if "error" in saved:
                raise ValueError(saved["error"])
            _, code = parse_completion(saved["completion"])
            render_audio(code, audio / f"{pid}.wav", cache, scoring_seconds=DEFAULT_SECONDS)
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            failure = {"prompt_id": pid, "attempt": attempt, "error": str(error)}
            feedback = [_feedback(str(error))]
            print(f"{pid} attempt {attempt} failed: {feedback[0]}", file=sys.stderr, flush=True)
        else:
            return None
    return failure


def _record_run(audio: Path, settings: dict, present: dict, resume: bool) -> None:
    metadata = audio / "generation.json"
    if metadata.is_file():
        prior = read_json(metadata)
        history = prior.pop("previous_settings", [])
        identity = ("provider", "model", "prompt_roster_sha256", "render_seconds")
        if any(prior.get(key) != settings[key] for key in identity):
            raise ValueError(
                f"{audio.name} already uses a different model or benchmark; use --name"
            )
        if prior != settings:
            if not resume:
                raise ValueError(
                    f"{audio.name} uses earlier settings; pass --resume or a new --name"
                )
            write_json(metadata, {**settings, "previous_settings": [*history, prior]})
    elif present or (audio / "completions").exists():
        raise ValueError(f"{audio.name} already contains another experiment; choose a new --name")
    else:
        write_json(metadata, settings)


def generate_audio(
    benchmark: Path, name: str, config: TeacherConfig, attempts: int, resume: bool = False
) -> dict:
    prompts = read_benchmark(benchmark)
    reference = _runtime_reference()
    audio = benchmark / "audio" / name
    settings = {
        "schema_version": 2,
        "provider": config.provider,
        "model": config.model,
        "max_tokens": config.max_tokens,
        "thinking_level": config.thinking_level,
        "prompt_roster_sha256": content_hash(prompts),
        "system_sha256": content_hash(SYSTEM),
        "reference_sha256": content_hash(reference),
        "render_seconds": RENDER_SECONDS,
    }
    with generation_lock(audio), ExitStack() as resources:
        present = model_audio(audio)
        _record_run(audio, settings, present, resume)
        client = None
        settings_hash = content_hash(settings)

        def request(pid, prompt, attempt, feedback):
            nonlocal client
            if client is None:
                client = resources.enter_context(
                    Teacher(config, system=SYSTEM, response_model=Composition)
                )
            return {
                **_request(client, pid, prompt, attempt, reference, feedback),
                "settings_sha256": settings_hash,
                "feedback": feedback,
            }

        missing = [(pid, prompt) for pid, prompt in prompts if pid not in present]
        failures = []
        for index, (pid, prompt) in enumerate(missing, 1):
            print(f"{name} {index}/{len(missing)} {pid}", file=sys.stderr, flush=True)
            if failure := _save_prompt(
                pid, prompt, audio, benchmark / ".render-assets", attempts, request
            ):
                failures.append(failure)
        failure_path = audio / "failures.json"
        if failures:
            write_json(failure_path, failures)
        else:
            failure_path.unlink(missing_ok=True)

    return {
        "experiment": name,
        "audio_dir": str(audio),
        "prompts": len(prompts),
        "reused_wavs": len(prompts) - len(missing),
        "new_wavs": len(missing) - len(failures),
        "failed_prompts": len(failures),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=[*DEFAULT_ENV, "claude"])
    parser.add_argument("--model", required=True, help="Provider's exact model ID")
    parser.add_argument("--benchmark", default="datasets/audio-benchmarks/musicprefs")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--name", help="Experiment directory name (default: provider-model)")
    parser.add_argument("--attempts", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--max-tokens", type=int, help="Override the provider profile's token budget"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Apply updated settings to an existing run"
    )
    args = parser.parse_args()
    try:
        load_env_file(args.env_file)
        config = _provider_config(args.provider, args.model, args.max_tokens)
        name = args.name or f"{config.provider}-{re.sub(r'[^a-zA-Z0-9._-]+', '-', config.model)}"
        if not SAFE_NAME.fullmatch(name):
            raise ValueError("--name must be a single directory name")
        result = generate_audio(
            project_path(args.benchmark), name, config, args.attempts, args.resume
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        sys.exit(str(error))
    print(json.dumps(result, indent=2))
    if result["failed_prompts"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
