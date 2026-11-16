"""Evaluate saved model audio and fill missing TuneLM benchmark renders."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tunelm.core.io import (
    ROOT,
    _atomic_text,
    generation_lock,
    load_config,
    output_path,
    project_path,
    read_json,
    write_json,
)
from tunelm.core.schemas import Record

AUDIO_SUFFIXES = {".wav", ".mp4", ".mp3", ".flac"}
DEFAULT_SECONDS = 5.0
SAFE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\Z")


class AudioExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    audio_dir: str | None = Field(default=None, min_length=1)


class AudioEvalConfig(Record):
    benchmark: str = "datasets/audio-benchmarks/musicprefs"
    tunelm_name: str = "tunelm-sft"
    inference_config: str = "configs/inference.yaml"
    inference_python: str = ".venv/bin/python"
    env_file: str = ".env.local"
    baseline_only: bool = False
    output: str
    seconds: float = Field(default=DEFAULT_SECONDS, ge=1, le=30)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    fad_reference: str | None = None
    mad_reference: str
    embedding_cache: str | None = None
    experiments: list[AudioExperiment] | None = Field(default=None, min_length=1)
    batch_size: int = Field(default=1, ge=1, le=32)
    render_workers: int = Field(default=1, ge=1, le=16)
    attempts: int = Field(default=1, ge=1, le=8)
    retry_temperature: float = Field(default=0.4, gt=0, le=2)
    match_rendered: bool = False

    @field_validator("experiments", mode="before")
    @classmethod
    def parse_experiments(cls, value):
        if isinstance(value, list):
            return [{"name": entry} if isinstance(entry, str) else entry for entry in value]
        return value

    @field_validator("experiments")
    @classmethod
    def unique_experiments(cls, value):
        if value and len({entry.name for entry in value}) != len(value):
            raise ValueError("Experiment names must be unique")
        return value

    @classmethod
    def from_cli(cls, args):
        fields = vars(args).keys() & cls.model_fields.keys()
        settings = load_config(args.config) if args.config else {}
        settings.update(
            {key: getattr(args, key) for key in fields if getattr(args, key) is not None}
        )
        config = cls.model_validate(settings)
        if selected := getattr(args, "selected_experiments", None):
            configured = {entry.name: entry for entry in config.experiments or []}
            config.experiments = cls.unique_experiments(
                [
                    configured[name] if name in configured else AudioExperiment(name=name)
                    for name in selected
                ]
            )
        return config


@dataclass(frozen=True)
class AudioPrediction:
    model_name: str
    audio_path: Path
    prompt: str
    prompt_id: str


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]


def write_csv(path: Path, columns: list[str], rows) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def read_csv(path: Path, columns: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} requires columns: {', '.join(sorted(columns))}")
        rows = []
        for row in reader:
            if None in row or any(not (row.get(name) or "").strip() for name in columns):
                raise ValueError(f"{path}:{reader.line_num}: missing or malformed CSV value")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def read_predictions(path: Path) -> list[AudioPrediction]:
    rows = read_csv(path, {"model_name", "audio_path", "prompt"})
    predictions = []
    seen = set()
    prompts = {}
    for row in rows:
        clip = (path.parent / row["audio_path"].strip()).resolve()
        if clip.suffix.lower() not in AUDIO_SUFFIXES or not clip.is_file():
            raise ValueError(f"Expected a saved WAV, MP4, MP3 or FLAC file: {clip}")
        model, prompt = row["model_name"].strip(), row["prompt"]
        identifier = row.get("prompt_id") or prompt_id(prompt)
        if prompts.setdefault(identifier, prompt) != prompt:
            raise ValueError(f"Prompt ID {identifier} has inconsistent text")
        key = (model, identifier)
        if key in seen:
            raise ValueError(f"Expected one take for {model} / {identifier}")
        seen.add(key)
        predictions.append(AudioPrediction(model, clip, prompt, identifier))
    expected = set(prompts)
    for model in {row.model_name for row in predictions}:
        actual = {row.prompt_id for row in predictions if row.model_name == model}
        if actual != expected:
            raise ValueError(f"{model} is missing {len(expected - actual)} prompt IDs")
    if len(expected) < 2:
        raise ValueError("Audio evaluation requires at least two prompts per model")
    return sorted(predictions, key=lambda row: (row.model_name, row.prompt_id))


def read_benchmark(benchmark: Path) -> list[tuple[str, str]]:
    rows = read_csv(benchmark / "prompts.csv", {"prompt_id", "prompt"})
    prompts = {}
    for row in rows:
        identifier = row["prompt_id"]
        if not SAFE_NAME.fullmatch(identifier) or identifier in prompts:
            raise ValueError(f"Invalid or repeated benchmark prompt ID: {identifier!r}")
        prompts[identifier] = row["prompt"]
    return sorted(prompts.items())


def model_audio(directory: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()) if directory.is_dir() else ():
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            if path.stem in files:
                raise ValueError(f"Multiple audio files for {directory.name} / {path.stem}")
            files[path.stem] = path
    return files


def benchmark_inputs(benchmark: Path, tunelm_name: str, baseline_only: bool, prompts=None):
    if prompts is None:
        prompts = read_benchmark(benchmark)
    audio_root = benchmark / "audio"
    metadata = benchmark / "musicprefs.json"
    if metadata.is_file():
        models = set(read_json(metadata)["models"])
        if tunelm_name in models:
            raise ValueError("Use a TuneLM model name distinct from the MusicPrefs model names")
    else:
        models = {path.name for path in audio_root.iterdir() if path.is_dir()} - {tunelm_name}
    if not baseline_only:
        models.add(tunelm_name)
    if not models:
        raise ValueError(f"No model audio directories under {audio_root}")
    for model_name in sorted(models):
        directory = audio_root / model_name
        clips = model_audio(directory)
        missing = [identifier for identifier, _ in prompts if identifier not in clips]
        if missing:
            raise ValueError(
                f"{model_name} is missing {len(missing)} clips, including {missing[0]}"
            )
        for identifier, prompt in prompts:
            yield (model_name, str(clips[identifier].resolve()), prompt, identifier)


def score_audio(args, benchmark: Path | None = None, prompts=None, coverage=None, *, rows=None):
    from tunelm.evaluation.audio_metrics import file_sha256, score_models

    output = output_path(Path(args.output) / "metrics.json").parent
    with generation_lock(output):
        if benchmark is None and rows is None:
            manifest = project_path(args.input)
        else:
            manifest = output / "inputs.csv"
            if rows is None:
                rows = benchmark_inputs(benchmark, args.tunelm_name, args.baseline_only, prompts)
            write_csv(
                manifest,
                ["model_name", "audio_path", "prompt", "prompt_id"],
                (
                    (name, os.path.relpath(path, output), prompt, pid)
                    for name, path, prompt, pid in rows
                ),
            )
        inputs = read_predictions(manifest)
        with redirect_stdout(sys.stderr):
            summary, clips = score_models(
                inputs,
                output,
                seconds=args.seconds,
                device=args.device,
                fad_reference=project_path(args.fad_reference) if args.fad_reference else None,
                mad_reference=project_path(args.mad_reference),
                embedding_cache=project_path(args.embedding_cache)
                if args.embedding_cache
                else None,
            )
        summary["input"] = str(manifest)
        summary["input_sha256"] = file_sha256(manifest)
        if coverage is not None:
            summary["coverage"] = coverage
        write_csv(
            output / "metrics.csv",
            list(summary["models"][0]),
            (model.values() for model in summary["models"]),
        )
        write_csv(output / "clap.csv", list(clips[0]), (clip.values() for clip in clips))
        write_json(output / "metrics.json", summary)
    return {"metrics": str(output / "metrics.json"), **summary}


def _evaluate_experiments(args: AudioEvalConfig, benchmark: Path) -> dict:
    prompts = dict(read_benchmark(benchmark))
    shared = set(prompts)
    clips_by_model, coverage = {}, {}
    for entry in args.experiments:
        directory = (
            project_path(entry.audio_dir) if entry.audio_dir else benchmark / "audio" / entry.name
        )
        clips = model_audio(directory)
        available = set(prompts) & clips.keys()
        if not available:
            raise ValueError(f"{entry.name} has no saved clips for this benchmark in {directory}")
        missing = sorted(prompts.keys() - available)
        if missing and not args.match_rendered:
            raise ValueError(f"{entry.name} is missing {len(missing)} benchmark clips")
        clips_by_model[entry.name] = clips
        coverage[entry.name] = {"available_prompts": len(available), "missing_prompt_ids": missing}
        shared &= available
    if len(shared) < 2:
        raise ValueError("Selected experiments need at least two shared saved prompts")
    rows = (
        (name, clips[pid].resolve(), prompts[pid], pid)
        for name, clips in clips_by_model.items()
        for pid in sorted(shared)
    )
    return score_audio(
        args,
        rows=rows,
        coverage={
            "requested_prompts": len(prompts),
            "scored_prompts": len(shared),
            "experiments": coverage,
        },
    )


def _generate_tunelm(args, benchmark: Path) -> None:
    # Resolving a venv's python symlink would lose its environment.
    python = ROOT / Path(args.inference_python).expanduser()
    command = [
        str(python),
        str(ROOT / "scripts" / "generate_benchmark_audio.py"),
        "--benchmark",
        str(benchmark),
        "--name",
        args.tunelm_name,
        "--config",
        args.inference_config,
        "--env-file",
        args.env_file,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--render-workers",
        str(args.render_workers),
        "--seconds",
        str(args.seconds),
        "--attempts",
        str(args.attempts),
        "--retry-temperature",
        str(args.retry_temperature),
    ]
    code = subprocess.run(command, stdout=sys.stderr, check=False).returncode
    if code and not (code == 2 and args.match_rendered):
        raise RuntimeError("TuneLM inference or rendering failed. Saved clips can be reused.")


def evaluate_audio(args: AudioEvalConfig) -> dict:
    benchmark = project_path(args.benchmark)
    if not project_path(args.mad_reference).exists():
        raise ValueError(f"MAD reference does not exist: {args.mad_reference}")
    if args.experiments:
        if args.baseline_only:
            raise ValueError(
                "Select archived models in experiments instead of using --baseline-only"
            )
        return _evaluate_experiments(args, benchmark)
    if not SAFE_NAME.fullmatch(args.tunelm_name):
        raise ValueError("--tunelm-name must be a single directory name")
    prompts = read_benchmark(benchmark)
    audio_dir = benchmark / "audio" / args.tunelm_name
    list(benchmark_inputs(benchmark, args.tunelm_name, baseline_only=True))
    if args.baseline_only:
        return score_audio(args, benchmark)
    generation = audio_dir / "generation.json"
    if generation.exists() and read_json(generation)["settings"] != load_config(
        args.inference_config
    ):
        raise ValueError(f"{args.tunelm_name} uses different settings; choose a new --tunelm-name")
    if set(dict(prompts)) - model_audio(audio_dir).keys():
        _generate_tunelm(args, benchmark)
    files = model_audio(audio_dir)
    missing = [identifier for identifier, _ in prompts if identifier not in files]
    coverage = {
        "requested_prompts": len(prompts),
        "rendered_tunelm_prompts": len(prompts) - len(missing),
        "missing_tunelm_prompt_ids": missing,
    }
    scored = [row for row in prompts if row[0] in files] if args.match_rendered else prompts
    return score_audio(args, benchmark, prompts=scored, coverage=coverage)
