"""Shared engine that authors one input bank (SFT or RL) before any music answers exist.

A run's `stage` picks its `AuthorStage` (`prompting.SFT_STAGE`, `rl_prompts.RL_STAGE`);
this module only owns quotas, journals, resume, de-duplication, and export. Banks
share one output_dir so a request can never repeat across SFT and RL. Each author
model runs at the same time; journals and the duplicate check stay serialized.
"""

import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tunelm.generation_progress import GenerationProgress, RunOptions
from tunelm.generation_state import allocate_models
from tunelm.io import (
    append_jsonl,
    content_hash,
    generation_lock,
    load_config,
    output_path,
    project_path,
    read_json,
    read_jsonl,
    recover_journal,
    write_json,
    write_jsonl,
)
from tunelm.prompting import SFT_STAGE, AuthorStage, PromptConfig, validate_prompt_style
from tunelm.rl_prompts import RL_STAGE
from tunelm.schemas import Category, MusicRequest, Split
from tunelm.teachers import Teacher, TeacherModel
from tunelm.verifier import Verifier

STAGES: dict[str, AuthorStage] = {"sft": SFT_STAGE, "rl": RL_STAGE}


class PromptGenerationConfig(RunOptions):
    """One bank per config. Counts describe inputs, not music answers or rollouts."""

    stage: Literal["sft", "rl"]
    run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    output_dir: str
    palette: PromptConfig
    counts: dict[Split, int]
    categories: list[Category] = Field(default_factory=lambda: ["stack", "arrange"])
    max_attempts: int = Field(default=2, ge=1, le=10)
    avoid_sample: int = Field(default=40, ge=0)
    models: list[TeacherModel] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_counts(self):
        counts = list(self.counts.values())
        if not counts or min(counts) < 0 or sum(counts) < 1 or not self.categories:
            raise ValueError("Prompt counts must be nonnegative and total at least one")
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("Author model IDs must be unique")
        if abs(sum(m.fraction for m in self.models) - 1) > 1e-9:
            raise ValueError("Author model fractions must sum to 1")
        return self

    @property
    def count(self) -> int:
        return sum(self.counts.values())

    @property
    def author(self) -> AuthorStage:
        return STAGES[self.stage]


def load_prompt_generation_config(path: str) -> PromptGenerationConfig:
    """Load one self-contained bank config (configs/prompts-sft.yaml or prompts-rl.yaml)."""
    return PromptGenerationConfig.model_validate(load_config(path))


def prompt_preview(config: PromptGenerationConfig) -> dict:
    """Show author quotas and split counts without clients, downloads, or writes."""
    return {
        "stage": config.stage,
        "counts": config.counts,
        "total": config.count,
        "model_targets": allocate_models(config.models, config.count),
        "styles": config.palette.schedule,
        "output_dir": config.output_dir,
        "avoid_sample": config.avoid_sample,
        "max_requests": config.count * config.max_attempts,
        "api_requests": 0,
        "state_written": False,
        **config.runtime_settings(),
    }


def _prompt_jobs(config, models):
    assignments = [m["id"] for m in models for _ in range(m["target"])]
    jobs = []
    for split, count in config.counts.items():
        for _ in range(count):
            index = len(jobs)
            job = {
                "stage": config.stage,
                "split": split,
                "model_id": assignments[index],
                "category": config.categories[index % len(config.categories)],
                "brief": config.palette.brief(index),
            }
            job["id"] = content_hash([config.run_id, index, job])[:24]
            jobs.append(job)
    return jobs


def _open_prompt_run(config, directory):
    import os

    saved_path = project_path(directory / "run.json")
    saved = read_json(saved_path) if saved_path.exists() else {}
    previous = {m["id"]: m for m in saved.get("models", [])}
    targets = allocate_models(config.models, config.count)
    models = []
    for model in config.models:
        named = os.environ.get(model.model_env) if model.model_env else None
        name = model.model or named or previous.get(model.id, {}).get("model")
        if targets[model.id] and not name:
            raise ValueError(f"Set model for {model.id} in the config")
        models.append({"id": model.id, "model": name, "target": targets[model.id]})
    identity = content_hash([config.identity_config(), models, config.author.system])
    manifest = {"identity": identity, "config": config.model_dump(), "models": models}
    if saved:
        jobs = read_jsonl(directory / "jobs.jsonl")
        if saved["jobs_hash"] != content_hash(jobs):
            raise ValueError("Prompt jobs changed; use a new prompt run_id and output_dir")
        if any(job["stage"] != config.stage for job in jobs):
            raise ValueError(f"run_id {config.run_id} is not a {config.stage} bank run")
        if saved["identity"] != identity:
            saved = {**saved, "identity": identity, "config": manifest["config"], "models": models}
            write_json(saved_path, saved)
        return saved, jobs
    jobs = _prompt_jobs(config, models)
    manifest["jobs_hash"] = content_hash(jobs)
    write_jsonl(directory / "jobs.jsonl", jobs)
    write_json(saved_path, manifest)
    return manifest, jobs


def _exported_rows(config) -> list[dict]:
    rows = []
    for stage in ("sft", "rl"):
        for split in ("train", "val", "benchmark"):
            path = Path(config.output_dir) / stage / f"{split}.jsonl"
            if project_path(path).exists():
                rows.extend(read_jsonl(path))
    return rows


def _claim_saved_prompts(config, manifest, jobs, directory):
    """Pull this run's exported requests back into its journal.

    Rows from another run_id stay in the shared output_dir and are not taken
    as finished jobs of this run.
    """
    del manifest
    job_ids = {job["id"] for job in jobs}
    rows = [row for row in _exported_rows(config) if row["id"] in job_ids]
    known = {row["id"]: row for row in recover_journal(directory / "accepted.jsonl")}
    for row in rows:
        MusicRequest.model_validate(row)
        current = known.get(row["id"])
        if current is not None and current != row:
            raise ValueError(
                "Prompt JSONL was reviewed/edited. Keep it for answer generation; "
                "choose a fresh author output_dir to generate more prompts."
            )
        if current is None:
            append_jsonl(directory / "accepted.jsonl", row)
    return jobs


@contextmanager
def _output_lock(directory: Path):
    """One shared-bank write at a time. Other runs wait instead of replacing the file."""
    import fcntl

    with output_path(directory / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _restore_prompts(directory):
    accepted = {r["id"]: r for r in recover_journal(directory / "accepted.jsonl")}
    history = defaultdict(list)
    for row in recover_journal(directory / "attempts.jsonl"):
        history[row["job_id"]].append(row)
        completed = row.get("accepted")
        if completed and completed["id"] not in accepted:
            append_jsonl(directory / "accepted.jsonl", completed)
            accepted[completed["id"]] = completed
    for row in accepted.values():
        MusicRequest.model_validate(row)
    return accepted, history


def _prompt_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def _remember_prompt(seen: list[tuple[str, list[str], set[str], str]], prompt: str) -> None:
    words = _prompt_words(prompt)
    seen.append((" ".join(words), words, set(words), prompt))


def _seen_prompts(rows) -> list[tuple[str, list[str], set[str], str]]:
    seen: list[tuple[str, list[str], set[str], str]] = []
    for row in rows:
        _remember_prompt(seen, row["prompt"])
    return seen


def _shared_seen(config, accepted) -> list[tuple[str, list[str], set[str], str]]:
    """Requests already in the output_dir, plus this run's journal."""
    seen = _seen_prompts(_exported_rows(config))
    have = {prompt for _, _, _, prompt in seen}
    for row in accepted.values():
        if row["prompt"] not in have:
            _remember_prompt(seen, row["prompt"])
            have.add(row["prompt"])
    return seen


def _avoid_sample(seen: list[tuple[str, list[str], set[str], str]], limit: int) -> list[str]:
    """Pick up to `limit` accepted requests for the author to avoid.

    Below the cap, the author sees every request already accepted. Above it, the
    sample is a new random subset on each attempt.
    """
    prompts = [prompt for _, _, _, prompt in seen]
    if limit <= 0 or not prompts:
        return []
    if len(prompts) <= limit:
        return prompts
    # ponytail: unseeded, so a retry does not repeat the same subset. Seed from the job id to make one retry stable. Cap is avoid_sample.
    return random.sample(prompts, limit)


def _reject_repeat(prompt: str, seen: list[tuple[str, list[str], set[str], str]]) -> None:
    """Reject a request that repeats an accepted one past a swapped word or two.

    Eight shared opening words, or half the vocabulary, is the same ask. The
    author then receives the earlier text and has to write a different one.
    """
    words = _prompt_words(prompt)
    exact = " ".join(words)
    tokens = set(words)
    for previous, previous_words, previous_tokens, _prompt in seen:
        if exact == previous:
            raise ValueError(f"This repeats an accepted request. Write a different one: {previous}")
        shared = 0
        for left, right in zip(words, previous_words):
            if left != right:
                break
            shared += 1
        if shared >= 8 or (
            len(tokens) >= 4
            and len(previous_tokens) >= 4
            and len(tokens & previous_tokens) / len(tokens | previous_tokens) >= 0.5
        ):
            raise ValueError(
                "This is too close to an accepted request. Change the situation and the music, "
                f"not just a few words: {previous}"
            )


def _request_row(reply, job, teacher, seen, author: AuthorStage):
    validate_prompt_style(reply.prompt, job["brief"], allow_tempo_key=author.allow_tempo_key)
    _reject_repeat(reply.prompt, seen)
    row = MusicRequest(
        id=job["id"],
        stage=job["stage"],
        split=job["split"],
        prompt=reply.prompt,
        category=job["category"],
        prompt_style=job["brief"]["style"],
        prompt_scenario=None,
        original_code=reply.original_code,
        author=teacher.identity,
        author_id=teacher.id,
        contract=getattr(reply, "contract", None),
    )
    author.check(row)
    if row.original_code:
        result = Verifier().request(row.original_code)
        if not result["valid"]:
            raise ValueError(f"Co-creation starting pattern failed: {result.get('error', result)}")
    return row.model_dump()


def _author_payload(config, job, seen, lock) -> dict:
    payload = {
        "brief": job["brief"],
        "category": job["category"],
        "stage": job["stage"],
        **config.author.payload(),
    }
    with lock:
        shown = _avoid_sample(seen, config.avoid_sample)
    if shown:
        payload["already_written"] = shown
    return payload


def _copy_seen(seen, lock):
    with lock:
        return list(seen)


def _commit_attempt(directory, job, teacher, retry, reply, accepted, feedback, seen, entries, lock):
    """Journal one attempt. A prompt taken by a concurrent author becomes a retry."""
    with lock:
        if accepted:
            try:
                _reject_repeat(accepted["prompt"], seen)
            except ValueError as error:
                accepted, feedback = None, [str(error)]
            else:
                _remember_prompt(seen, accepted["prompt"])
        entry = {
            "job_id": job["id"],
            "retry": retry,
            "model_id": teacher.id,
            "reply": reply.model_dump() if reply else None,
            "accepted": accepted,
            "feedback": feedback,
            "usage": teacher.last_usage,
        }
        append_jsonl(directory / "attempts.jsonl", entry)
        entries.append(entry)
        if accepted:
            append_jsonl(directory / "accepted.jsonl", accepted)
    return accepted


def _author_job(config, job, teacher, entries, seen, directory, progress, lock):
    feedback = entries[-1]["feedback"] if entries else []
    for retry in range(len(entries), config.max_attempts):
        reply, accepted = None, None
        with lock:
            progress.attempt(teacher.id, retry, config.max_attempts)
        try:
            reply = teacher.generate(_author_payload(config, job, seen, lock), feedback)
            accepted = _request_row(reply, job, teacher, _copy_seen(seen, lock), config.author)
            feedback = []
        except ValueError as error:
            feedback = [str(error)]
        accepted = _commit_attempt(
            directory, job, teacher, retry, reply, accepted, feedback, seen, entries, lock
        )
        if accepted:
            return accepted
    return None


def _export_prompt_bank(config, manifest, accepted, status):
    """Merge this run into the shared output_dir. Other runs' requests stay."""
    with _output_lock(Path(config.output_dir)):
        _check_reviewed_inputs(config, accepted)
        merged = {row["id"]: row for row in _exported_rows(config)}
        merged.update(accepted)
        rows = sorted(merged.values(), key=lambda r: r["id"])
        for stage in ["sft", "rl"]:
            for split in ["train", "val", "benchmark"]:
                write_jsonl(
                    Path(config.output_dir) / stage / f"{split}.jsonl",
                    [r for r in rows if r["stage"] == stage and r["split"] == split],
                )
        bank = [r for r in rows if r["stage"] == config.stage]
        report = {
            "identity": manifest["identity"],
            "stage": config.stage,
            "output_dir": config.output_dir,
            "status": status,
            "accepted": len(bank),
            "target": config.count,
            "valid": len(accepted) == config.count,
            "splits": dict(Counter(r["split"] for r in bank)),
            "records_hash": content_hash(bank),
            **config.runtime_settings(),
        }
        write_json(Path(config.output_dir) / config.stage / "manifest.json", report)
        return report


def _reassign_removed_authors(config, jobs, accepted, history, directory) -> None:
    """Give unfinished jobs to the current authors when the model list has changed.

    Accepted requests keep the author that wrote them, and the job id stays so the
    journal still matches.
    """
    current = {model.id for model in config.models}
    pending = [
        job
        for job in jobs
        if job["model_id"] not in current
        and job["id"] not in accepted
        and len(history[job["id"]]) < config.max_attempts
    ]
    if not pending:
        return
    quotas = allocate_models(config.models, len(pending))
    assigned = [model_id for model_id, count in quotas.items() for _ in range(count)]
    for job, model_id in zip(pending, assigned, strict=True):
        job["model_id"] = model_id
    write_jsonl(directory / "jobs.jsonl", jobs)
    saved = read_json(directory / "run.json")
    saved["jobs_hash"] = content_hash(jobs)
    write_json(directory / "run.json", saved)


def generate_prompts(config: PromptGenerationConfig) -> dict:
    """Generate input-only banks; resume accepted/rejected attempts without authoring answers."""
    directory = Path("results/prompt-generation") / config.run_id
    with generation_lock(directory):
        manifest, jobs = _open_prompt_run(config, directory)
        jobs = _claim_saved_prompts(config, manifest, jobs, directory)
        accepted, history = _restore_prompts(directory)
        _reassign_removed_authors(config, jobs, accepted, history, directory)
        # Refresh a lagging export from the authoritative journals before resuming.
        _export_prompt_bank(config, manifest, accepted, "running")
        completed = sum(
            j["id"] in accepted or len(history[j["id"]]) >= config.max_attempts for j in jobs
        )
        with GenerationProgress(
            "Prompts",
            len(jobs),
            accepted=len(accepted),
            completed=completed,
            attempts=sum(len(h) for h in history.values()),
            enabled=config.progress,
        ) as progress:
            try:
                _run_authors(config, manifest, jobs, accepted, history, directory, progress)
            except BaseException:
                _prompt_checkpoint(config, directory, progress, "interrupted")
                _export_prompt_bank(config, manifest, accepted, "interrupted")
                raise
            status = "complete" if len(accepted) == config.count else "exhausted"
            _prompt_checkpoint(config, directory, progress, status)
            return _export_prompt_bank(config, manifest, accepted, status)


def _check_reviewed_inputs(config, accepted):
    """Do not overwrite an edited request from this run. Other runs' rows stay."""
    current = {row["id"]: row for row in _exported_rows(config)}
    changed = any(current[row_id] != row for row_id, row in accepted.items() if row_id in current)
    if changed:
        raise ValueError(
            "Prompt JSONL was reviewed/edited. Keep it for answer generation; "
            "choose a fresh author output_dir to generate more prompts."
        )


def _prompt_checkpoint(config, directory, progress, status):
    if status != "running":
        progress.finish(status)
    write_json(
        directory / "progress.json", {**progress.snapshot(status), **config.runtime_settings()}
    )


def _pending_by_model(config, jobs, accepted, history) -> dict[str, list]:
    pending = defaultdict(list)
    for job in jobs:
        if job["id"] not in accepted and len(history[job["id"]]) < config.max_attempts:
            pending[job["model_id"]].append(job)
    return pending


def _publish(config, manifest, accepted, directory, progress, lock, saved_at):
    """Record one finished job and refresh the export. The lock keeps exports ordered."""
    with lock:
        progress.advance(len(accepted))
        _prompt_checkpoint(config, directory, progress, "running")
        if len(accepted) - saved_at[0] >= config.save_every:
            _export_prompt_bank(config, manifest, accepted, "running")
            saved_at[0] = len(accepted)


def _author_model(config, model_id, model_jobs, run):
    """Own one model's client and work through its jobs. Other models run alongside."""
    names, accepted, history, seen, directory, progress, lock, manifest, saved_at = run
    profile = next((model for model in config.models if model.id == model_id), None)
    if profile is None:
        raise ValueError(f"Job author {model_id} is not in the current model list")
    with Teacher(
        profile.model_copy(update={"model": names[profile.id]}),
        model_id=profile.id,
        system=config.author.system,
        response_model=config.author.reply,
    ) as teacher:
        for job in model_jobs:
            row = _author_job(
                config, job, teacher, history[job["id"]], seen, directory, progress, lock
            )
            with lock:
                if row:
                    accepted[row["id"]] = row
            _publish(config, manifest, accepted, directory, progress, lock, saved_at)


def _run_authors(config, manifest, jobs, accepted, history, directory, progress):
    """Run every author model at once. One model keeps the old one-at-a-time behavior."""
    pending = _pending_by_model(config, jobs, accepted, history)
    if not pending:
        return
    run = (
        {model["id"]: model["model"] for model in manifest["models"]},
        accepted,
        history,
        _shared_seen(config, accepted),
        directory,
        progress,
        threading.Lock(),
        manifest,
        [len(accepted)],
    )
    errors = []
    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futures = [
            pool.submit(_author_model, config, model_id, model_jobs, run)
            for model_id, model_jobs in pending.items()
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(error)
    if errors:
        raise errors[0]
