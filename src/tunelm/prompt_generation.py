"""Author independent SFT and RL input banks before any music answers exist.

Frozen briefs, exact author quotas, accepted requests, usage, and rejected attempts
are journaled separately from composition generation. Both stages resume by job ID.
"""

import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
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
from tunelm.prompting import PromptConfig, validate_prompt_style
from tunelm.schemas import Category, MusicRequest, Record, Split
from tunelm.teachers import Teacher, TeacherModel
from tunelm.verifier import Verifier

PROMPT_SYSTEM = """Write ONE realistic end-user request for instrumental music.
Return JSON with schema_version=2, prompt, and original_code (null normally).
This is the input-authoring stage. Do NOT write the answer, an implementation plan,
musical decisions for the composer, or a finished composition. You have no source
composition to describe. The brief assigns a writing style, a category, and, when present, an opening and a
focus. Follow that opening and that focus. They are the variation for this request.
Invent the situation yourself. No situation list is provided. Choose a fresh reason
for the music on every request. The reason can be a few words.
When already_written is present, those requests are already accepted. Do not reuse
their situation or their musical ask.
Some requests are only a few words. Others are a few sentences. Do not make them
all long. Vary the emotion too: joy, tenderness, tension, play, grief, calm,
unease, anger, warmth, excitement. Do not make the bank mostly cozy or nostalgic.
Do not recycle a scene you have used before, and do not fall back on a stock setup.
The listener may be vague or specify several preferences. Leave genuine choices
open; do not add a BPM/key/instrument inventory just to aid validation.
Follow the assigned request style and category. Arrangement requests should ask
for development in natural language. Atomic requests ask for one part. Only for
co-creation, provide a short original Strudel starting pattern as original_code,
and ask to develop it; that starting pattern is input material, not the answer.
No real artist imitation, promised vocals, marketing copy, or fabricated API names.
"""


class PromptReply(Record):
    prompt: str = Field(min_length=1)
    original_code: str | None = None


class PromptGenerationConfig(RunOptions):
    """Counts describe independent inputs, not accepted music answers or rollouts."""

    run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    output_dir: str
    palette: PromptConfig
    counts: dict[Literal["sft", "rl"], dict[Split, int]]
    categories: list[Category] = Field(default_factory=lambda: ["stack", "arrange"])
    max_attempts: int = Field(default=2, ge=1, le=10)
    avoid_sample: int = Field(default=40, ge=0)
    models: list[TeacherModel] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_counts(self):
        counts = [n for splits in self.counts.values() for n in splits.values()]
        if not counts or min(counts) < 0 or sum(counts) < 1 or not self.categories:
            raise ValueError("Prompt counts must be nonnegative and total at least one")
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("Author model IDs must be unique")
        if abs(sum(m.fraction for m in self.models) - 1) > 1e-9:
            raise ValueError("Author model fractions must sum to 1")
        return self

    @property
    def count(self) -> int:
        return sum(n for splits in self.counts.values() for n in splits.values())


def load_prompt_generation_config(path: str) -> PromptGenerationConfig:
    """Load a prompt run, inheriting one parent file when `extends` is set."""
    return PromptGenerationConfig.model_validate(_read_prompt_config(path, []))


def _read_prompt_config(path: str, chain: list[str]) -> dict:
    if path in chain:
        raise ValueError(f"Prompt config extends itself: {path}")
    raw = dict(load_config(path))
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if chain:
        raise ValueError("A prompt config can extend only one file")
    if not isinstance(parent, str) or not parent.endswith((".yaml", ".yml")):
        raise ValueError("extends must be a YAML config path")
    return {**_read_prompt_config(parent, chain + [path]), **raw}


def prompt_preview(config: PromptGenerationConfig) -> dict:
    """Show author quotas and stage counts without clients, downloads, or writes."""
    return {
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
    for stage, splits in config.counts.items():
        for split, count in splits.items():
            for _ in range(count):
                index = len(jobs)
                job = {
                    "stage": stage,
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
    identity = content_hash([config.identity_config(), models, PROMPT_SYSTEM])
    manifest = {"identity": identity, "config": config.model_dump(), "models": models}
    if saved:
        jobs = read_jsonl(directory / "jobs.jsonl")
        if saved["jobs_hash"] != content_hash(jobs):
            raise ValueError("Prompt jobs changed; use a new prompt run_id and output_dir")
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


def _request_row(reply, job, teacher, seen):
    validate_prompt_style(reply.prompt, job["brief"])
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
    )
    if row.original_code:
        result = Verifier().request(row.original_code)
        if not result["valid"]:
            raise ValueError(f"Co-creation starting pattern failed: {result.get('error', result)}")
    return row.model_dump()


def _author_job(config, job, teacher, entries, seen, directory, progress):
    feedback = entries[-1]["feedback"] if entries else []
    for retry in range(len(entries), config.max_attempts):
        reply, accepted = None, None
        progress.attempt(teacher.id, retry, config.max_attempts)
        try:
            payload = {
                "brief": job["brief"],
                "category": job["category"],
                "stage": job["stage"],
            }
            shown = _avoid_sample(seen, config.avoid_sample)
            if shown:
                payload["already_written"] = shown
            reply = teacher.generate(payload, feedback)
            accepted = _request_row(reply, job, teacher, seen)
            feedback = []
        except ValueError as error:
            feedback = [str(error)]
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
        report = {
            "identity": manifest["identity"],
            "output_dir": config.output_dir,
            "status": status,
            "accepted": len(rows),
            "target": config.count,
            "valid": len(accepted) == config.count,
            "stages": dict(Counter(r["stage"] for r in rows)),
            "records_hash": content_hash(rows),
            **config.runtime_settings(),
        }
        write_json(Path(config.output_dir) / "manifest.json", report)
        return report


def generate_prompts(config: PromptGenerationConfig) -> dict:
    """Generate input-only banks; resume accepted/rejected attempts without authoring answers."""
    directory = Path("results/prompt-generation") / config.run_id
    with generation_lock(directory):
        manifest, jobs = _open_prompt_run(config, directory)
        jobs = _claim_saved_prompts(config, manifest, jobs, directory)
        accepted, history = _restore_prompts(directory)
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


def _run_authors(config, manifest, jobs, accepted, history, directory, progress):
    names = {m["id"]: m["model"] for m in manifest["models"]}
    with ExitStack() as stack:
        teachers = {}
        last_saved = len(accepted)
        for job in jobs:
            if job["id"] in accepted or len(history[job["id"]]) >= config.max_attempts:
                continue
            profile = next(m for m in config.models if m.id == job["model_id"])
            if profile.id not in teachers:
                teachers[profile.id] = stack.enter_context(
                    Teacher(
                        profile.model_copy(update={"model": names[profile.id]}),
                        model_id=profile.id,
                        system=PROMPT_SYSTEM,
                        response_model=PromptReply,
                    )
                )
            seen = _shared_seen(config, accepted)
            row = _author_job(
                config, job, teachers[profile.id], history[job["id"]], seen, directory, progress
            )
            if row:
                accepted[row["id"]] = row
                _remember_prompt(seen, row["prompt"])
            progress.advance(len(accepted))
            _prompt_checkpoint(config, directory, progress, "running")
            if len(accepted) - last_saved >= config.save_every:
                _export_prompt_bank(config, manifest, accepted, "running")
                last_saved = len(accepted)
