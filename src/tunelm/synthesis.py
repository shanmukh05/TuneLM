"""Ground teacher generation in verified seeds and compatible stem assemblies.

Accepted examples and rejected attempts use separate resumable JSONL journals.
No remote requests happen until the user explicitly runs the generate command.
"""

from __future__ import annotations

import math
import os
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tunelm.generation_state import (
    GenerationState,
    allocate_models,
    read_generation_snapshot,
    sample_task,
)
from tunelm.io import (
    append_jsonl,
    content_hash,
    generation_lock,
    project_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tunelm.memory import Memory
from tunelm.planner import pack_context
from tunelm.prompting import load_prompt_config, validate_prompt_style
from tunelm.responses import completion
from tunelm.rewards import quality_failures, score
from tunelm.schemas import Category, Example, Plan, Record, Stem, Task, Voice
from tunelm.teachers import DEFAULT_ENV, SYSTEM, Teacher, TeacherModel
from tunelm.teachers import TeacherReply as TeacherReply
from tunelm.telemetry import Telemetry
from tunelm.verifier import Verifier

PERSONAS = ("vibe", "producer", "algorithmic", "cocreate")


class GenerationConfig(Record):
    run_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    seeds_file: str = "datasets/seeds/compositions.jsonl"
    stems_file: str = "datasets/stems/verified.jsonl"
    memory_dir: str = "datasets/memory"
    dataset_dir: str = "datasets"
    count: int = Field(default=200, ge=1)
    seed: int = 42
    pathway: Literal["backtranslate", "recombine", "both"] = "both"
    context_tokens: int = Field(default=550, ge=64)
    max_candidates_per_sample: int = Field(default=5, ge=1, le=100)
    max_attempts: int = Field(default=3, ge=1, le=10)
    # Exact candidate coverage for inspection runs; failed jobs remain visible as shortfalls.
    task_counts: dict[Category, int] | None = None
    balanced_contexts: bool = False
    prompt_config: str | None = None
    append_from_run: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    models: list[TeacherModel] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_mix(self):
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("Generation model IDs must be unique")
        if not math.isclose(sum(m.fraction for m in self.models), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Generation model fractions must sum to 1.0")
        if self.task_counts is not None:
            if (
                any(n < 1 for n in self.task_counts.values())
                or sum(self.task_counts.values()) != self.count
            ):
                raise ValueError("Positive task_counts must sum to count")
            if self.pathway == "backtranslate":
                raise ValueError("Explicit task coverage requires recombine or both pathways")
        if self.append_from_run == self.run_id:
            raise ValueError("An append batch needs a different run_id from its parent")
        return self


def _assemble(stems: list[Stem], plan: Plan, verifier: Verifier) -> str:
    expressions = []
    for stem in stems:
        snippet = stem.setup + "\n" + stem.code
        if plan.key and "scale" in stem.methods and stem.code != plan.original_code:
            shifted = verifier.request(snippet, operation="transpose", key=plan.key)
            if not shifted["valid"]:
                raise ValueError(f"Unsupported transposition: {stem.id}")
            # Transpose setup and expression together without losing register.
            expressions.append(_scoped(shifted["setup"], shifted["expression"]))
            continue
        expressions.append(_scoped(stem.setup, stem.code))
    stack = "stack(" + ",\n".join(expressions) + ")"
    if plan.category == "atomic":
        stack = expressions[0]
    if plan.category == "arrange":
        intro = expressions[0]
        stack = f"arrange([4, {intro}], [8, {stack}], [4, {expressions[-1]}], [8, {stack}.fast(2)])"
    return f"setcpm({plan.cpm});\n{stack}"


def _scoped(setup: str, expression: str) -> str:
    return f"(()=>{{{setup}\nreturn ({expression});}})()" if setup else expression


def _backtranslation(seed, persona):
    roles = seed.get("roles") or [seed.get("role") or "fx_texture"]
    category = seed.get("category", "atomic")
    plan = Plan(
        bpm=(seed.get("cpm") or 30) * 4,
        key=seed.get("key"),
        category=category,
        short_loop=category == "atomic",
        cycles=32,
        voices=[Voice(role=r, query=seed["description"]) for r in roles],
    )
    return (
        seed["code"],
        plan,
        [seed["family"]],
        seed["split"],
        seed["id"],
        persona,
        "backtranslate",
    )


def _choose_stems(stems, split, rng, category=None):
    available = [s for s in stems if s.split == split]
    by_role = {s.role: [v for v in available if v.role == s.role] for s in available}
    if not by_role:
        return [], "atomic"
    category = (
        category or rng.choices(["atomic", "stack", "arrange", "cocreate"], [20, 45, 15, 20])[0]
    )
    roles = rng.sample(sorted(by_role), min(len(by_role), rng.randint(3, 5)))
    if category == "atomic":
        roles = [rng.choice(roles)]
    chosen = [rng.choice(by_role[r]) for r in roles]
    if category == "cocreate" and chosen[0].setup:
        category = "stack"
    return chosen, category


def _composition_plan(chosen, category, rng):
    import re

    key = rng.choice(["C:minor", "D:dorian", "F:major"])
    original = chosen[0].code if category == "cocreate" else None
    if original:
        scale = re.search(r"\.scale\([\"\']([A-G][#b]?)(?:\d+)?:([a-z]+)", original)
        key = scale[1] + ":" + scale[2] if scale else None
    if all(s.role in {"drums_core", "perc_tops", "fx_texture"} for s in chosen):
        key = None
    return Plan(
        bpm=rng.choice([84, 100, 120, 128, 140]),
        key=key,
        category=category,
        voices=[Voice(role=s.role, query=s.description) for s in chosen],
        cycles=32 if category == "arrange" else 16,
        original_code=original,
        explicit=["bpm", "key", "roles"],
        short_loop=category == "atomic",
    )


def _candidates(config, seeds, stems, verifier):
    if config.task_counts:
        yield from _balanced_candidates(config, seeds, stems, verifier)
        return
    rng = random.Random(config.seed)
    back_count = 0 if config.pathway == "recombine" else min(config.count, len(seeds) * 4)
    for i in range(back_count):
        yield _backtranslation(seeds[i // 4], PERSONAS[i % 4])
    if config.pathway == "backtranslate":
        return
    for i in range(config.count - back_count):
        split = rng.choice(seeds)["split"]
        chosen, category = _choose_stems(stems, split, rng)
        if not chosen:
            continue
        plan = _composition_plan(chosen, category, rng)
        try:
            code = _assemble(chosen, plan, verifier)
        except ValueError:
            continue
        if verifier.request(code, cycles=plan.cycles)["valid"]:
            yield (
                code,
                plan,
                sorted({s.family for s in chosen}),
                split,
                content_hash(code)[:20],
                PERSONAS[i % 4],
                "recombine",
            )


def _balanced_candidates(config, seeds, stems, verifier):
    """Produce a bounded, verified train-only bank covering the requested task types."""
    rng = random.Random(config.seed)
    scheduled = [category for category, count in config.task_counts.items() for _ in range(count)]
    atomic_seeds = [s for s in seeds if s["split"] == "train" and s.get("category") == "atomic"]
    for index in range(config.count):
        category = scheduled[index % len(scheduled)]
        persona = PERSONAS[index % len(PERSONAS)]
        if category == "atomic" and config.pathway == "both" and atomic_seeds:
            yield _backtranslation(rng.choice(atomic_seeds), persona)
            continue
        yield _verified_composition(stems, category, persona, rng, verifier)


def _verified_composition(stems, category, persona, rng, verifier):
    for _ in range(100):
        chosen, actual = _choose_stems(stems, "train", rng, category)
        if not chosen or actual != category:
            continue
        plan = _composition_plan(chosen, category, rng)
        try:
            code = _assemble(chosen, plan, verifier)
        except ValueError:
            continue
        if verifier.request(code, cycles=plan.cycles)["valid"]:
            return (
                code,
                plan,
                sorted({s.family for s in chosen}),
                "train",
                content_hash(code)[:20],
                persona,
                "recombine",
            )
    raise ValueError(f"Could not assemble a verified {category} candidate")


def _context(config, plan, families, memory, tokenizer, rng, regime=None):
    regime = regime or rng.choices(["high", "partial", "empty"], [40, 40, 20])[0]
    retrieval = memory.retrieve(plan, exclude_families=families if regime == "partial" else [])
    if regime == "empty":
        retrieval["matches"] = []
    if not retrieval["matches"]:
        regime = "empty"
    packed = pack_context(plan, retrieval, tokenizer, config.context_tokens)
    return packed, regime, [m["stem"]["id"] for m in retrieval["matches"]]


def _attempt(
    task,
    source_code,
    persona,
    pathway,
    seed_id,
    teacher,
    verifier,
    journal,
    history=None,
    job_index=None,
    max_attempts=3,
    prompt_brief=None,
):
    if history is None:
        history = []
    saved = next((r["accepted"] for r in history if r.get("accepted")), None)
    if saved:
        return Example.model_validate(saved)
    feedback = history[-1]["feedback"] if history else []
    for retry in range(len(history), max_attempts):
        reply = None
        accepted = None
        try:
            reply = teacher.generate(
                {
                    "verified_code": source_code,
                    "plan": task.plan.model_dump(),
                    "memory_context": task.context,
                    "persona": persona,
                    "pathway": pathway,
                    "prompt_brief": prompt_brief,
                },
                feedback,
            )
            validate_prompt_style(reply.prompt, prompt_brief)
            if pathway == "backtranslate" and reply.code.strip() != source_code.strip():
                raise ValueError("Back-translation changed the verified source code")
            summary = "\n".join(f"{i}. {value}" for i, value in enumerate(reply.decisions, 1))
            from tunelm.planner import make_plan

            observed = make_plan(reply.prompt)
            # Explicit prompt facts must match code, even on back-translated data.
            validation_plan = task.plan.model_copy(
                update={
                    "explicit": observed.explicit,
                    "bpm": observed.bpm if "bpm" in observed.explicit else task.plan.bpm,
                    "key": observed.key if "key" in observed.explicit else task.plan.key,
                    "voices": observed.voices if "roles" in observed.explicit else task.plan.voices,
                }
            )
            scored = score(completion(summary, reply.code), validation_plan, verifier)
            feedback = quality_failures(scored, validation_plan)
            if not feedback:
                accepted = Example(
                    **{
                        **task.model_dump(),
                        "prompt": reply.prompt,
                        "plan": validation_plan.model_dump(),
                    },
                    summary=summary,
                    code=reply.code,
                    teacher=teacher.identity,
                    teacher_id=teacher.id,
                    seed_id=seed_id,
                    pathway=pathway,
                    persona=persona,
                    verification=Telemetry.model_validate(scored["diagnostic"]["telemetry"]),
                    verifier_version=scored["diagnostic"]["runtime_version"],
                )
        except ValueError as error:
            feedback, scored = [str(error)], None
        entry = {
            "schema_version": 2,
            "task_id": task.id,
            "model_id": teacher.id,
            "job_index": job_index,
            "retry": retry,
            "reply": reply.model_dump() if reply else None,
            "feedback": feedback,
            "score": scored,
            "accepted": accepted.model_dump() if accepted else None,
            "usage": teacher.last_usage,
            "prompt_style": task.prompt_style,
            "prompt_scenario": task.prompt_scenario,
        }
        append_jsonl(journal, entry)
        history.append(entry)
        if accepted is not None:
            return accepted
    return None


def generation_preview(config: GenerationConfig) -> dict:
    """Inspect exact quotas and prerequisites without clients, weights, or state writes."""
    targets = allocate_models(config.models, config.count)
    models = [
        {
            "id": m.id,
            "provider": m.provider,
            "model": m.model or os.environ.get(m.model_env),
            "model_env": m.model_env,
            "fraction": m.fraction,
            "target": targets[m.id],
            "key_env": m.key_env or DEFAULT_ENV[m.provider][0],
        }
        for m in config.models
    ]
    missing = [
        str(path)
        for path in [
            config.seeds_file,
            config.stems_file,
            str(Path(config.memory_dir) / "metadata.json"),
        ]
        if not project_path(path).exists()
    ]
    prompts = load_prompt_config(config.prompt_config)
    parent = read_generation_snapshot(config.append_from_run)[0] if config.append_from_run else None
    return {
        "count": config.count,
        "candidate_task_counts": config.task_counts,
        "balanced_contexts": config.balanced_contexts,
        "max_teacher_attempts": config.count
        * config.max_candidates_per_sample
        * config.max_attempts,
        "models": models,
        "missing_inputs": missing,
        "api_requests": 0,
        "state_written": False,
        "prompt_style_schedule": prompts.schedule if prompts else None,
        "append_from": parent,
        "combined_target": config.count + (parent["accepted_count"] if parent else 0),
    }


def _resolve_models(config, saved, targets):
    previous = {m["id"]: m for m in saved.get("models", [])}
    resolved = []
    for model in config.models:
        name = (
            model.model
            or os.environ.get(model.model_env)
            or previous.get(model.id, {}).get("model")
        )
        name = name.strip() if name else None
        if targets[model.id] and not name:
            raise ValueError(f"Set models.{model.id}.model or {model.model_env}")
        resolved.append(
            {
                "id": model.id,
                "provider": model.provider,
                "model": name,
                "fraction": model.fraction,
                "target": targets[model.id],
                "candidate_budget": targets[model.id] * config.max_candidates_per_sample,
            }
        )
    return resolved


def _freeze_jobs(config, tokenizer, memory, seeds, stems, identity, budget):
    rng, verifier = random.Random(config.seed), Verifier()
    prompts = load_prompt_config(config.prompt_config)
    jobs, seen = [], set()
    candidate_config = config.model_copy(update={"count": budget})
    for code, plan, families, split, seed_id, persona, pathway in _candidates(
        candidate_config, seeds, stems, verifier
    ):
        requested = (
            ["high", "partial", "empty"][len(jobs) % 3] if config.balanced_contexts else None
        )
        packed, regime, ids = _context(config, plan, families, memory, tokenizer, rng, requested)
        brief = prompts.brief(len(jobs)) if prompts else None
        fingerprint = content_hash([code, persona, packed["context"], brief])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        task = Task(
            id=content_hash([identity, fingerprint])[:24],
            prompt="pending teacher prompt",
            plan=plan,
            context=packed["context"],
            source_families=families,
            split=split,
            memory_version=memory.version,
            retrieval_regime=regime,
            retrieved_ids=ids,
            prompt_style=brief["style"] if brief else None,
            prompt_scenario=brief["scenario"]["situation"] if brief else None,
        )
        jobs.append(
            {
                "task": task.model_dump(),
                "code": code,
                "seed_id": seed_id,
                "persona": persona,
                "pathway": pathway,
                "prompt_brief": brief,
            }
        )
    if config.task_counts and len(jobs) != budget:
        raise ValueError("Duplicate pilot candidates reduced coverage; choose another seed")
    return jobs


def _prepare_generation(config, tokenizer, directory):
    memory = Memory(config.memory_dir)
    seeds = read_jsonl(config.seeds_file)
    stems = [Stem.model_validate(r) for r in read_jsonl(config.stems_file)]
    if not seeds:
        raise ValueError("Ingest a licensed seed corpus first")
    path = project_path(directory / "run.json")
    saved = read_json(path) if path.exists() else {}
    targets = allocate_models(config.models, config.count)
    models = _resolve_models(config, saved, targets)
    prompts = load_prompt_config(config.prompt_config)
    parent = read_generation_snapshot(config.append_from_run)[0] if config.append_from_run else None
    tokenizer_hash = content_hash(
        {"vocabulary": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template}
    )
    identity = content_hash(
        {
            "config": config.model_dump(),
            "models": models,
            "tokenizer_hash": tokenizer_hash,
            "memory": memory.version,
            "seeds": seeds,
            "stems": [s.model_dump() for s in stems],
            "prompt_policy": {
                "system": SYSTEM,
                "config": prompts.model_dump() if prompts else None,
            },
            "parent": parent,
        }
    )
    _check_export(config.dataset_dir, identity, parent["identity"] if parent else None)
    manifest = {
        "schema_version": 2,
        "generation_version": 3,
        "identity": identity,
        "config": config.model_dump(),
        "models": models,
        "memory_version": memory.version,
        "tokenizer_hash": tokenizer_hash,
        "parent": parent,
    }
    budget = max(m["candidate_budget"] for m in models)
    return GenerationState.open(
        directory,
        manifest,
        lambda: _freeze_jobs(config, tokenizer, memory, seeds, stems, identity, budget),
    )


def _run_teacher(state, profile, teacher):
    verifier = Verifier()
    while state.pending(profile.id):
        index = state.cursors[profile.id]
        job = state.jobs[index]
        task = sample_task(job, profile.id)
        row = _attempt(
            task,
            job["code"],
            job["persona"],
            job["pathway"],
            job["seed_id"],
            teacher,
            verifier,
            state.directory / "attempts.jsonl",
            history=state.history[task.id],
            job_index=index,
            max_attempts=state.manifest["config"].get("max_attempts", 3),
            prompt_brief=job.get("prompt_brief"),
        )
        if row:
            state.record(row)
        state.cursors[profile.id] += 1
        state.checkpoint()


def _save_generation(config, state, status):
    parent, previous = (None, [])
    if config.append_from_run:
        parent, previous = read_generation_snapshot(config.append_from_run)
        if parent != state.manifest["parent"]:
            raise ValueError("Parent generation changed; refusing to replace combined exports")
    report = export_training(
        previous + [r.model_dump() for r in state.accepted.values()],
        state.manifest["identity"],
        config.dataset_dir,
        allowed_previous=parent["identity"] if parent else None,
    )
    progress = state.progress(status)
    report.update(
        {
            "status": status,
            "models": progress["models"],
            "target": config.count + len(previous),
            "new_accepted": len(state.accepted),
            "previous_accepted": len(previous),
            "parent": parent,
            "valid": status == "complete",
        }
    )
    write_json(Path(config.dataset_dir) / "manifests/training.json", report)
    state.checkpoint(status)
    return report


def generate(config: GenerationConfig, tokenizer) -> dict:
    """Resume frozen per-model jobs, preserving accepted quotas and retry progress."""
    directory = Path("results/generation") / config.run_id
    with ExitStack() as locks:
        locks.enter_context(generation_lock(directory))
        if config.append_from_run:
            locks.enter_context(
                generation_lock(Path("results/generation") / config.append_from_run)
            )
        state = _prepare_generation(config, tokenizer, directory)
        names = {m["id"]: m["model"] for m in state.manifest["models"]}
        try:
            with ExitStack() as stack:
                pending = [m for m in config.models if state.pending(m.id)]
                teachers = {
                    m.id: stack.enter_context(
                        Teacher(m.model_copy(update={"model": names[m.id]}), model_id=m.id)
                    )
                    for m in pending
                }
                for model in pending:
                    _run_teacher(state, model, teachers[model.id])
        except BaseException:
            _save_generation(config, state, "interrupted")
            raise
        complete = all(state.counts[name] == target for name, target in state.targets.items())
        return _save_generation(config, state, "complete" if complete else "exhausted")


def _check_export(directory: str, identity: str, allowed_previous: str | None = None) -> None:
    previous = project_path(Path(directory) / "manifests/training.json")
    if previous.exists() and read_json(previous)["identity"] not in {identity, allowed_previous}:
        raise ValueError("Training outputs belong to another run; choose a fresh dataset_dir")


def export_training(
    rows: list[dict],
    identity: str,
    directory: str = "datasets",
    *,
    allowed_previous: str | None = None,
) -> dict:
    """Write SFT targets and answer-free RL tasks from the accepted journal."""
    _check_export(directory, identity, allowed_previous)
    directory = Path(directory)
    examples = [Example.model_validate(r) for r in rows]
    for split in ["train", "val", "benchmark"]:
        subset = [r for r in examples if r.split == split]
        tasks = [
            Task.model_validate(
                {k: v for k, v in r.model_dump().items() if k in Task.model_fields}
            ).model_dump()
            for r in subset
        ]
        if split == "benchmark":
            write_jsonl(directory / "evaluation/benchmark.jsonl", tasks)
            continue
        # Keep inherited records unchanged while validating their schemas above.
        write_jsonl(directory / f"sft/{split}.jsonl", [r for r in rows if r["split"] == split])
        write_jsonl(directory / f"rl/{split}.jsonl", tasks)
    report = {
        "schema_version": 2,
        "identity": identity,
        "accepted": len(examples),
        "splits": {s: sum(r.split == s for r in examples) for s in ["train", "val", "benchmark"]},
    }
    write_json(directory / "manifests/training.json", report)
    return report
