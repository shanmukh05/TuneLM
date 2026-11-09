"""Exact model quotas and crash-recoverable JSONL generation state.

Frozen jobs prevent retrieval or candidate changes on resume. Attempt journals
are authoritative; progress summaries and dataset exports are rebuildable.
Append batches inherit hash-checked accepted snapshots without changing old journals.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from tunelm.io import (
    append_jsonl,
    content_hash,
    project_path,
    read_json,
    read_jsonl,
    recover_journal,
    write_json,
    write_jsonl,
)
from tunelm.schemas import Example, Task
from tunelm.teachers import TeacherModel


def allocate_models(models: list[TeacherModel], count: int) -> dict[str, int]:
    """Use largest remainders, with config order breaking ties, to total count exactly."""
    fractions = [Decimal(str(model.fraction)) for model in models]
    total = sum(fractions)
    raw = [Decimal(count) * fraction / total for fraction in fractions]
    counts = [int(value) for value in raw]
    ranking = sorted(range(len(models)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in ranking[: count - sum(counts)]:
        counts[i] += 1
    return {model.id: quota for model, quota in zip(models, counts, strict=True)}


def sample_task(job: dict, model_id: str) -> Task:
    task = Task.model_validate(job["task"])
    return task.model_copy(update={"id": content_hash([task.id, model_id])[:24]})


def read_generation_snapshot(
    run_id: str, ancestors: tuple[str, ...] = ()
) -> tuple[dict, list[dict]]:
    """Read accepted rows plus validated ancestors without altering their journals."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id) or run_id in ancestors:
        raise ValueError("Invalid or circular append_from_run lineage")
    directory = Path("results/generation") / run_id
    manifest = read_json(directory / "run.json")
    inherited = []
    if manifest.get("parent"):
        snapshot, inherited = read_generation_snapshot(
            manifest["parent"]["run_id"], (*ancestors, run_id)
        )
        if snapshot != manifest["parent"]:
            raise ValueError("Parent generation changed after this batch was frozen")
    path = project_path(directory / "accepted.jsonl")
    rows = inherited + (read_jsonl(path) if path.exists() else [])
    for row in rows:
        Example.model_validate(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs in appended generation lineage")
    return {
        "run_id": run_id,
        "identity": manifest["identity"],
        "accepted_hash": content_hash(rows),
        "accepted_count": len(rows),
    }, rows


class GenerationState:
    """One locked run's frozen jobs, accepted records, and per-model cursors."""

    def __init__(self, directory: Path, manifest: dict, jobs: list[dict]):
        self.directory = directory
        self.manifest = manifest
        self.jobs = jobs
        self.targets = {m["id"]: m["target"] for m in manifest["models"]}
        self.budgets = {m["id"]: min(m["candidate_budget"], len(jobs)) for m in manifest["models"]}
        self.history = defaultdict(list)
        self.accepted = {}
        self.counts = Counter()
        self.cursors = {name: 0 for name in self.targets}
        self._restore()

    @classmethod
    def open(cls, directory: Path, manifest: dict, make_jobs):
        path = project_path(directory / "run.json")
        if path.exists():
            saved = read_json(path)
            if saved.get("generation_version") != 3 or saved["identity"] != manifest["identity"]:
                raise ValueError("Generation inputs/model mix changed; choose a new run_id")
            jobs = read_jsonl(directory / "jobs.jsonl")
            if content_hash(jobs) != saved["jobs_hash"]:
                raise ValueError("Frozen generation jobs were modified or truncated")
            return cls(directory, saved, jobs)
        jobs = make_jobs()
        if not jobs:
            raise ValueError("No verified generation candidates are available")
        manifest = {**manifest, "jobs_hash": content_hash(jobs)}
        write_jsonl(directory / "jobs.jsonl", jobs)
        write_json(directory / "run.json", manifest)
        return cls(directory, manifest, jobs)

    def _restore(self):
        self._load_journals()
        self._recover_completed()
        for name in self.targets:
            self._restore_cursor(name)

    def _load_journals(self):
        for entry in recover_journal(self.directory / "attempts.jsonl"):
            if entry.get("model_id") not in self.targets:
                raise ValueError("Attempt journal contains an unknown model profile")
            self.history[entry["task_id"]].append(entry)
        for value in recover_journal(self.directory / "accepted.jsonl"):
            row = Example.model_validate(value)
            if row.id in self.accepted:
                raise ValueError("Duplicate accepted record in generation journal")
            self._index(row)

    def _recover_completed(self):
        # A completed attempt may have reached disk immediately before interruption.
        for entries in self.history.values():
            for entry in entries:
                if entry.get("accepted") and entry["task_id"] not in self.accepted:
                    self.record(Example.model_validate(entry["accepted"]))

    def _index(self, row: Example):
        if row.teacher_id not in self.targets:
            raise ValueError("Accepted record has an unknown teacher_id")
        if self.counts[row.teacher_id] >= self.targets[row.teacher_id]:
            raise ValueError("Accepted journal exceeds the configured model quota")
        self.accepted[row.id] = row
        self.counts[row.teacher_id] += 1

    def _restore_cursor(self, name: str):
        while self.cursors[name] < self.budgets[name]:
            task = sample_task(self.jobs[self.cursors[name]], name)
            entries = self.history[task.id]
            if task.id not in self.accepted and len(entries) < self.manifest["config"].get(
                "max_attempts", 3
            ):
                break
            self.cursors[name] += 1

    def record(self, row: Example):
        if row.id in self.accepted:
            return
        self._index(row)
        append_jsonl(self.directory / "accepted.jsonl", row.model_dump())

    def pending(self, model_id: str) -> bool:
        return (
            self.counts[model_id] < self.targets[model_id]
            and self.cursors[model_id] < self.budgets[model_id]
        )

    def progress(self, status: str) -> dict:
        models = {
            name: {
                "target": quota,
                "accepted": self.counts[name],
                "next_job": self.cursors[name],
                "candidate_budget": self.budgets[name],
                "shortfall": quota - self.counts[name],
            }
            for name, quota in self.targets.items()
        }
        return {
            "schema_version": 2,
            "status": status,
            "models": models,
            "accepted": len(self.accepted),
            "target": sum(self.targets.values()),
        }

    def checkpoint(self, status: str = "running"):
        write_json(self.directory / "progress.json", self.progress(status))
