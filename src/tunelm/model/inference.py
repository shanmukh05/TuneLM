"""Generate and evaluate predictions with the same retrieval and response contract.

Model weights are loaded only for explicit inference; scoring accepts saved JSONL.
Each prediction bank also writes viewer.jsonl, a training-shaped record the
dataset viewer can open while a benchmark is still running.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from tunelm.composition.memory import Memory
from tunelm.composition.planner import make_plan, pack_context
from tunelm.composition.responses import generate_completion, parse_completion, render_prompt
from tunelm.core.io import (
    append_jsonl,
    content_hash,
    project_path,
    read_json,
    read_jsonl,
    recover_journal,
    write_json,
    write_jsonl,
)
from tunelm.core.schemas import Plan, Task
from tunelm.model.rewards import contract_score, score
from tunelm.model.training import ModelSpec, _base_model, _file_hash, load_tokenizer


def load_checkpoint(spec: ModelSpec, checkpoint: str | None = None):
    """Reload base or PEFT checkpoints, including the GRPO adapter's merged SFT base."""
    tokenizer = load_tokenizer(spec)
    if checkpoint:
        from peft import PeftConfig, PeftModel

        path = project_path(checkpoint)
        adapter = PeftConfig.from_pretrained(str(path))
        base = adapter.base_model_name_or_path
        if Path(base).is_absolute() and not Path(base).exists():
            manifest = read_json(path / "run.json")
            base = str(project_path(manifest["config"]["merged_sft"]))
        if not project_path(base).exists() and base != spec.name:
            raise ValueError("Adapter base is missing or belongs to another model")
        model = PeftModel.from_pretrained(_base_model(spec, base), str(path))
    else:
        model = _base_model(spec)
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if device == "cpu":
        model = model.float()
    model.to(device).eval()
    return model, tokenizer


def make_task(
    prompt: str,
    memory: Memory | None,
    tokenizer,
    *,
    plan: Plan | None = None,
    budget: int = 6400,
    split: str = "benchmark",
) -> Task:
    plan = plan or make_plan(prompt)
    retrieved = (
        memory.retrieve(plan)
        if memory
        else {"matches": [], "missing_roles": [], "version": "api-guide"}
    )
    packed = pack_context(plan, retrieved, tokenizer, budget)
    return Task(
        id=content_hash([prompt, plan.model_dump(), retrieved["version"]])[:24],
        prompt=prompt,
        plan=plan,
        context=packed["context"],
        source_families=[],
        split=split,
        memory_version=retrieved["version"],
        retrieval_regime="partial" if retrieved["matches"] else "empty",
        retrieved_ids=[m["stem"]["id"] for m in retrieved["matches"]],
    )


def predict(
    task: Task,
    model,
    tokenizer,
    *,
    max_new_tokens: int = 3072,
    temperature: float = 0.0,
    max_length: int = 6144,
) -> dict:
    text, _ = generate_completion(
        render_prompt(task, tokenizer),
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_length=max_length,
    )
    return {"schema_version": 2, "id": task.id, "completion": text, **_score(text, task)}


def _score(text: str, task: Task) -> dict:
    """RL-bank tasks use the GRPO contract reward; SFT-derived tasks keep the plan gate."""
    if task.contract:
        return contract_score(text, task.contract, task.plan)
    return score(text, task.plan)


def evaluate(tasks_file: str, predictions_file: str, directory: str) -> dict:
    """Score saved predictions and report missing/extra IDs instead of hiding them."""
    tasks = [Task.model_validate(r) for r in read_jsonl(tasks_file)]
    predictions = read_jsonl(predictions_file)
    ids = [r["id"] for r in predictions]
    if len(set(ids)) != len(ids) or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task or prediction IDs")
    lookup = {r["id"]: r for r in predictions}
    records = []
    for task in tasks:
        prediction = lookup.get(task.id)
        scored = _score(prediction["completion"] if prediction else "", task)
        records.append(
            {
                "schema_version": 2,
                "id": task.id,
                "category": task.plan.category,
                "retrieval_regime": task.retrieval_regime,
                "missing_prediction": prediction is None,
                **scored,
            }
        )
    if not records:
        raise ValueError("Benchmark has no tasks")
    metrics = {
        "schema_version": 2,
        "tasks": len(tasks),
        "missing": sum(r["missing_prediction"] for r in records),
        "extra_prediction_ids": sorted(set(lookup) - {t.id for t in tasks}),
        "execution_rate": sum(r["valid"] for r in records) / len(records),
        "mean_reward": sum(r["total"] for r in records) / len(records),
        "listening_review": "pending",
    }
    write_jsonl(Path(directory) / "scores.jsonl", records)
    write_json(Path(directory) / "summary.json", metrics)
    metrics["viewer"] = export_viewer(
        tasks_file, predictions_file, _teacher_from_manifest(predictions_file)
    )
    return metrics


def predict_bank(
    tasks_file: str,
    output: str,
    spec: ModelSpec,
    checkpoint: str | None,
    *,
    max_new_tokens: int = 3072,
) -> dict:
    """Generate a frozen task bank without rerunning retrieval or changing contexts.

    Completed task ids are kept, so a stopped benchmark resumes. Each saved
    completion is also written to viewer.jsonl beside the predictions, with the
    task prompt and the parsed summary and code. Progress goes to stderr and
    the caller's JSON result stays on stdout.
    """
    tasks = [Task.model_validate(r) for r in read_jsonl(tasks_file)]
    manifest_path = output + ".manifest.json"
    identity = content_hash(
        {
            "tasks": [t.model_dump() for t in tasks],
            "model": spec.model_dump(),
            "checkpoint": checkpoint,
            "adapter_hash": _file_hash(project_path(checkpoint) / "adapter_model.safetensors")
            if checkpoint
            else None,
        }
    )
    if project_path(manifest_path).exists():
        if read_json(manifest_path)["identity"] != identity:
            raise ValueError("Predictions belong to another model/task bank; choose a fresh output")
    elif project_path(output).exists():
        raise ValueError("Existing prediction file has no provenance manifest")
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "identity": identity,
            "model": spec.model_dump(),
            "checkpoint": checkpoint,
        },
    )
    known = {r["id"] for r in recover_journal(output)}
    pending = [task for task in tasks if task.id not in known]
    teacher = _teacher_name(checkpoint, spec)
    export_viewer(tasks_file, output, teacher)
    started = time.monotonic()
    if not pending:
        return {"predictions": output, "viewer": _viewer_path(output), "generated": 0, "seconds": 0}
    model, tokenizer = load_checkpoint(spec, checkpoint)
    for index, task in enumerate(pending, 1):
        print(f"benchmark {index}/{len(pending)} {task.id}", file=sys.stderr, flush=True)
        append_jsonl(
            output,
            predict(
                task,
                model,
                tokenizer,
                max_new_tokens=max_new_tokens,
                max_length=spec.max_length,
            ),
        )
        export_viewer(tasks_file, output, teacher)
    return {
        "predictions": output,
        "viewer": _viewer_path(output),
        "generated": len(pending),
        "seconds": round(time.monotonic() - started, 2),
    }


def export_viewer(tasks_file: str, predictions_file: str, teacher: str) -> str:
    """Rewrite viewer.jsonl from the task bank and every saved completion.

    The row keeps the task prompt and plan, and adds the summary and Strudel
    code the dataset viewer plays. A completion that does not parse still keeps
    its text in the summary.
    """
    tasks = {row["id"]: row for row in read_jsonl(tasks_file)}
    rows = []
    for prediction in recover_journal(predictions_file):
        task = tasks.get(prediction.get("id"))
        if task is None:
            raise ValueError(f"Prediction {prediction.get('id')} is not in the task bank")
        rows.append(_viewer_row(task, prediction, teacher))
    dest = _viewer_path(predictions_file)
    write_jsonl(dest, rows)
    return dest


def _viewer_path(predictions_file: str) -> str:
    return str(Path(predictions_file).with_name("viewer.jsonl"))


def _teacher_name(checkpoint: str | None, spec: ModelSpec | None = None) -> str:
    if checkpoint:
        return f"tunelm:{Path(checkpoint).name}"
    if spec is not None:
        return f"tunelm:{spec.name}"
    return "tunelm:benchmark"


def _teacher_from_manifest(predictions_file: str) -> str:
    manifest = project_path(predictions_file + ".manifest.json")
    if not manifest.is_file():
        return _teacher_name(None)
    return _teacher_name(read_json(manifest).get("checkpoint"))


def _viewer_row(task: dict, prediction: dict, teacher: str) -> dict:
    record = dict(task)
    record["teacher"] = teacher
    for key in ("valid", "total", "components", "verification"):
        source = "diagnostic" if key == "verification" else key
        if prediction.get(source) is not None:
            record[key] = prediction[source]
    try:
        summary, code = parse_completion(prediction.get("completion", ""))
    except ValueError:
        record["summary"] = prediction.get("completion", "")
        record["code"] = ""
        return record
    record["summary"] = summary
    record["code"] = code
    return record
