"""Generate and evaluate predictions with the same retrieval and response contract.

Model weights are loaded only for explicit inference; scoring accepts saved JSONL.
"""

from __future__ import annotations

import time
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
from tunelm.memory import Memory
from tunelm.planner import make_plan, pack_context
from tunelm.responses import render_prompt, restore_thinking
from tunelm.rewards import score
from tunelm.schemas import Plan, Task
from tunelm.training import ModelSpec, _base_model, _file_hash, load_tokenizer


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
    memory: Memory,
    tokenizer,
    *,
    plan: Plan | None = None,
    budget: int = 550,
    split: str = "benchmark",
) -> Task:
    plan = plan or make_plan(prompt)
    retrieved = memory.retrieve(plan)
    packed = pack_context(plan, retrieved, tokenizer, budget)
    return Task(
        id=content_hash([prompt, plan.model_dump(), memory.version])[:24],
        prompt=prompt,
        plan=plan,
        context=packed["context"],
        source_families=[],
        split=split,
        memory_version=memory.version,
        retrieved_ids=[m["stem"]["id"] for m in retrieved["matches"]],
    )


def predict(
    task: Task,
    model,
    tokenizer,
    *,
    max_new_tokens: int = 1536,
    temperature: float = 0.0,
    max_length: int = 4096,
) -> dict:
    import torch

    prompt = render_prompt(task, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    size = inputs["input_ids"].shape[1]
    if size + max_new_tokens > max_length:
        raise ValueError(
            "Prompt plus completion exceeds configured context; reduce input or output"
        )
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.inference_mode():
        tokens = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(tokens[0, size:], skip_special_tokens=False)
    text = text.split(tokenizer.eos_token)[0].rstrip()
    text = restore_thinking(text)
    scored = score(text, task.plan)
    return {"schema_version": 2, "id": task.id, "completion": text, **scored}


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
        scored = score(prediction["completion"] if prediction else "", task.plan)
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
    return metrics


def predict_bank(tasks_file: str, output: str, spec: ModelSpec, checkpoint: str | None) -> dict:
    """Generate a frozen task bank without rerunning retrieval or changing contexts."""
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
    model, tokenizer = load_checkpoint(spec, checkpoint)
    started = time.monotonic()
    for task in tasks:
        if task.id not in known:
            append_jsonl(output, predict(task, model, tokenizer, max_length=spec.max_length))
    return {"predictions": output, "seconds": round(time.monotonic() - started, 2)}
