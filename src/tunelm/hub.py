"""Upload one merged TuneLM checkpoint to Hugging Face at the end of a stage.

The trainer still saves a LoRA adapter locally. This module keeps the best adapter
aside, then writes the base weights with that adapter merged in. Modal and a local
app load the result as a normal causal LM. The token stays in the environment.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from transformers import TrainerCallback

from tunelm.io import project_path, read_json, write_json

_REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def validate_hub(config) -> None:
    """Reject a repo pattern that would not be a Hugging Face repo id."""
    if config.hub is None:
        return
    if "{which}" not in config.hub.repo or "{time}" not in config.hub.repo:
        raise ValueError("Hub repo pattern must include {which} and {time}")
    _repo_id(config, "best", "20000101-0000")
    _repo_id(config, "last", "20000101-0000")


def require_hub_token(config) -> None:
    """Fail before a long run when the upload is configured and no token is set."""
    if config.hub is not None and _token() is None:
        raise ValueError("Set HF_TOKEN to upload the merged checkpoint")


class BestAdapter(TrainerCallback):
    """Copy the adapter from the save with the best configured metric.

    `save_total_limit` deletes older trainer checkpoints. The copy is only the
    adapter, so the best one is still here when training ends.
    """

    def __init__(self, metric: str, greater_is_better: bool, dest: Path):
        self.metric = metric
        self.greater = greater_is_better
        self.dest = dest
        self.latest: float | None = None
        self.best: float | None = None
        self.step: int | None = None
        saved = dest / "best.json"
        if saved.is_file():
            record = read_json(saved)
            self.best = float(record["value"])
            self.step = int(record["step"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, control, kwargs
        if not logs or logs.get(self.metric) is None:
            return
        value = logs[self.metric]
        self.latest = float(value.item() if hasattr(value, "item") else value)

    def on_save(self, args, state, control, **kwargs):
        del control, kwargs
        if self.latest is None or not _improved(self.latest, self.best, self.greater):
            return
        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not (src / "adapter_config.json").is_file():
            return
        self.best = self.latest
        self.step = state.global_step
        _copy_adapter(src, self.dest)
        write_json(
            self.dest / "best.json",
            {"step": self.step, "metric": self.metric, "value": self.best},
        )


def publish_checkpoints(config, trainer, tokenizer) -> dict[str, str]:
    """Merge the final weights, and the best adapter when it differs, and upload both."""
    token = _token()
    if token is None:
        raise ValueError("Set HF_TOKEN to upload the merged checkpoint")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    dest = project_path(config.output_dir) / "hub-export"
    best = _best_record(project_path(config.output_dir) / "best-adapter")
    same = best is not None and best["step"] == trainer.state.global_step
    _export_merged(trainer.model, tokenizer, config.model, dest)
    trainer.model = None
    _release()
    uploaded = {
        "last": _upload(config, "last", stamp, dest, token, trainer.state.global_step, None)
    }
    if same:
        uploaded["best"] = _upload(config, "best", stamp, dest, token, best["step"], best)
    if best is None or same:
        shutil.rmtree(dest, ignore_errors=True)
        return uploaded
    shutil.rmtree(dest, ignore_errors=True)
    _export_merged(_load_adapter(config, best["path"]), tokenizer, config.model, dest)
    _release()
    uploaded["best"] = _upload(config, "best", stamp, dest, token, best["step"], best)
    shutil.rmtree(dest, ignore_errors=True)
    return uploaded


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _stage_name(config) -> str:
    if config.hub.stage:
        return config.hub.stage
    if config.stage == "sft" and config.curriculum == "warmup":
        return "warmup"
    return config.stage


def _repo_id(config, which: str, stamp: str) -> str:
    try:
        leaf = config.hub.repo.format(
            model=config.hub.model, stage=_stage_name(config), which=which, time=stamp
        )
    except KeyError as error:
        raise ValueError(f"Hub repo pattern has an unknown placeholder: {error}") from error
    repo = f"{config.hub.org}/{leaf}"
    if not _REPO.fullmatch(repo):
        raise ValueError(f"Hub repo id has characters Hugging Face rejects: {repo}")
    return repo


def _improved(value: float, best: float | None, greater: bool) -> bool:
    if best is None:
        return True
    return value > best if greater else value < best


def _copy_adapter(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in list(dest.iterdir()):
        if path.name.startswith("adapter_"):
            path.unlink()
    for path in src.iterdir():
        if path.name.startswith("adapter_"):
            shutil.copy2(path, dest / path.name)


def _best_record(dest: Path) -> dict | None:
    path = dest / "best.json"
    if not path.is_file() or not (dest / "adapter_config.json").is_file():
        return None
    record = read_json(path)
    record["path"] = dest
    return record


def _load_adapter(config, path: Path):
    from peft import PeftModel

    from tunelm.training import _base_model

    info = read_json(path / "adapter_config.json")
    base = _base_model(config.model, info.get("base_model_name_or_path"))
    return PeftModel.from_pretrained(base, str(path))


def _export_merged(model, tokenizer, spec, dest: Path) -> None:
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    model.save_pretrained(dest, safe_serialization=True)
    tokenizer.save_pretrained(dest)
    _copy_remote_code(spec, dest)


def _release() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _copy_remote_code(spec, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(spec.name, revision=spec.revision, allow_patterns=["*.py", "*.jinja"])
    )
    for path in root.iterdir():
        target = dest / path.name
        if path.is_file() and path.suffix in {".py", ".jinja"} and not target.exists():
            shutil.copy2(path, target)


def _upload(config, which, stamp, dest, token, step, best) -> str:
    from huggingface_hub import HfApi

    repo = _repo_id(config, which, stamp)
    (dest / "README.md").write_text(_card(config, which, step, best))
    api = HfApi(token=token)
    api.create_repo(repo, private=config.hub.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(dest),
        repo_id=repo,
        commit_message=f"Upload merged TuneLM {which} checkpoint",
    )
    return repo


def _card(config, which: str, step: int, best: dict | None) -> str:
    lines = [
        "---",
        f"base_model: {config.model.name}",
        "tags:",
        "- tunelm",
        "---",
        "",
        f"Merged full weights for TuneLM `{_stage_name(config)}` ({which}).",
        "The LoRA update is baked into the base weights. This repo is not an adapter.",
        "",
        f"Base: `{config.model.name}` `{config.model.revision}`",
        f"Step: {step}",
    ]
    if best is not None:
        lines.append(f"Selection: {best['metric']}={best['value']} at step {best['step']}.")
    return "\n".join(lines) + "\n"
