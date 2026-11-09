"""Required SFT and GRPO entry points, checkpoint handoff, and dry-run validation.

Heavy model imports stay inside entry points; dry runs never load model weights.
GRPO uses a saved merged SFT base so fresh RL adapters reload correctly.
"""

from __future__ import annotations

import os
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import Field

from tunelm.io import (
    append_jsonl,
    content_hash,
    output_path,
    project_path,
    read_json,
    read_jsonl,
    write_json,
)
from tunelm.responses import render_prompt, restore_thinking, tokenize_example
from tunelm.rewards import score
from tunelm.schemas import Example, Record, Task
from tunelm.verifier import Verifier


class ModelSpec(Record):
    name: str = "LiquidAI/LFM3-1.7B-Base"
    revision: str = Field(
        default="7bca5038823c0d0f14efe06439109354a10e014c", pattern=r"^[a-f0-9]{40}$"
    )
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    max_length: int = Field(default=4096, ge=128)
    trust_remote_code: bool = True
    attention: str = "sdpa"


class TrainConfig(Record):
    stage: Literal["sft", "grpo"]
    model: ModelSpec = Field(default_factory=ModelSpec)
    train_file: str
    val_file: str | None = None
    output_dir: str
    results_dir: str
    memory_dir: str = "datasets/memory"
    curriculum: Literal["warmup", "full"] = "full"
    adapter: str | None = None
    sft_checkpoint: str | None = None
    merged_sft: str | None = None
    resume: str | bool = False
    training: dict = Field(default_factory=dict)
    lora: dict = Field(
        default_factory=lambda: {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj",
                "out_proj",
                "w1",
                "w2",
                "w3",
            ],
        }
    )


def load_tokenizer(spec: ModelSpec, *, offline: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.name,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=offline,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _base_model(spec: ModelSpec, name: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    source = name or spec.name
    local = project_path(source).exists()
    return AutoModelForCausalLM.from_pretrained(
        str(project_path(source)) if local else source,
        revision=None if local else spec.revision,
        code_revision=None if local else spec.revision,
        trust_remote_code=spec.trust_remote_code,
        dtype=getattr(torch, spec.dtype),
        attn_implementation=spec.attention,
    )


def _rows(config: TrainConfig):
    cls = Example if config.stage == "sft" else Task
    train = [cls.model_validate(row) for row in read_jsonl(config.train_file)]
    val = (
        [cls.model_validate(row) for row in read_jsonl(config.val_file)] if config.val_file else []
    )
    if config.stage == "sft" and config.curriculum == "warmup":
        train = [
            r
            for r in train
            if r.plan.category == "atomic"
            or (r.plan.category == "stack" and r.retrieval_regime == "high")
        ]
    if not train or any(r.split != "train" for r in train) or any(r.split != "val" for r in val):
        raise ValueError("Training/validation rows are empty or assigned to the wrong split")
    train_families = {f for r in train for f in r.source_families}
    if train_families & {f for r in val for f in r.source_families}:
        raise ValueError("Source-family leakage between training and validation")
    if len({r.id for r in train + val}) != len(train + val):
        raise ValueError("Duplicate training/validation IDs")
    return train, val


def preflight(config: TrainConfig, *, tokenize: bool = False, offline: bool = False) -> dict:
    """Validate data/config/handoff without weights, optimizer steps, or paid calls."""
    from trl import GRPOConfig, SFTConfig

    native = SFTConfig if config.stage == "sft" else GRPOConfig
    unknown = set(config.training) - {f.name for f in fields(native)}
    reserved = {"hub_token", "output_dir"}
    if config.stage == "sft":
        reserved |= {
            "max_length",
            "packing",
            "completion_only_loss",
            "dataset_kwargs",
            "assistant_only_loss",
        }
    invalid = unknown | (reserved & set(config.training))
    if invalid:
        raise ValueError(f"Unknown or reserved training fields: {sorted(invalid)}")
    if config.training.get("push_to_hub"):
        raise ValueError("Automatic publication is not part of this pipeline")
    train, val = _rows(config)
    if config.training.get("eval_strategy", "no") != "no" and not val:
        raise ValueError("Configured evaluation requires a nonempty validation split")
    if config.adapter:
        _validate_adapter(config.adapter, config.model)
    output_path(config.output_dir + "/run.json")
    output_path(config.results_dir + "/run.json")
    from tunelm.memory import Memory

    memory = Memory(config.memory_dir)
    if any(row.memory_version != memory.version for row in train + val):
        raise ValueError("Training data does not match the frozen memory snapshot")
    if config.stage == "grpo":
        _validate_handoff(config)
    report = {
        "stage": config.stage,
        "train_rows": len(train),
        "val_rows": len(val),
        "model": config.model.model_dump(),
        "weights_loaded": False,
        "memory_version": memory.version,
        "data_hash": content_hash([r.model_dump() for r in train + val]),
        "libraries": {m: version(m) for m in ["transformers", "trl", "peft", "datasets"]},
    }
    _guard_run(config, report)
    if tokenize:
        tokenizer = load_tokenizer(config.model, offline=offline)
        _check_lengths(config, train + val, tokenizer)
        report["tokenizer_checked"] = len(train + val)
    return report


def _validate_adapter(path: str, spec: ModelSpec) -> None:
    adapter = project_path(path) / "adapter_config.json"
    if not adapter.is_file():
        raise ValueError("Adapter checkpoint is missing; finish the preceding training stage")
    if read_json(adapter).get("base_model_name_or_path") != spec.name:
        raise ValueError("Adapter belongs to a different base model")


def _guard_run(config: TrainConfig, report: dict) -> None:
    settings = config.model_dump(exclude={"resume"})
    identity = content_hash(
        {"config": settings, "data": report["data_hash"], "memory": report["memory_version"]}
    )
    report["identity"] = identity
    previous = project_path(Path(config.output_dir) / "run.json")
    if not previous.exists():
        return
    if not config.resume:
        raise ValueError("Training output already exists; use --resume or a fresh output_dir")
    if read_json(previous).get("identity") != identity:
        raise ValueError("Resume configuration, data, or memory differs from the original run")


def _validate_handoff(config: TrainConfig) -> None:
    if not config.sft_checkpoint or not config.merged_sft:
        raise ValueError("GRPO requires sft_checkpoint and merged_sft paths")
    sft = project_path(config.sft_checkpoint)
    _validate_adapter(config.sft_checkpoint, config.model)
    output_path(Path(config.merged_sft) / "handoff.json")
    if len({project_path(config.output_dir), sft, project_path(config.merged_sft)}) != 3:
        raise ValueError("SFT, merged base, and GRPO output paths must differ")


def _check_lengths(config, rows, tokenizer):
    if config.stage == "sft":
        for row in rows:
            tokenize_example(row, tokenizer, config.model.max_length)
        return
    completion_limit = config.training.get("max_completion_length", 1536)
    for row in rows:
        size = len(tokenizer.encode(render_prompt(row, tokenizer), add_special_tokens=False))
        if size + completion_limit > config.model.max_length:
            raise ValueError(f"RL prompt plus completion exceeds context limit: {row.id}")


def _lora(config: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(task_type="CAUSAL_LM", **config.lora)


def _require_gpu():
    import torch

    # ponytail: one CUDA process avoids concurrent merge writes; add rank barriers for DDP.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The initial runner supports one CUDA process; do not use DDP")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; use --dry-run locally")


def _save_run(config, trainer, tokenizer, report):
    directory = output_path(Path(config.output_dir) / "run.json").parent
    trainer.save_model(str(directory))
    trainer.save_state()
    tokenizer.save_pretrained(directory)
    write_json(
        directory / "run.json",
        {**report, "weights_loaded": True, "status": "complete", "config": config.model_dump()},
    )
    write_json(Path(config.results_dir) / "metrics.json", trainer.state.log_history)


def train_sft(config: TrainConfig) -> dict:
    """Run completion-masked SFT, optionally continuing the warmup adapter."""
    _require_gpu()
    report = preflight(config, tokenize=True)
    from peft import PeftModel, get_peft_model
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset

    tokenizer = load_tokenizer(config.model)
    train, val = _rows(config)
    model = _base_model(config.model)
    model.config.use_cache = False
    model = (
        PeftModel.from_pretrained(model, str(project_path(config.adapter)), is_trainable=True)
        if config.adapter
        else get_peft_model(model, _lora(config))
    )
    args = SFTConfig(
        output_dir=str(project_path(config.output_dir)),
        max_length=config.model.max_length,
        packing=False,
        completion_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        **config.training,
    )

    def dataset(rows):
        return Dataset.from_list(
            [tokenize_example(r, tokenizer, config.model.max_length) for r in rows]
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset(train),
        eval_dataset=dataset(val) if val else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    write_json(
        Path(config.output_dir) / "run.json",
        {**report, "weights_loaded": True, "status": "running", "config": config.model_dump()},
    )
    trainer.train(resume_from_checkpoint=config.resume or None)
    _save_run(config, trainer, tokenizer, report)
    return {"checkpoint": config.output_dir, "stage": "sft"}


def _merged_sft(config: TrainConfig) -> str:
    from peft import PeftModel

    dest = output_path(Path(config.merged_sft) / "handoff.json").parent
    source = project_path(config.sft_checkpoint)
    identity = content_hash(
        {
            "adapter": (source / "adapter_config.json").read_text(),
            "weights_hash": _file_hash(source / "adapter_model.safetensors"),
            "model": config.model.model_dump(),
        }
    )
    if (dest / "handoff.json").exists():
        if read_json(dest / "handoff.json")["identity"] != identity:
            raise ValueError("Merged SFT base belongs to a different adapter; choose a fresh path")
        return str(dest)
    model = PeftModel.from_pretrained(_base_model(config.model), str(source)).merge_and_unload()
    model.save_pretrained(dest, safe_serialization=True)
    load_tokenizer(config.model).save_pretrained(dest)
    write_json(dest / "handoff.json", {"identity": identity, "source": str(source)})
    del model
    import torch

    torch.cuda.empty_cache()
    return str(dest)


def _file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MusicalReward:
    """TRL callable retaining component rewards and diagnostics per worker process."""

    __name__ = "strudel_musical_reward"

    def __init__(self, results_dir: str):
        rank = os.environ.get("RANK", "0")
        self.path = Path(results_dir) / f"rollouts-rank{rank}-{os.getpid()}.jsonl"
        self.verifier = Verifier()

    def __call__(self, prompts, completions, plan_json, task_id, **kwargs):
        from tunelm.schemas import Plan

        rewards = []
        for text, raw_plan, identity in zip(completions, plan_json, task_id, strict=True):
            scored = score(
                restore_thinking(text), Plan.model_validate_json(raw_plan), self.verifier
            )
            append_jsonl(
                self.path,
                {
                    "schema_version": 2,
                    "task_id": identity,
                    "completion": restore_thinking(text),
                    **scored,
                },
            )
            rewards.append(scored["total"])
        return rewards


def train_grpo(config: TrainConfig) -> dict:
    """Perform policy updates against real verifier rewards on a frozen task bank."""
    _require_gpu()
    report = preflight(config, tokenize=True)
    from peft import get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    from datasets import Dataset

    merged = _merged_sft(config)
    model = _base_model(config.model, merged)
    model.config._name_or_path = merged
    model = get_peft_model(model, _lora(config))
    tokenizer = load_tokenizer(config.model)
    tokenizer.padding_side = "left"
    train, val = _rows(config)

    def dataset(rows):
        return Dataset.from_list(
            [
                {
                    "prompt": render_prompt(row, tokenizer),
                    "plan_json": row.plan.model_dump_json(),
                    "task_id": row.id,
                }
                for row in rows
            ]
        )

    args = GRPOConfig(output_dir=str(project_path(config.output_dir)), **config.training)
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        reward_funcs=MusicalReward(config.results_dir),
        train_dataset=dataset(train),
        eval_dataset=dataset(val) if val else None,
    )
    write_json(
        Path(config.output_dir) / "run.json",
        {**report, "weights_loaded": True, "status": "running", "config": config.model_dump()},
    )
    trainer.train(resume_from_checkpoint=config.resume or None)
    _save_run(config, trainer, tokenizer, report)
    return {"checkpoint": config.output_dir, "base": merged, "stage": "grpo"}
