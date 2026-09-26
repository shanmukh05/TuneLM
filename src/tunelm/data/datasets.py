"""Load TuneLM JSONL rows for training without Arrow schema inference on nullable task fields."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from tunelm.config import project_path


def load_jsonl_dataset(path: str | Path) -> Dataset:
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Dataset.from_dict({"json": lines})


def load_sft_datasets(config: dict) -> tuple[Dataset, Dataset | None]:
    """Return train and optional eval datasets, excluding val ids from train."""
    data = config.get("data", {})
    train_path = project_path(data["train_file"])
    val_file = data.get("val_file")
    if not val_file:
        return load_jsonl_dataset(train_path), None

    eval_dataset = load_jsonl_dataset(project_path(val_file))
    val_ids = {json.loads(row["json"])["id"] for row in eval_dataset}
    train_lines = [
        line.strip()
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["id"] not in val_ids
    ]
    return Dataset.from_dict({"json": train_lines}), eval_dataset
