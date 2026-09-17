from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tunelm.data.prompt_builder import normalize_prompt
from tunelm.io import read_jsonl


def prompt_hash(prompt: str, instruction: str | None = None) -> str:
    payload = normalize_prompt(prompt)
    if instruction:
        payload = f"{payload}|{normalize_prompt(instruction)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_signature(row: dict[str, Any]) -> str:
    instruction = row.get("instruction")
    task_type = row.get("task_type")
    if task_type == "edit":
        original = row.get("original_code") or row.get("id") or ""
        return prompt_hash(
            row.get("prompt", ""),
            f"{instruction or ''}:{original[:120]}",
        )
    if instruction:
        return prompt_hash(row.get("prompt", ""), instruction)
    if task_type == "repair":
        corruption = row.get("corruption") or "unknown"
        broken = row.get("broken_code") or row.get("id") or ""
        return prompt_hash(row.get("prompt", ""), f"{corruption}:{broken[:120]}")
    if task_type == "compose":
        constraints = row.get("constraints") or {}
        return prompt_hash(
            row.get("prompt", ""),
            json.dumps(constraints, sort_keys=True, default=str),
        )
    return prompt_hash(row.get("prompt", ""))


def load_prompt_hashes(path: str | Path) -> set[str]:
    resolved = Path(path)
    if not resolved.is_file():
        return set()
    return {task_signature(row) for row in read_jsonl(resolved)}


def reject_collisions(
    tasks: list[dict[str, Any]],
    *,
    forbidden: set[str] | None = None,
    dedupe_internal: bool = False,
) -> list[dict[str, Any]]:
    forbidden = forbidden or set()
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for task in tasks:
        signature = task_signature(task)
        if signature in forbidden or (dedupe_internal and signature in seen):
            continue
        seen.add(signature)
        accepted.append(task)
    return accepted


def compute_diversity_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row.get("task_type", "unknown"), []).append(row)

    metrics: dict[str, Any] = {"total": len(rows), "by_task_type": {}}
    for task_type, items in by_type.items():
        prompts = [normalize_prompt(item.get("prompt", "")) for item in items]
        instructions = [
            normalize_prompt(item.get("instruction", ""))
            for item in items
            if item.get("instruction")
        ]
        corruptions = [item.get("corruption") for item in items if item.get("corruption")]
        linguistic = instructions or prompts
        type_metrics = {
            "count": len(items),
            "unique_prompts": len(set(item.get("prompt", "") for item in items)),
            "unique_normalized_prompts": len(set(linguistic)),
            "unique_instructions": len(set(instructions)) if instructions else 0,
            "unique_corruptions": len(set(corruptions)) if corruptions else 0,
        }
        if task_type == "compose":
            constraint_fields = Counter()
            for item in items:
                constraints = item.get("constraints") or {}
                for key, value in constraints.items():
                    if value not in (None, [], {}):
                        constraint_fields[key] += 1
            type_metrics["constraint_field_coverage"] = dict(constraint_fields)
        metrics["by_task_type"][task_type] = type_metrics
    return metrics


def check_diversity_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    by_type = metrics.get("by_task_type", {})
    compose = by_type.get("compose", {})
    edit = by_type.get("edit", {})
    repair = by_type.get("repair", {})
    if compose.get("unique_normalized_prompts", 0) < int(
        thresholds.get("min_compose_prompt_patterns", 0)
    ):
        errors.append("compose prompt pattern diversity below threshold")
    if edit.get("unique_instructions", 0) < int(thresholds.get("min_edit_instructions", 0)):
        errors.append("edit instruction diversity below threshold")
    if repair.get("unique_corruptions", 0) < int(thresholds.get("min_repair_corruptions", 0)):
        errors.append("repair corruption diversity below threshold")
    if repair.get("unique_normalized_prompts", 0) < int(
        thresholds.get("min_repair_prompt_patterns", 0)
    ):
        errors.append("repair prompt style diversity below threshold")
    return errors


def benchmark_disjoint_ratio(
    benchmark_rows: list[dict[str, Any]],
    train_hashes: set[str],
) -> float:
    if not benchmark_rows:
        return 1.0
    novel = sum(1 for row in benchmark_rows if task_signature(row) not in train_hashes)
    return novel / len(benchmark_rows)
