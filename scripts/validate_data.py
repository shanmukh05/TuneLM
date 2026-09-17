#!/usr/bin/env python3
"""Validate TuneLM datasets against schemas, verifiers, and diversity thresholds."""

from __future__ import annotations

import argparse
import json

from pydantic import ValidationError

from tunelm.config import project_path
from tunelm.data.diversity import (
    benchmark_disjoint_ratio,
    check_diversity_thresholds,
    compute_diversity_metrics,
    load_prompt_hashes,
)
from tunelm.io import read_jsonl
from tunelm.schemas import ComposeTask, EditTask, RepairTask, RLTask, SFTExample, parse_task
from tunelm.sft_data.validator import validate_solution
from tunelm.strudel.executor import StrudelExecutor


TASK_MODELS = {"compose": ComposeTask, "edit": EditTask, "repair": RepairTask}


def _behavior_error(
    value: SFTExample | RLTask, kind: str, executor: StrudelExecutor
) -> tuple[str, str] | None:
    if kind == "sft":
        payload = value.model_dump(mode="json")
        task_model = TASK_MODELS[value.task_type.value]
        task = task_model.model_validate(
            {key: payload[key] for key in task_model.model_fields if key in payload}
        )
        report = validate_solution(task, value.response, executor)
        if not report.valid:
            return "solution", report.error or "verification failed"
        return None
    if value.task_type == "repair" and value.reference_code:
        result = executor.run(value.reference_code)
        if not result.valid:
            return "execution", result.error or "reference does not execute"
    return None


def _load_rows(
    path: str, kind: str, executor: StrudelExecutor, check_execution: bool
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    counts = {"total": 0, "schema": 0, "execution": 0, "solution": 0}
    for row in read_jsonl(path):
        counts["total"] += 1
        try:
            value = SFTExample.model_validate(row) if kind == "sft" else parse_task(row)
        except (ValidationError, ValueError) as exc:
            counts["schema"] += 1
            print(f"row {counts['total']}: schema error: {exc}")
            continue
        rows.append(row)
        error = _behavior_error(value, kind, executor) if check_execution else None
        if error:
            category, message = error
            counts[category] += 1
            print(f"row {counts['total']}: {category} invalid: {message}")
    return rows, counts


def _check_diversity(args: argparse.Namespace, rows: list[dict], metrics: dict) -> list[str]:
    errors: list[str] = []
    if args.diversity_config:
        thresholds = json.loads(project_path(args.diversity_config).read_text(encoding="utf-8"))
        errors.extend(check_diversity_thresholds(metrics, thresholds))
    if not args.disjoint_from:
        return errors
    ratio = benchmark_disjoint_ratio(rows, load_prompt_hashes(args.disjoint_from))
    print(f"disjoint_ratio={ratio:.4f}")
    minimum = float(args.min_unique_prompts or 0.95)
    if ratio < minimum:
        errors.append(f"benchmark overlap too high: {ratio:.2%} novel < {minimum:.2%} required")
    return errors


def main() -> None:
    """Validate one JSONL dataset and exit nonzero on any failed check."""
    parser = argparse.ArgumentParser(description="Validate TuneLM JSONL datasets")
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=["sft", "rl"], required=True)
    parser.add_argument("--check-execution", action="store_true")
    parser.add_argument("--backend", choices=["auto", "node", "static"], default="auto")
    parser.add_argument(
        "--disjoint-from", help="Path to a dataset that must not share prompt hashes"
    )
    parser.add_argument(
        "--min-unique-prompts", type=float, help="Minimum disjoint ratio vs --disjoint-from"
    )
    parser.add_argument("--diversity-config", help="JSON file with diversity thresholds")
    args = parser.parse_args()

    executor = StrudelExecutor(backend=args.backend)
    rows, counts = _load_rows(args.input, args.kind, executor, args.check_execution)

    metrics = compute_diversity_metrics(rows)
    print(json.dumps(metrics, indent=2))

    diversity_errors = _check_diversity(args, rows, metrics)

    print(
        f"rows={counts['total']} schema_invalid={counts['schema']} "
        f"execution_invalid={counts['execution']} solution_invalid={counts['solution']} "
        f"valid={counts['total'] - sum(counts[key] for key in ('schema', 'execution', 'solution'))}"
    )
    if any(counts[key] for key in ("schema", "execution", "solution")) or diversity_errors:
        for error in diversity_errors:
            print(f"diversity error: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
