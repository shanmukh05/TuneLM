#!/usr/bin/env python3
"""Build reviewable quality SFT and RL banks without changing source datasets."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from tunelm.io import read_jsonl, write_jsonl
from tunelm.models.templates import task_from_example
from tunelm.rewards.constraints import has_verifiable_constraints, stated_constraints
from tunelm.schemas import ModelResponse, parse_task
from tunelm.sft_data.validator import validate_solution
from tunelm.strudel.executor import StrudelExecutor


def curate_sft(args) -> None:
    for path in (args.output, args.rejected):
        if Path(path).exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace it")
    executor = StrudelExecutor(
        backend="node", cycles=args.cycles, timeout_seconds=args.timeout, max_events=8192
    )
    accepted = []
    rejected = []
    reasons: Counter[str] = Counter()
    for index, row in enumerate(read_jsonl(args.source)):
        if args.limit is not None and index >= args.limit:
            break
        task = task_from_example(row)
        response = ModelResponse.model_validate(row["response"])
        report = validate_solution(
            task,
            response,
            executor,
            minimum_constraint_score=args.min_constraints,
            minimum_task_consistency_score=args.min_task_consistency,
            minimum_response_consistency_score=args.min_response_consistency,
            minimum_compose_richness=args.min_richness,
        )
        output_row = copy.deepcopy(row)
        output_row.setdefault("metadata", {})["valid"] = report.valid
        output_row["metadata"]["validation"] = report.as_dict()
        if report.valid:
            accepted.append(output_row)
        else:
            output_row["metadata"]["quality_rejection"] = report.error
            rejected.append(output_row)
            for reason in (report.error or "unknown").split("; "):
                reasons[reason] += 1
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "reasons": reasons}))


def _brief(row: dict) -> dict:
    row = copy.deepcopy(row)
    constraints = row.get("constraints") or {}
    required = list(constraints.get("required_instruments") or [])[:2]
    forbidden = list(constraints.get("forbidden_instruments") or [])[:1]
    constraints["required_instruments"] = required
    constraints["forbidden_instruments"] = forbidden
    tempo = constraints.get("tempo")
    if isinstance(tempo, dict):
        bpm = round((float(tempo["min"]) + float(tempo["max"])) / 2)
    else:
        bpm = round(float(tempo)) if tempo is not None else None
    requirements = []
    if bpm is not None:
        requirements.append(f"{bpm} BPM")
    if required:
        requirements.append("use " + " and ".join(required))
    if forbidden:
        requirements.append("no " + " or ".join(forbidden))
    if constraints.get("min_layers"):
        requirements.append(f"at least {constraints['min_layers']} layers")
    if requirements:
        row["prompt"] = row["prompt"].rstrip(" .") + ". Please make it " + ", ".join(requirements) + "."
    row["constraints"] = constraints
    row["tags"] = [tag for tag in row.get("tags", []) if not tag.startswith("style-")]
    row["tags"].append("style-visible-brief")
    parse_task(row)
    return row


def curate_rl(args) -> None:
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite to replace it")
    rows = list(read_jsonl(args.source))
    compose_ids = [row["id"] for row in rows if row.get("task_type") == "compose"]
    rng = random.Random(args.seed)
    rng.shuffle(compose_ids)
    selected = set(compose_ids[: round(len(compose_ids) * args.brief_fraction)])
    output = [_brief(row) if row["id"] in selected else row for row in rows]
    compose = [row for row in output if row.get("task_type") == "compose"]
    covered = sum(
        has_verifiable_constraints(stated_constraints(row["prompt"], row.get("constraints", {})))
        for row in compose
    )
    coverage = covered / len(compose) if compose else 0
    if coverage < args.min_coverage:
        raise ValueError(f"compose constraint coverage {coverage:.1%} < {args.min_coverage:.1%}")
    write_jsonl(args.output, output)
    print(json.dumps({"rows": len(output), "compose": len(compose), "constraint_coverage": coverage}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sft = subparsers.add_parser("sft")
    sft.add_argument("--source", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--rejected", required=True)
    sft.add_argument("--limit", type=int)
    sft.add_argument("--overwrite", action="store_true")
    sft.add_argument("--cycles", type=float, default=8)
    sft.add_argument("--timeout", type=float, default=10)
    sft.add_argument("--min-constraints", type=float, default=0.75)
    sft.add_argument("--min-task-consistency", type=float, default=0.8)
    sft.add_argument("--min-response-consistency", type=float, default=0.8)
    sft.add_argument("--min-richness", type=float, default=0.6)
    rl = subparsers.add_parser("rl")
    rl.add_argument("--source", required=True)
    rl.add_argument("--output", required=True)
    rl.add_argument("--seed", type=int, default=9001)
    rl.add_argument("--brief-fraction", type=float, default=0.6)
    rl.add_argument("--min-coverage", type=float, default=0.3)
    rl.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "sft":
        curate_sft(args)
    else:
        curate_rl(args)


if __name__ == "__main__":
    main()
