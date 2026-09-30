#!/usr/bin/env python3
"""Add fixed real-user-style compose requests to a generated task benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tunelm.eval.manual_prompts import MANUAL_PROMPTS
from tunelm.io import read_jsonl, write_jsonl
from tunelm.schemas import parse_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="datasets/evaluation/benchmark_quality.jsonl")
    parser.add_argument("--output", default="datasets/evaluation/benchmark_ready.jsonl")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Benchmark already exists: {args.output}")
    rows = list(read_jsonl(args.source))
    seen = {row["prompt"].casefold().strip() for row in rows}
    for index, prompt in enumerate(MANUAL_PROMPTS, start=1):
        if prompt.casefold().strip() in seen:
            continue
        row = {
            "id": f"benchmark-manual-{index:03d}",
            "task_type": "compose",
            "prompt": prompt,
            "constraints": {},
            "tags": ["manual", "user-style"],
        }
        rows.append(parse_task(row).model_dump(mode="json"))
        seen.add(prompt.casefold().strip())
    write_jsonl(args.output, rows)
    print(json.dumps({"tasks": len(rows), "manual": sum("manual" in r.get("tags", []) for r in rows)}))


if __name__ == "__main__":
    main()
