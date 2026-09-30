from __future__ import annotations

import argparse
import csv
import json

from tunelm.config import load_config, project_path
from tunelm.eval.metrics import aggregate
from tunelm.io import read_jsonl, write_jsonl
from tunelm.models.templates import task_from_example
from tunelm.rewards import RewardEngine
from tunelm.rewards.richness import is_short_loop_request
from tunelm.strudel.executor import executor_from_config
from tunelm.strudel.parser import ResponseParseError, parse_model_response


def evaluate_predictions(config: dict) -> dict:
    task_rows = list(read_jsonl(project_path(config["tasks"])))
    tasks = {row["id"]: task_from_example(row) for row in task_rows}
    predictions = read_jsonl(project_path(config["predictions"]))
    reward_cfg = config.get("rewards", {})
    weights = {
        name: float(values.get("weight", 1.0))
        for name, values in reward_cfg.items()
        if values.get("enabled", True)
    }
    engine = RewardEngine(executor_from_config(config), weights or None)
    records = []
    seen: set[str] = set()
    for prediction in predictions:
        task_id = prediction.get("task_id", prediction.get("id"))
        if task_id not in tasks:
            raise ValueError(f"Prediction references unknown task_id: {task_id}")
        if task_id in seen:
            raise ValueError(f"Duplicate prediction for task_id: {task_id}")
        seen.add(task_id)
        task = tasks[task_id]
        completion = prediction.get("completion", prediction.get("response"))
        if isinstance(completion, dict):
            completion = json.dumps(completion, ensure_ascii=False)
        if completion is None:
            completion = (prediction.get("metadata") or {}).get("raw_completion", "")
        score = engine.score(completion, task)
        try:
            parse_model_response(completion)
            parse_valid = True
        except ResponseParseError:
            parse_valid = False
        execution = score.execution
        features = execution.features if execution else {}
        bpm = features.get("tempo")
        distinct = features.get("distinct_cycles") or 0
        unique_seconds = distinct * 240 / bpm if bpm else 0.0
        short_loop = is_short_loop_request(task.prompt)
        diagnostics = {
            "parse_valid": parse_valid,
            "execution_valid": bool(execution and execution.valid),
            "playable": bool(execution and execution.playable),
            "active_cycles": features.get("active_cycles", 0),
            "distinct_cycles": distinct,
            "unique_seconds_proxy": round(unique_seconds, 2),
            "form_target_met": None if task.task_type.value != "compose" or short_loop else unique_seconds >= 15,
            "execution_error": execution.error if execution else None,
        }
        records.append(
            {
                "task_id": task_id,
                "task_type": task.task_type.value,
                "total_reward": score.total,
                "rewards": score.components,
                "diagnostics": diagnostics,
            }
        )
    output = project_path(config.get("output", "results/evaluation.jsonl"))
    write_jsonl(output, records)
    summary = aggregate(records)
    summary["task_count"] = len(tasks)
    summary["prediction_coverage"] = len(seen) / len(tasks) if tasks else 0.0
    summary["missing_predictions"] = len(tasks) - len(seen)
    review_path = project_path(config.get("review_file", str(output.with_suffix(".review.csv"))))
    if not review_path.exists():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["task_id", "task_type", "prompt", "audible", "instruction_match", "musical_quality", "notes"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "task_id": record["task_id"],
                        "task_type": record["task_type"],
                        "prompt": tasks[record["task_id"]].prompt,
                    }
                )
    summary["review_file"] = str(review_path)
    manual = {"audible": [], "instruction_match": [], "musical_quality": []}
    with review_path.open(newline="", encoding="utf-8") as handle:
        for review in csv.DictReader(handle):
            if review.get("task_id") not in seen:
                continue
            for name, values in manual.items():
                if review.get(name, "").strip():
                    values.append(float(review[name]))
    summary["manual_review"] = {
        name: {"count": len(values), "mean": sum(values) / len(values) if values else None}
        for name, values in manual.items()
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TuneLM predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Generate benchmark predictions before scoring")
    parser.add_argument("--revision", help="Hugging Face branch, tag, or commit")
    parser.add_argument("--limit", type=int, help="Only generate this many predictions")
    parser.add_argument("--predictions", help="Score an existing prediction file, including Modal JSONL")
    parser.add_argument("--output", help="Write evaluation results and review CSV to this path")
    parser.add_argument("--tasks", help="Use these tasks instead of the config task file")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, help="Completion token limit")
    parser.add_argument("--mode", choices=["combined", "code-first"], help="Inference mode")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.predictions:
        config["predictions"] = args.predictions
    if args.output:
        config["output"] = args.output
    if args.tasks:
        config["tasks"] = args.tasks
    if args.temperature is not None:
        config.setdefault("generation", {})["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        config.setdefault("generation", {})["max_new_tokens"] = args.max_new_tokens
    if args.mode is not None:
        config.setdefault("generation", {})["mode"] = args.mode
    if args.checkpoint:
        from tunelm.eval.predictor import generate_predictions

        count = generate_predictions(
            project_path(config["tasks"]),
            project_path(config["predictions"]),
            args.checkpoint,
            temperature=float(config.get("generation", {}).get("temperature", 0.0)),
            max_new_tokens=int(config.get("generation", {}).get("max_new_tokens", 1536)),
            limit=args.limit,
            mode=config.get("generation", {}).get("mode", "combined"),
            revision=args.revision,
        )
        print(f"generated {count} predictions")
    print(json.dumps(evaluate_predictions(config), indent=2))


if __name__ == "__main__":
    main()
