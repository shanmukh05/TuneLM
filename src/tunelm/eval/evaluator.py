from __future__ import annotations

import argparse
import json

from tunelm.config import load_config, project_path
from tunelm.eval.metrics import aggregate
from tunelm.io import read_jsonl, write_jsonl
from tunelm.rewards import RewardEngine
from tunelm.schemas import parse_task
from tunelm.strudel.executor import executor_from_config


def evaluate_predictions(config: dict) -> dict:
    task_rows = list(read_jsonl(project_path(config["tasks"])))
    tasks = {row["id"]: parse_task(row) for row in task_rows}
    predictions = read_jsonl(project_path(config["predictions"]))
    reward_cfg = config.get("rewards", {})
    weights = {
        name: float(values.get("weight", 1.0))
        for name, values in reward_cfg.items()
        if values.get("enabled", True)
    }
    engine = RewardEngine(executor_from_config(config), weights or None)
    records = []
    for prediction in predictions:
        task_id = prediction["task_id"]
        if task_id not in tasks:
            raise ValueError(f"Prediction references unknown task_id: {task_id}")
        task = tasks[task_id]
        score = engine.score(prediction["completion"], task)
        records.append(
            {
                "task_id": task_id,
                "task_type": task.task_type.value,
                "total_reward": score.total,
                "rewards": score.components,
            }
        )
    output = project_path(config.get("output", "results/evaluation.jsonl"))
    write_jsonl(output, records)
    summary = aggregate(records)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TuneLM predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Generate benchmark predictions before scoring")
    parser.add_argument("--limit", type=int, help="Only generate this many predictions")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        from tunelm.eval.predictor import generate_predictions

        count = generate_predictions(
            project_path(config["tasks"]),
            project_path(config["predictions"]),
            args.checkpoint,
            temperature=float(config.get("generation", {}).get("temperature", 0.0)),
            max_new_tokens=int(config.get("generation", {}).get("max_new_tokens", 1536)),
            limit=args.limit,
        )
        print(f"generated {count} predictions")
    print(json.dumps(evaluate_predictions(config), indent=2))


if __name__ == "__main__":
    main()
