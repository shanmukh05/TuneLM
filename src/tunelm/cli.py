"""Explicit CLI for corpus preparation, inference, and required SFT/GRPO stages.

Commands that contact a teacher or load model weights are deliberately separate.
"""

from __future__ import annotations

import argparse
import json
import sys

from tunelm.io import load_config, load_env_file, write_json


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="tunelm", description="Grounded Strudel music pipelines")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ["verify", "decompose"]:
        p = commands.add_parser(name)
        p.add_argument("file")
        p.add_argument("--cycles", type=int, default=32)
        p.add_argument("--timeout", type=float, default=5)
        p.add_argument("--output")
    p = commands.add_parser("ingest")
    p.add_argument("sources")
    p.add_argument("--cycles", type=int, default=32)
    p = commands.add_parser("build-memory")
    p.add_argument("--stems", default="datasets/sources/verified.jsonl")
    p.add_argument("--config", default="configs/registry/memory.yaml")
    p.add_argument("--output", default="datasets/memory")
    p.add_argument(
        "--lexical", action="store_true", help="Explicit offline baseline, not semantic embeddings"
    )
    p = commands.add_parser("plan")
    _prompt_arguments(p)
    p.add_argument("--output", default="results/plans/latest.json")
    p = commands.add_parser("generate")
    p.add_argument("--config", default="configs/samples-sft.yaml")
    p.add_argument(
        "--env-file", default=".env.local", help="Literal KEY=value file; shell env wins"
    )
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument(
        "--provider", choices=["openai", "gemini", "anthropic", "claude", "deepseek", "openrouter"]
    )
    p.add_argument("--teacher-model")
    p.add_argument("--run-id")
    p.add_argument("--dataset-dir")
    p.add_argument("--count", type=int)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze SFT inputs and benchmark tasks; no answer calls",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview model quotas without API calls or state writes",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use a cached tokenizer; teacher calls still require network",
    )
    p = commands.add_parser("create-prompts")
    p.add_argument(
        "--config", required=True, help="configs/prompts-sft.yaml or configs/prompts-rl.yaml"
    )
    p.add_argument("--env-file", default=".env.local")
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("export-rl", help="Write answer-free GRPO tasks from the RL bank")
    p.add_argument("--prompts-dir", default="datasets/prompts/generated")
    p.add_argument("--dataset-dir", default="datasets/creative")
    p = commands.add_parser("view")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--env-file", default=".env.local")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--memory", default="datasets/memory")
    p.add_argument("--offline", action="store_true")
    for name in ["train-sft", "train-grpo"]:
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--model-config", default="configs/model.yaml")
        p.add_argument("--env-file", default=".env.local")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--tokenize", action="store_true", help="Also verify actual tokenizer masks and lengths"
        )
        p.add_argument("--offline", action="store_true")
        p.add_argument("--resume", nargs="?", const=True)
    p = commands.add_parser("infer")
    _prompt_arguments(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results/inference/latest.json")
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p = commands.add_parser("predict")
    p.add_argument("--tasks", default="datasets/evaluation/benchmark.jsonl")
    p.add_argument("--checkpoint", help="Omit to generate with the pinned base model")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--tasks", default="datasets/evaluation/benchmark.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    return cli


def _prompt_arguments(p):
    p.add_argument("prompt")
    p.add_argument("--plan", help="Explicit Plan YAML instead of prose-rule inference")
    p.add_argument("--memory", default="datasets/memory")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--offline", action="store_true")


def _verify(args):
    from tunelm.io import project_path
    from tunelm.verifier import Verifier

    result = Verifier(timeout=args.timeout).request(
        project_path(args.file).read_text(), operation=args.command, cycles=args.cycles
    )
    if args.output:
        write_json(args.output, result)
    return {k: v for k, v in result.items() if k != "events"}


def _memory(args):
    from tunelm.memory import build_memory

    settings = load_config(args.config)
    if args.lexical:
        settings = {"mode": "lexical"}
    return build_memory(args.stems, args.output, settings)


def _task(args):
    from pathlib import Path

    from tunelm.inference import make_task
    from tunelm.memory import Memory
    from tunelm.schemas import Plan
    from tunelm.training import ModelSpec, load_tokenizer

    spec = ModelSpec.model_validate(load_config(args.model_config))
    tokenizer = load_tokenizer(spec, offline=args.offline)
    explicit = Plan.model_validate(load_config(args.plan)) if args.plan else None
    from tunelm.io import project_path

    memory = (
        Memory(args.memory) if project_path(Path(args.memory) / "metadata.json").exists() else None
    )
    task = make_task(args.prompt, memory, tokenizer, plan=explicit)
    return task, spec, tokenizer


def _plan(args):
    task, _, _ = _task(args)
    write_json(args.output, task.model_dump())
    return {"task": args.output, "plan": task.plan.model_dump()}


def _generate(args):
    from tunelm.synthesis import generate, generation_preview
    from tunelm.training import ModelSpec, load_tokenizer

    load_env_file(args.env_file)
    config = generation_config(args)
    if args.dry_run:
        return generation_preview(config)
    spec = ModelSpec.model_validate(load_config(args.model_config))
    return generate(
        config,
        load_tokenizer(spec, offline=args.offline),
        prepare_only=args.prepare_only,
    )


def generation_config(args):
    """Apply explicit CLI selections before validating provider and output settings."""
    from tunelm.synthesis import GenerationConfig

    raw = load_config(args.config)
    overrides = {"run_id": args.run_id, "dataset_dir": args.dataset_dir, "count": args.count}
    raw.update({key: value for key, value in overrides.items() if value is not None})
    if args.provider or args.teacher_model:
        if not args.provider or not args.teacher_model:
            raise ValueError("A single-model override requires both --provider and --teacher-model")
        raw["models"] = [
            {
                "id": "cli-teacher",
                "provider": args.provider,
                "model": args.teacher_model,
                "fraction": 1.0,
            }
        ]
    return GenerationConfig.model_validate(raw)


def _create_prompts(args):
    from tunelm.prompt_generation import generate_prompts, load_prompt_generation_config, prompt_preview

    load_env_file(args.env_file)
    config = load_prompt_generation_config(args.config)
    return prompt_preview(config) if args.dry_run else generate_prompts(config)


def _export_rl(args):
    from tunelm.rl_prompts import export_rl_tasks

    return export_rl_tasks(args.prompts_dir, args.dataset_dir)


def _train(args):
    from tunelm.training import TrainConfig, preflight, train_grpo, train_sft

    load_env_file(args.env_file)
    raw = load_config(args.config)
    raw.setdefault("model", load_config(args.model_config))
    config = TrainConfig.model_validate(raw)
    if args.command != "train-" + config.stage:
        raise ValueError("Training stage does not match the selected command")
    if args.resume is not None:
        config.resume = args.resume
    if args.dry_run:
        return preflight(config, tokenize=args.tokenize, offline=args.offline)
    return train_sft(config) if config.stage == "sft" else train_grpo(config)


def _infer(args):
    from tunelm.inference import load_checkpoint, predict

    task, spec, _ = _task(args)
    model, tokenizer = load_checkpoint(spec, args.checkpoint)
    result = predict(
        task, model, tokenizer, max_new_tokens=args.max_new_tokens, max_length=spec.max_length
    )
    write_json(args.output, {"task": task.model_dump(), "prediction": result})
    return result


def _predict(args):
    from tunelm.inference import predict_bank
    from tunelm.training import ModelSpec

    return predict_bank(
        args.tasks,
        args.output,
        ModelSpec.model_validate(load_config(args.model_config)),
        args.checkpoint,
    )


def _evaluate(args):
    from tunelm.inference import evaluate

    return evaluate(args.tasks, args.predictions, args.output)


def _view(args):
    from tunelm.viewer import serve_viewer

    return serve_viewer(args.port, args.env_file, args.model_config, args.memory, args.offline)


def _ingest(args):
    from tunelm.corpus import ingest

    return ingest(args.sources, cycles=args.cycles)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    handlers = {
        "verify": _verify,
        "decompose": _verify,
        "ingest": _ingest,
        "build-memory": _memory,
        "plan": _plan,
        "generate": _generate,
        "create-prompts": _create_prompts,
        "export-rl": _export_rl,
        "train-sft": _train,
        "train-grpo": _train,
        "infer": _infer,
        "predict": _predict,
        "evaluate": _evaluate,
        "view": _view,
    }
    try:
        result = handlers[args.command](args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        if result.get("valid") is False:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(json.dumps({"error": str(error), "command": args.command}), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
