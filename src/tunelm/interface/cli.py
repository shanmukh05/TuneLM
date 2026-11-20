"""Explicit CLI for corpus preparation, inference, and required SFT/GRPO stages.

Commands that contact a teacher or load model weights are deliberately separate.
"""

from __future__ import annotations

import argparse
import json
import sys

from tunelm.core.io import load_config, load_env_file, write_json


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
    p.add_argument("--dataset-dir", default="datasets/samples")
    p = commands.add_parser("view")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--env-file", default=".env.local")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--memory", default="datasets/memory")
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--modal-app", default="tunelm-sft-inference", help="Modal app for SFT inference"
    )
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
    p.add_argument("--max-new-tokens", type=int, default=3072)
    p = commands.add_parser("predict")
    p.add_argument("--tasks", default="datasets/evaluation/benchmark.jsonl")
    p.add_argument("--checkpoint", help="Omit to generate with the pinned base model")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--tasks", default="datasets/evaluation/benchmark.jsonl")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("evaluate-audio", help="Fill and score a saved audio benchmark")
    p.add_argument("--config", help="Versioned audio evaluation YAML")
    p.add_argument("--benchmark")
    p.add_argument(
        "--experiments",
        dest="selected_experiments",
        nargs="+",
        metavar="NAME",
        help="Override the YAML list with saved experiment names",
    )
    p.add_argument("--tunelm-name")
    p.add_argument("--inference-config")
    p.add_argument("--inference-python")
    p.add_argument("--env-file")
    p.add_argument("--baseline-only", action="store_true", default=None)
    _audio_arguments(p, standalone=False)
    p = commands.add_parser("score-audio", help="Score a CSV of saved model audio")
    p.add_argument("--input", required=True)
    _audio_arguments(p)
    return cli


def _prompt_arguments(p):
    p.add_argument("prompt")
    p.add_argument("--plan", help="Explicit Plan YAML instead of prose-rule inference")
    p.add_argument("--memory", default="datasets/memory")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--offline", action="store_true")


def _audio_arguments(p, *, standalone=True):
    from tunelm.evaluation.audio_eval import DEFAULT_SECONDS

    p.add_argument("--output", required=standalone)
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS if standalone else None)
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto" if standalone else None
    )
    p.add_argument("--fad-reference", help="Override the bundled FMA-Pop statistics NPZ")
    p.add_argument(
        "--mad-reference", required=standalone, help="Reference audio or saved MAD embeddings"
    )
    p.add_argument("--embedding-cache", help="Reuse clip embeddings in this directory across runs")


def _verify(args):
    from tunelm.core.io import project_path
    from tunelm.strudel.verifier import Verifier

    result = Verifier(timeout=args.timeout).request(
        project_path(args.file).read_text(), operation=args.command, cycles=args.cycles
    )
    if args.output:
        write_json(args.output, result)
    return {k: v for k, v in result.items() if k != "events"}


def _memory(args):
    from tunelm.composition.memory import build_memory

    settings = load_config(args.config)
    if args.lexical:
        settings = {"mode": "lexical"}
    return build_memory(args.stems, args.output, settings)


def _task(args):
    from pathlib import Path

    from tunelm.composition.memory import Memory
    from tunelm.core.schemas import Plan
    from tunelm.model.inference import make_task
    from tunelm.model.training import ModelSpec, load_tokenizer

    spec = ModelSpec.model_validate(load_config(args.model_config))
    tokenizer = load_tokenizer(spec, offline=args.offline)
    explicit = Plan.model_validate(load_config(args.plan)) if args.plan else None
    from tunelm.core.io import project_path

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
    from tunelm.data.synthesis import generate, generation_preview
    from tunelm.model.training import ModelSpec, load_tokenizer

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
    from tunelm.data.synthesis import GenerationConfig

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
    from tunelm.data.prompt_generation import (
        generate_prompts,
        load_prompt_generation_config,
        prompt_preview,
    )

    load_env_file(args.env_file)
    config = load_prompt_generation_config(args.config)
    return prompt_preview(config) if args.dry_run else generate_prompts(config)


def _export_rl(args):
    from tunelm.data.rl_prompts import export_rl_tasks

    return export_rl_tasks(args.prompts_dir, args.dataset_dir)


def _train(args):
    from tunelm.model.training import TrainConfig, preflight, train_grpo, train_sft

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
    from tunelm.model.inference import load_checkpoint, predict

    task, spec, _ = _task(args)
    model, tokenizer = load_checkpoint(spec, args.checkpoint)
    result = predict(
        task, model, tokenizer, max_new_tokens=args.max_new_tokens, max_length=spec.max_length
    )
    write_json(args.output, {"task": task.model_dump(), "prediction": result})
    return result


def _predict(args):
    from tunelm.model.inference import predict_bank
    from tunelm.model.training import ModelSpec

    return predict_bank(
        args.tasks,
        args.output,
        ModelSpec.model_validate(load_config(args.model_config)),
        args.checkpoint,
    )


def _evaluate(args):
    from tunelm.model.inference import evaluate

    return evaluate(args.tasks, args.predictions, args.output)


def _audio(args):
    from tunelm.evaluation.audio_eval import AudioEvalConfig, evaluate_audio, score_audio

    if args.command == "evaluate-audio":
        return evaluate_audio(AudioEvalConfig.from_cli(args))
    return score_audio(args)


def _view(args):
    from tunelm.interface.viewer import serve_viewer

    return serve_viewer(
        args.port, args.env_file, args.model_config, args.memory, args.offline, args.modal_app
    )


def _ingest(args):
    from tunelm.data.corpus import ingest

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
        "evaluate-audio": _audio,
        "score-audio": _audio,
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
