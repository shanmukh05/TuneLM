"""Generate and render only missing TuneLM audio for a frozen benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tunelm.composition.responses import (
    listener_messages,
    parse_completion,
    render_chat,
    restore_thinking,
)
from tunelm.core.io import (
    generation_lock,
    load_config,
    load_env_file,
    project_path,
    read_json,
    write_json,
)
from tunelm.evaluation.audio_eval import (
    SAFE_NAME,
    AudioEvalConfig,
    model_audio,
    prompt_id,
    read_benchmark,
)
from tunelm.evaluation.render import RENDER_SECONDS, render_audio


def load_composer(config: dict, device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a Vast.ai GPU host or --device cpu.")
    checkpoint = config["model"]
    local = project_path(checkpoint)
    if local.is_dir():
        checkpoint = str(local)
        revision = None
    else:
        model_config = hf_hub_download(checkpoint, "config.json", revision=config["revision"])
        revision = Path(model_config).parent.name
    options = {"revision": revision, "trust_remote_code": config["trust_remote_code"]}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            **options,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    model.generation_config.use_cache = True
    return model, tokenizer, revision


def _load_and_record(config: dict, device: str, metadata: Path):
    model, tokenizer, revision = load_composer(config, device)
    if metadata.exists() and read_json(metadata)["resolved_revision"] != revision:
        raise ValueError(f"{metadata.parent.name} resolves to a different checkpoint revision")
    write_json(
        metadata,
        {
            "settings": config,
            "resolved_revision": revision,
            "render_seconds": RENDER_SECONDS,
        },
    )
    return model, tokenizer


def _generate_batch(records, model, tokenizer, config, temperature, attempt):
    import torch

    prompts = [render_chat(listener_messages(prompt), tokenizer) for _, prompt in records]
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    width = inputs.input_ids.shape[1]
    max_tokens = config["max_new_tokens"]
    if width + max_tokens > config["max_length"]:
        raise ValueError("Prompt plus completion exceeds configured context")
    options = {"do_sample": temperature > 0}
    if temperature > 0:
        torch.manual_seed(int(prompt_id(f"{attempt}:{records}")[:8], 16))
        options["temperature"] = temperature
    sizes = inputs.attention_mask.sum(dim=1).tolist()
    with torch.inference_mode():
        output = model.generate(
            **inputs.to(model.device),
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **options,
        )[:, width:].tolist()
    for (pid, prompt), size, tokens in zip(records, sizes, output, strict=True):
        eos = tokenizer.eos_token_id
        if eos in tokens:
            tokens = tokens[: tokens.index(eos) + 1]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if tokenizer.eos_token is not None:
            text = text.split(tokenizer.eos_token, 1)[0]
        yield {
            "prompt_id": pid,
            "prompt": prompt,
            "completion": restore_thinking(text.rstrip()),
            "attempt": attempt,
            "usage": {
                "prompt_tokens": size,
                "completion_tokens": len(tokens),
                "total_tokens": size + len(tokens),
            },
        }


def _try_render(path, prompt, audio, cache, seconds):
    generated = read_json(path)
    if generated["prompt"] != prompt:
        raise ValueError(f"Saved completion {path.name} uses different prompt text")
    pid = generated["prompt_id"]
    try:
        _, code = parse_completion(generated["completion"])
        render_audio(code, audio / f"{pid}.wav", cache, scoring_seconds=seconds)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"TuneLM {pid} render failed: {error}", file=sys.stderr, flush=True)
        return {"prompt_id": pid, "error": str(error)}
    return None


def _run_attempt(rows, attempt, audio, cache, settings, config, checkpoint, pool):
    futures = []
    suffix = "" if attempt == 1 else f"-attempt-{attempt}"
    for start in range(0, len(rows), settings.batch_size):
        batch = rows[start : start + settings.batch_size]
        paths = {pid: audio / "completions" / f"{pid}{suffix}.json" for pid, _ in batch}
        pending = [(pid, prompt) for pid, prompt in batch if not paths[pid].is_file()]
        if pending:
            if checkpoint is None:
                checkpoint = _load_and_record(config, settings.device, audio / "generation.json")
            print(
                f"TuneLM attempt {attempt}, prompts {start + 1}–{start + len(batch)}/{len(rows)}",
                file=sys.stderr,
                flush=True,
            )
            temperature = config["temperature"] if attempt == 1 else settings.retry_temperature
            for row in _generate_batch(pending, *checkpoint, config, temperature, attempt):
                write_json(paths[row["prompt_id"]], row)
        for pid, prompt in batch:
            futures.append(
                pool.submit(_try_render, paths[pid], prompt, audio, cache, settings.seconds)
            )
    failures = [failure for future in futures if (failure := future.result())]
    return failures, checkpoint


def generate_missing(benchmark: Path, name: str, config: dict, settings) -> int:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError("--name must be a single directory name")
    prompts = read_benchmark(benchmark)
    audio = benchmark / "audio" / name
    metadata = audio / "generation.json"
    with generation_lock(audio):
        if metadata.exists() and read_json(metadata)["settings"] != config:
            raise ValueError(f"{name} was generated with different settings; use a new model name")
        present = model_audio(audio)
        pending = [(pid, prompt) for pid, prompt in prompts if pid not in present]
        count = len(pending)
        if not count:
            return 0
        checkpoint = None
        with ThreadPoolExecutor(max_workers=settings.render_workers) as pool:
            for attempt in range(1, settings.attempts + 1):
                failures, checkpoint = _run_attempt(
                    pending,
                    attempt,
                    audio,
                    benchmark / ".render-assets",
                    settings,
                    config,
                    checkpoint,
                    pool,
                )
                if not failures:
                    break
                failed = {row["prompt_id"] for row in failures}
                pending = [(pid, prompt) for pid, prompt in pending if pid in failed]
        failure_file = audio / "failures.json"
        if failures:
            write_json(failure_file, failures)
            print(f"{len(failures)} TuneLM renders failed; see {failure_file}", file=sys.stderr)
            return -1
        failure_file.unlink(missing_ok=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config")
    parser.add_argument("--benchmark")
    parser.add_argument("--name", dest="tunelm_name", default="tunelm-sft")
    parser.add_argument("--config", dest="inference_config", default="configs/inference.yaml")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--render-workers", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-temperature", type=float, default=0.4)
    args = parser.parse_args()
    try:
        if args.eval_config:
            settings = AudioEvalConfig.model_validate(load_config(args.eval_config))
        else:
            fields = vars(args).keys() & AudioEvalConfig.model_fields.keys()
            settings = AudioEvalConfig(
                output="",
                mad_reference="",
                **{name: getattr(args, name) for name in fields},
            )
        load_env_file(settings.env_file)
        count = generate_missing(
            project_path(settings.benchmark),
            settings.tunelm_name,
            load_config(settings.inference_config),
            settings,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        sys.exit(str(error))
    if count < 0:
        raise SystemExit(2)
    print(f"Saved {count} new TuneLM audio clips")


if __name__ == "__main__":
    main()
