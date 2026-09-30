#!/usr/bin/env python3
"""Merge a LoRA adapter on GPU and publish the full model to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, snapshot_download, upload_folder
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tunelm.config import repo_root


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required to publish models")
    return token


def _hf_username(token: str, override: str | None) -> str:
    if override:
        return override
    who = HfApi(token=token).whoami()
    name = who.get("name") or who.get("user")
    if not name:
        raise RuntimeError("could not resolve Hugging Face username; pass --hf-username")
    return str(name)


def _read_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter_config.json not found under {adapter_dir}")
    base_model = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base_model:
        raise RuntimeError(f"base_model_name_or_path missing in {config_path}")
    return str(base_model)


def _model_card(
    *,
    repo_id: str,
    base_model: str,
    adapter_dir: Path,
    tag: str | None,
    step: int | None,
    eval_loss: float | None,
    run_name: str | None,
    license_id: str = "other",
) -> str:
    lines = [
        "---",
        f"base_model: {base_model}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        f"license: {license_id}",
        "---",
        "",
        f"# {repo_id}",
        "",
        f"TuneLM merged checkpoint derived from `{base_model}`.",
        "The weights were modified by fine-tuning for Strudel generation.",
        "",
    ]
    if run_name:
        lines.append(f"- Run name: `{run_name}`")
    if step is not None:
        lines.append(f"- Training step: `{step}`")
    if tag:
        lines.append(f"- Checkpoint tag: `{tag}`")
    if eval_loss is not None:
        lines.append(f"- Best eval loss: `{eval_loss}`")
    experiment = adapter_dir / "experiment.json"
    if experiment.is_file():
        metadata = json.loads(experiment.read_text(encoding="utf-8"))
        commit = metadata.get("git_commit")
        if commit:
            lines.append(f"- Git commit: `{commit}`")
    lines.extend(["", "See LICENSE and NOTICE for the base-model license and modification notice."])
    return "\n".join(lines) + "\n"


def _write_license_and_notice(dest: Path, *, base_model: str, license_id: str) -> None:
    if license_id != "apache-2.0":
        raise RuntimeError(f"public distribution is not configured for base license {license_id!r}")
    with urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30) as response:
        (dest / "LICENSE").write_bytes(response.read())
    (dest / "NOTICE").write_text(
        "Gemma 4 is developed by Google DeepMind.\n"
        f"This TuneLM artifact modifies {base_model} through fine-tuning "
        "for Strudel music-code generation.\n",
        encoding="utf-8",
    )


def _verify_public(api: HfApi, repo_id: str) -> None:
    if api.model_info(repo_id).private:
        raise RuntimeError(f"{repo_id} remains private; make it public before treating publish as complete")


def default_repo_id(
    *,
    hf_username: str,
    run_name: str,
    step: int | None,
    tag: str | None,
) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d")
    suffix = f"{step}" if step is not None else "ckpt"
    repo = f"{hf_username}/tunelm-{run_name}-{date}-{suffix}"
    if tag:
        repo = f"{repo}-{tag}"
    return repo


def merge_and_publish(
    adapter_dir: Path,
    *,
    repo_id: str | None = None,
    base_model: str | None = None,
    private: bool = True,
    hf_username: str | None = None,
    tag: str | None = None,
    step: int | None = None,
    run_name: str | None = None,
    eval_loss: float | None = None,
) -> dict[str, str | int | float | None]:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

    token = _hf_token()
    username = _hf_username(token, hf_username)
    base_model = base_model or _read_base_model(adapter_dir)
    repo_id = repo_id or default_repo_id(
        hf_username=username,
        run_name=run_name or adapter_dir.name,
        step=step,
        tag=tag,
    )
    api = HfApi(token=token)
    base_info = api.model_info(base_model)
    license_id = str(getattr(base_info.card_data, "license", None) or "other")
    if not private and license_id != "apache-2.0":
        raise RuntimeError(f"public distribution is not configured for base license {license_id!r}")

    temp_root = Path(tempfile.mkdtemp(prefix="tunelm-publish-"))
    base_dir = temp_root / "base"
    merged_dir = temp_root / "merged"
    try:
        snapshot_download(
            base_model,
            local_dir=str(base_dir),
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        merged = model.merge_and_unload()
        merged_dir.mkdir(parents=True)
        merged.save_pretrained(merged_dir)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=False)
        tokenizer.save_pretrained(merged_dir)
        (merged_dir / "README.md").write_text(
            _model_card(
                repo_id=repo_id,
                base_model=base_model,
                adapter_dir=adapter_dir,
                tag=tag,
                step=step,
                eval_loss=eval_loss,
                run_name=run_name,
                license_id=license_id,
            ),
            encoding="utf-8",
        )
        if license_id == "apache-2.0":
            _write_license_and_notice(merged_dir, base_model=base_model, license_id=license_id)

        create_repo(repo_id, private=private, exist_ok=True, token=token)
        upload_folder(
            folder_path=str(merged_dir),
            repo_id=repo_id,
            token=token,
            commit_message=f"Publish merged TuneLM checkpoint ({tag or 'merged'})",
        )
        if not private:
            _verify_public(api, repo_id)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "repo_id": repo_id,
        "base_model": base_model,
        "adapter": str(adapter_dir),
        "tag": tag,
        "step": step,
        "eval_loss": eval_loss,
        "run_name": run_name,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a TuneLM adapter and publish to Hugging Face")
    parser.add_argument("--adapter", required=True, help="path to adapter or checkpoint directory")
    parser.add_argument("--repo-id", help="target HF repo id (default: auto-generated)")
    parser.add_argument("--base-model", help="override base model id (default: adapter_config.json)")
    parser.add_argument("--hf-username", help="HF namespace (default: token owner)")
    parser.add_argument("--run-name", help="training run name used in auto repo ids")
    parser.add_argument("--step", type=int, help="training step used in auto repo ids")
    parser.add_argument("--tag", choices=("best", "final"), help="checkpoint tag for repo naming")
    parser.add_argument("--eval-loss", type=float, help="recorded eval loss for the model card")
    parser.add_argument(
        "--public",
        action="store_true",
        help="create a public repo (default: private)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = repo_root()
    adapter = (root / args.adapter).resolve() if not Path(args.adapter).is_absolute() else Path(args.adapter)
    result = merge_and_publish(
        adapter,
        repo_id=args.repo_id,
        base_model=args.base_model,
        private=not args.public,
        hf_username=args.hf_username,
        tag=args.tag,
        step=args.step,
        run_name=args.run_name,
        eval_loss=args.eval_loss,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
