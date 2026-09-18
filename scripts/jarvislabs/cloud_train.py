#!/usr/bin/env python3
"""Launch TuneLM SFT or GRPO training on a JarvisLabs instance via the ``jl`` CLI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tunelm.config import load_config, repo_root


REMOTE_DIRNAME = "tunelm"
SESSION_NAME = ".jarvislabs_session.json"
SKIP_DIR_NAMES = frozenset(
    {
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        "node_modules",
        "checkpoints",
        "results",
    }
)
DETACH_EXIT_CODES = {130, 143, -2, -15}


def load_local_env(root: Path) -> None:
    for name in (".env", ".env.local"):
        path = root / name
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'").strip('"')


def session_path(root: Path) -> Path:
    return root / "results" / SESSION_NAME


def load_session(root: Path) -> dict[str, str]:
    path = session_path(root)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def save_session(root: Path, **fields: str) -> None:
    path = session_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_session(root)
    current.update({key: value for key, value in fields.items() if value})
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def _ignore(_directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name in SKIP_DIR_NAMES or name.endswith(".egg-info")]


def _copy_into_payload(root: Path, dest: Path, value: str) -> None:
    source = (root / value).resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"cloud artifacts must be inside the repository: {source}") from exc
    if not source.exists():
        raise FileNotFoundError(f"required cloud artifact not found: {relative}")
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=_ignore)
    else:
        shutil.copy2(source, target)


def stage_payload(root: Path, dest: Path, config_path: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(root / "src", dest / "src", ignore=_ignore)
    shutil.copytree(root / "configs", dest / "configs")
    shutil.copytree(root / "scripts", dest / "scripts")
    config = load_config(root / config_path)
    _copy_into_payload(root, dest, config["data"]["train_file"])
    adapter = config.get("model", {}).get("adapter")
    if adapter and (root / adapter).exists():
        _copy_into_payload(root, dest, adapter)
    for name in ("pyproject.toml", "package.json", "package-lock.json", "README.md"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, dest / name)


def _decode_json(text: str) -> Any:
    text = text.strip()
    return json.loads(text) if text else None


def command_with_json(args: list[str]) -> list[str]:
    command = ["jl", *args]
    separator = command.index("--") if "--" in command else len(command)
    command.insert(separator, "--json")
    return command


def jl(
    args: list[str], *, capture: bool = False, json_output: bool = False
) -> subprocess.CompletedProcess[str]:
    command = command_with_json(args) if json_output else ["jl", *args]
    completed = subprocess.run(command, check=False, capture_output=capture, text=True)
    if completed.returncode == 0:
        return completed
    detail = _jl_error_detail(completed) if capture else ""
    raise RuntimeError(detail or f"jl {' '.join(args)} failed with exit {completed.returncode}")


def _jl_error_detail(completed: subprocess.CompletedProcess[str]) -> str:
    payload = _decode_json(completed.stdout) if completed.stdout else None
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return (completed.stderr or completed.stdout or "").strip()


def jl_json(args: list[str]) -> Any:
    completed = jl(args, capture=True, json_output=True)
    payload = _decode_json(completed.stdout)
    if payload is None:
        raise RuntimeError(f"jl {' '.join(args)} returned empty JSON")
    return payload


def require_ssh_identity() -> None:
    candidates = [Path.home() / ".ssh" / name for name in ("id_ed25519", "id_rsa", "id_ecdsa")]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise RuntimeError(
            "no ~/.ssh/id_ed25519 or id_rsa; generate a key and run jl ssh-key add <key>.pub"
        )
    listed = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True)
    if listed.returncode == 0:
        return
    locked = [
        path
        for path in existing
        if b"ENCRYPTED" in path.read_bytes()[:120] or b"aes256" in path.read_bytes()[:120]
    ]
    if locked:
        raise RuntimeError(f"unlock your SSH key first: ssh-add {locked[0]}")


def require_jl() -> None:
    if shutil.which("jl") is None:
        raise RuntimeError(
            "jl is not on PATH; install with `uv tool install jarvislabs` then `jl setup`"
        )


def machine_id_from(payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("machine_id") is not None:
        return str(payload["machine_id"])
    raise RuntimeError(f"instance id missing from jl output: {payload!r}")


def instance_status(machine_id: str) -> str:
    payload = jl_json(["get", machine_id])
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected jl get payload: {payload!r}")
    return str(payload.get("status") or "")


def ensure_running(machine_id: str) -> str:
    status = instance_status(machine_id).lower()
    if status == "running":
        return machine_id
    if status != "paused":
        raise RuntimeError(
            f"instance {machine_id} is {status or 'unknown'}; resume or create a new one"
        )
    payload = jl_json(["resume", machine_id, "--yes"])
    if isinstance(payload, dict) and payload.get("machine_id"):
        return str(payload["machine_id"])
    return machine_id


def create_instance(args: argparse.Namespace) -> str:
    create = ["create", "--storage", str(args.storage), "--name", args.name, "--yes"]
    if args.region:
        create.extend(["--region", args.region])
    if args.cpu:
        create.extend(["--vm", "--cpu", "--vcpus", str(args.vcpus), "--ram", str(args.ram)])
    else:
        create.extend(["--gpu", args.gpu, "--template", "pytorch"])
    return machine_id_from(jl_json(create))


def remote_home(machine_id: str) -> str:
    completed = jl(["exec", machine_id, "--", "sh", "-lc", 'printf %s "$HOME"'], capture=True)
    home = (completed.stdout or "").strip()
    if not home.startswith("/"):
        raise RuntimeError(f"could not read remote HOME for {machine_id}")
    return home


def upload_env_file(
    machine_id: str, remote_root: str, filename: str, values: dict[str, str]
) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="tunelm-env-", delete=False
    ) as handle:
        for key, value in values.items():
            handle.write(f"{key}={shlex.quote(value)}\n")
        temp_path = Path(handle.name)
    try:
        temp_path.chmod(0o600)
        jl(["upload", machine_id, str(temp_path), f"{remote_root}/{filename}"])
    finally:
        temp_path.unlink(missing_ok=True)


def setup_remote(machine_id: str, remote_root: str, config: str) -> None:
    script = f"{remote_root}/scripts/jarvislabs/remote_setup.sh"
    jl(
        [
            "exec",
            machine_id,
            "--",
            "sh",
            "-lc",
            f"chmod +x {shlex.quote(script)} && {shlex.quote(script)} {shlex.quote(config)}",
        ]
    )


def remote_train_shell(remote_root: str, config: str) -> str:
    command = [".venv/bin/python", "scripts/train.py", "--config", config]
    quoted = " ".join(shlex.quote(part) for part in command)
    return (
        f"set -euo pipefail; cd {shlex.quote(remote_root)}; "
        "if [ -f .remote.env ]; then set -a; . ./.remote.env; set +a; fi; "
        "mkdir -p checkpoints results; "
        f"exec {quoted}"
    )


def start_train(machine_id: str, remote_root: str, config: str) -> str:
    shell = remote_train_shell(remote_root, config)
    payload = jl_json(
        ["run", "--on", machine_id, "--no-follow", "--yes", "--", "bash", "-lc", shell]
    )
    if not isinstance(payload, dict) or not payload.get("run_id"):
        raise RuntimeError(f"jl run did not return a run_id: {payload!r}")
    return str(payload["run_id"])


def follow_run(run_id: str) -> bool:
    try:
        completed = subprocess.run(["jl", "run", "logs", run_id, "--follow"], check=False)
    except KeyboardInterrupt:
        return True
    if completed.returncode in DETACH_EXIT_CODES:
        return True
    if completed.returncode != 0:
        raise RuntimeError(f"jl run logs {run_id} failed with exit {completed.returncode}")
    return False


def run_state(run_id: str) -> str:
    payload = jl_json(["run", "status", run_id])
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("state") or payload.get("status") or "").lower()


def fetch_results(machine_id: str, remote_root: str, root: Path) -> None:
    jl(["download", machine_id, f"{remote_root}/checkpoints", str(root), "-r"])
    jl(["download", machine_id, f"{remote_root}/results", str(root), "-r"])


def stop_instance(machine_id: str, *, destroy: bool) -> None:
    if destroy:
        jl_json(["destroy", machine_id, "--yes"])
        return
    jl_json(["pause", machine_id, "--yes"])


def resolve_machine_id(args: argparse.Namespace, root: Path) -> str:
    if args.machine_id:
        return args.machine_id
    session = load_session(root)
    if session.get("machine_id"):
        return session["machine_id"]
    raise RuntimeError("pass --on MACHINE_ID or run `cloud_train.py run` first")


def _validate_cloud_credentials(config: dict) -> None:
    report_to = config.get("training", {}).get("report_to")
    if report_to == "wandb" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required because training.report_to=wandb")
    model_name = str(config.get("model", {}).get("name", ""))
    has_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if model_name.startswith("google/gemma") and not has_hf_token:
        raise RuntimeError("HF_TOKEN is required to download Gemma on a fresh instance")


def _validate_cloud_adapter(config: dict, args: argparse.Namespace, root: Path) -> None:
    adapter = config.get("model", {}).get("adapter")
    if adapter and not (root / adapter).exists() and not args.machine_id:
        raise FileNotFoundError(
            f"SFT adapter not found: {adapter}. Run/fetch SFT first or reuse its instance with --on."
        )


def validate_run(args: argparse.Namespace, root: Path) -> None:
    if not args.machine_id and not args.gpu and not args.cpu:
        raise RuntimeError("pass --gpu TYPE, --cpu, or --on MACHINE_ID")
    if Path(args.config).is_absolute():
        raise RuntimeError(
            "pass a repo-relative config path, for example configs/gemma4_4b/sft.yaml"
        )
    config_path = root / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"training config not found: {args.config}")
    config = load_config(config_path)
    _validate_cloud_credentials(config)
    _validate_cloud_adapter(config, args, root)


def cmd_self_check(args: argparse.Namespace) -> int:
    root = repo_root()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / REMOTE_DIRNAME
        stage_payload(root, dest, args.config)
        config = load_config(root / args.config)
        train_file = dest / config["data"]["train_file"]
        required = [
            dest / "pyproject.toml",
            dest / "src" / "tunelm",
            dest / "configs" / "gemma4_4b" / "smoke_sft.yaml",
            dest / "configs" / "qwen3_06b" / "sft.yaml",
            dest / "scripts" / "jarvislabs" / "remote_setup.sh",
            train_file,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError("staged payload missing: " + ", ".join(missing))
    print("self-check ok")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    require_jl()
    require_ssh_identity()
    root = repo_root()
    load_local_env(root)
    validate_run(args, root)
    machine_id = provision_machine(args, root)
    leave_running = False
    try:
        remote_root = sync_code(machine_id, root, args)
        run_id = start_train(machine_id, remote_root, args.config)
        save_session(
            root,
            machine_id=machine_id,
            remote_root=remote_root,
            run_id=run_id,
            config=args.config,
        )
        print(f"run {run_id}")
        if args.detach:
            leave_running = True
            print(f"follow with: jl run logs {run_id} --follow")
            return 0
        detached = follow_run(run_id)
        state = run_state(run_id)
        if detached and state not in {"succeeded", "failed"}:
            leave_running = True
            print(f"detached; training still running on {machine_id} as {run_id}")
            return 0
        if state != "succeeded":
            raise RuntimeError(f"managed run {run_id} finished with state {state or 'unknown'}")
        fetch_results(machine_id, remote_root, root)
        print(f"downloaded checkpoints/ and results/ to {root}")
    finally:
        if not args.keep and not leave_running:
            stop_instance(machine_id, destroy=args.destroy)
            print(f"instance {machine_id} {'destroyed' if args.destroy else 'paused'}")
    return 0


def provision_machine(args: argparse.Namespace, root: Path) -> str:
    machine_id = ensure_running(args.machine_id) if args.machine_id else create_instance(args)
    print(f"instance {machine_id}")
    save_session(root, machine_id=machine_id)
    return machine_id


def sync_code(machine_id: str, root: Path, args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / REMOTE_DIRNAME
        stage_payload(root, staging, args.config)
        jl(["upload", machine_id, str(staging)])
    remote_root = f"{remote_home(machine_id)}/{REMOTE_DIRNAME}"
    save_session(root, machine_id=machine_id, remote_root=remote_root)
    env_values = {
        key: value
        for key in (
            "WANDB_API_KEY",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_RUN_GROUP",
            "WANDB_MODE",
            "WANDB_LOG_MODEL",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
        )
        if (value := os.environ.get(key))
    }
    env_values.setdefault("WANDB_PROJECT", "tunelm")
    if env_values:
        upload_env_file(machine_id, remote_root, ".remote.env", env_values)
    if not args.skip_setup:
        setup_remote(machine_id, remote_root, args.config)
    return remote_root


def cmd_fetch(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = ensure_running(resolve_machine_id(args, root))
    session = load_session(root)
    remote_root = (
        args.remote_root
        or session.get("remote_root")
        or f"{remote_home(machine_id)}/{REMOTE_DIRNAME}"
    )
    save_session(root, machine_id=machine_id, remote_root=remote_root)
    fetch_results(machine_id, remote_root, root)
    print(f"downloaded checkpoints/ and results/ to {root}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = resolve_machine_id(args, root)
    stop_instance(machine_id, destroy=args.destroy)
    print(f"instance {machine_id} {'destroyed' if args.destroy else 'paused'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    require_jl()
    root = repo_root()
    machine_id = resolve_machine_id(args, root)
    session = load_session(root)
    print(
        json.dumps(
            {"machine_id": machine_id, **session, "status": instance_status(machine_id)}, indent=2
        )
    )
    if session.get("run_id"):
        jl(["run", "status", session["run_id"]])
    return 0


def _add_on_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--on", dest="machine_id", help="existing JarvisLabs machine id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TuneLM on a JarvisLabs GPU or CPU instance")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create or reuse an instance and run one training config")
    _add_on_flag(run)
    run.add_argument("--gpu", help="GPU type from `jl gpus` (for example L4 or A100)")
    run.add_argument(
        "--cpu", action="store_true", help="create a CPU VM instead of a GPU container"
    )
    run.add_argument("--vcpus", type=int, default=8)
    run.add_argument("--ram", type=int, default=32)
    run.add_argument("--storage", type=int, default=100)
    run.add_argument("--name", default="tunelm-train")
    run.add_argument("--region", help="IN1, IN2, or EU1")
    run.add_argument(
        "--config",
        default="configs/gemma4_4b/smoke_sft.yaml",
        help="repo-relative SFT or GRPO config",
    )
    run.add_argument("--detach", action="store_true")
    run.add_argument("--keep", action="store_true")
    run.add_argument("--destroy", action="store_true")
    run.add_argument("--skip-setup", action="store_true")

    fetch = commands.add_parser("fetch", help="download remote checkpoints/results")
    _add_on_flag(fetch)
    fetch.add_argument("--remote-root")

    down = commands.add_parser("down", help="pause or destroy the instance")
    _add_on_flag(down)
    down.add_argument("--destroy", action="store_true")

    status = commands.add_parser("status", help="print saved session and instance status")
    _add_on_flag(status)

    self_check = commands.add_parser(
        "self-check", help="stage a config-specific payload and verify required files"
    )
    self_check.add_argument("--config", default="configs/gemma4_4b/smoke_sft.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "self-check": cmd_self_check,
        "run": cmd_run,
        "fetch": cmd_fetch,
        "down": cmd_down,
        "status": cmd_status,
    }
    try:
        return commands[args.command](args)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
