"""Modal web endpoint for TuneLM inference from a merged Hugging Face model repo."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal
from pydantic import BaseModel

from tunelm.inference_limits import DEFAULT_MAX_NEW_TOKENS, MAX_NEW_TOKENS_CAP

HF_REPO = os.environ.get("TUNELM_HF_REPO", "")
HF_REVISION = os.environ.get("TUNELM_HF_REVISION") or None
RUN_VALIDATION = os.environ.get("TUNELM_VALIDATE", "0") == "1"


def _local_tunelm_src() -> str | None:
    """Return repo src/tunelm when building locally; None on Modal's /root/app.py mount."""
    here = Path(__file__).resolve()
    try:
        candidate = here.parents[2] / "src" / "tunelm"
    except IndexError:
        return None
    return str(candidate) if candidate.is_dir() else None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "accelerate>=1.12,<2",
        "datasets>=4.4,<5",
        "peft>=0.19,<1",
        "transformers>=5.8,<6",
        "torch",
        "huggingface_hub",
        "pydantic>=2.10,<3",
        "fastapi[standard]",
    )
    .env({"PYTHONPATH": "/root"})
)
_tunelm_src = _local_tunelm_src()
if _tunelm_src is not None:
    image = image.add_local_dir(_tunelm_src, remote_path="/root/tunelm", copy=True)

app = modal.App("tunelm")
secret = modal.Secret.from_name("tunelm")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.7
    run_validation: bool = False


def _generate_parsed(
    *,
    messages: list[dict[str, str]],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
):
    from tunelm.inference import generate_messages
    from tunelm.strudel.parser import ResponseParseError, parse_model_response

    completion = generate_messages(
        messages,
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    try:
        return parse_model_response(completion), completion
    except ResponseParseError as first_exc:
        if max_new_tokens >= MAX_NEW_TOKENS_CAP:
            retry_exc = ResponseParseError(f"{first_exc} (raw completion preserved)")
            retry_exc.raw_completion = completion  # ponytail: attach for 422 payload
            raise retry_exc from first_exc
        retry_tokens = min(max_new_tokens * 2, MAX_NEW_TOKENS_CAP)
        retry_completion = generate_messages(
            messages,
            model,
            tokenizer,
            max_new_tokens=retry_tokens,
            temperature=temperature,
        )
        try:
            return parse_model_response(retry_completion), retry_completion
        except ResponseParseError as second_exc:
            retry_exc = ResponseParseError(f"{second_exc} (raw completion preserved)")
            retry_exc.raw_completion = retry_completion
            raise retry_exc from second_exc


@app.cls(
    gpu="A10G",
    image=image,
    secrets=[secret],
    timeout=900,
    scaledown_window=300,
)
class TuneLMInference:
    @modal.enter()
    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tunelm.models.templates import SYSTEM_PROMPT

        repo_id = os.environ.get("TUNELM_HF_REPO") or HF_REPO
        if not repo_id:
            raise RuntimeError("set TUNELM_HF_REPO to a merged Hugging Face model repo id")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        revision = os.environ.get("TUNELM_HF_REVISION") or HF_REVISION
        kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "token": token,
        }
        if revision:
            kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            dtype="auto",
            **kwargs,
        )
        self.model.eval()
        self.system_prompt = SYSTEM_PROMPT

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        from fastapi import HTTPException

        from tunelm.strudel.executor import StrudelExecutor
        from tunelm.strudel.parser import ResponseParseError

        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        run_validation = request.run_validation or RUN_VALIDATION
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            response, _completion = _generate_parsed(
                messages=messages,
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
        except ResponseParseError as exc:
            raw_completion = getattr(exc, "raw_completion", None)
            raise HTTPException(
                status_code=422,
                detail={
                    "detail": str(exc),
                    "raw_completion": raw_completion,
                    "hint": f"Increase max_new_tokens (current {request.max_new_tokens}, cap {MAX_NEW_TOKENS_CAP})",
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        output = response.model_dump()
        if run_validation:
            output["validation"] = StrudelExecutor(backend="static").run(response.strudel_code).model_dump()
        return output


@app.local_entrypoint()
def smoke(prompt: str = "slow piano at dawn") -> None:
    import urllib.error
    import urllib.request

    url = os.environ.get(
        "TUNELM_MODAL_URL",
        "https://venkatashanmukhasainathg--tunelm-tunelminference-generate.modal.run",
    )
    payload = json.dumps(
        {
            "prompt": prompt,
            "run_validation": False,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temperature": 0.7,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(detail or str(error)) from error
    print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
