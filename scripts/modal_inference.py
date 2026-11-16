"""Serve a merged TuneLM checkpoint through Python and authenticated HTTP."""

import os

import modal
from pydantic import BaseModel, Field, model_validator

from tunelm.composition.responses import generate_completion, render_conversation
from tunelm.core.io import load_config
from tunelm.core.schemas import Record


class InferenceConfig(Record):
    model: str = Field(pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    revision: str = Field(default="main", min_length=1)
    gpu: str | list[str] = "L4"
    hf_secret: str | None = Field(default=None, min_length=1)
    trust_remote_code: bool = False
    max_length: int = Field(default=6144, ge=128)
    max_new_tokens: int = Field(default=3072, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)

    @model_validator(mode="after")
    def check_token_budget(self):
        if self.max_new_tokens >= self.max_length:
            raise ValueError("max_new_tokens must leave room for the prompt")
        return self


class ConversationTurn(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    code: str = Field(min_length=1, max_length=24000)


SETTINGS_ENV = "TUNELM_INFERENCE_SETTINGS"
if modal.is_local():
    config = InferenceConfig.model_validate(
        load_config(os.getenv("TUNELM_INFERENCE_CONFIG", "configs/inference.yaml"))
    )
else:
    config = InferenceConfig.model_validate_json(os.environ[SETTINGS_ENV])

secrets = (
    [modal.Secret.from_name(config.hf_secret, required_keys=["HF_TOKEN"])]
    if config.hf_secret
    else []
)

app = modal.App("tunelm-sft-inference")
cache = modal.Volume.from_name("tunelm-inference-hf-cache", create_if_missing=True)
base_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "pydantic>=2.10,<3", "PyYAML>=6,<7"
)
image = (
    base_image
    .apt_install("gcc", "libc6-dev")
    .uv_pip_install("torch==2.15.0", "transformers==5.20.2")
    .add_local_python_source("tunelm")
)
web_image = base_image.uv_pip_install("fastapi").add_local_python_source("tunelm")


@app.cls(
    image=image,
    gpu=config.gpu,
    env={SETTINGS_ENV: config.model_dump_json(), "HF_HUB_CACHE": "/hf-cache"},
    volumes={"/hf-cache": cache},
    secrets=secrets,
    max_containers=1,
    scaledown_window=60,
    startup_timeout=900,
    timeout=300,
)
class Composer:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        options = {"revision": config.revision, "trust_remote_code": config.trust_remote_code}
        print(f"Loading {config.model} at {config.revision}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config.model, dtype=torch.bfloat16, attn_implementation="sdpa", **options
            )
            .to("cuda")
            .eval()
        )
        cache.commit()
        print(f"Ready: {config.model}", flush=True)

    @modal.method()
    def predict(self, prompt: str) -> dict:
        return self._compose(prompt, [])

    @modal.method()
    def converse(self, prompt: str, history: list[dict[str, str]]) -> dict:
        return self._compose(prompt, history)

    def _compose(self, prompt: str, history: list[dict[str, str]]) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A listener prompt is required")
        rendered = render_conversation(
            prompt.strip(), history, self.tokenizer, config.max_length - config.max_new_tokens
        )
        completion, usage = generate_completion(
            rendered,
            self.model,
            self.tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_length=config.max_length,
        )
        return {"model": config.model, "completion": completion, "usage": usage}


@app.function(image=web_image, env={SETTINGS_ENV: config.model_dump_json()})
@modal.asgi_app(requires_proxy_auth=True)
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    class CompositionRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=1000)
        history: list[ConversationTurn] = Field(default_factory=list, max_length=6)

    service = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @service.post("/jobs", status_code=202)
    async def start_composition(body: CompositionRequest):
        prompt = body.prompt.strip()
        if not prompt:
            raise HTTPException(422, "Describe the music you want to hear.")
        history = [turn.model_dump() for turn in body.history]
        call = await Composer().converse.spawn.aio(prompt, history)
        return {"job_id": call.object_id}

    @service.get("/jobs/{job_id}")
    async def get_composition(job_id: str):
        try:
            call = modal.FunctionCall.from_id(job_id)
            return await call.get.aio(timeout=0)
        except (
            modal.exception.NotFoundError,
            modal.exception.InvalidError,
            modal.exception.OutputExpiredError,
        ):
            raise HTTPException(404, "Composition job not found.") from None
        except modal.exception.FunctionTimeoutError:
            raise HTTPException(504, "Composition timed out.") from None
        except (TimeoutError, modal.exception.TimeoutError):
            return JSONResponse({"status": "pending"}, status_code=202)

    return service
