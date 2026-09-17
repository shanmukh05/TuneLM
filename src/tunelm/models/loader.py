"""Load configured language models without importing training dependencies eagerly.

The public loader selects the existing Transformers or Unsloth path; adapter
attachment remains local to this module.
"""

from __future__ import annotations

from typing import Any

from tunelm.config import project_path


def _load_unsloth(model_cfg: dict[str, Any]):
    if model_cfg.get("model_class") == "multimodal":
        raise ValueError("The TuneLM Unsloth path currently supports text-only models")
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Install the optional backend: pip install -e '.[unsloth]'") from exc
    return FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=int(model_cfg.get("max_seq_length", 4096)),
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )


def _quantization_config(model_cfg: dict[str, Any], torch, config_class):
    if not model_cfg.get("load_in_4bit", False):
        return None
    return config_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("compute_dtype", "bfloat16")),
    )


def _multimodal_model_class(transformers):
    try:
        return transformers.AutoModelForMultimodalLM
    except AttributeError as exc:
        raise RuntimeError(
            "This Gemma 4 config requires a Transformers build with "
            "AutoModelForMultimodalLM support"
        ) from exc


def _attach_adapter(model, model_cfg: dict[str, Any], for_training: bool):
    adapter = model_cfg.get("adapter")
    if not adapter:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
    adapter_path = project_path(adapter)
    adapter_source = str(adapter_path) if adapter_path.exists() else adapter
    model = PeftModel.from_pretrained(model, adapter_source, is_trainable=for_training)
    model._tunelm_adapter_loaded = True
    return model


def _load_transformers(model_cfg: dict[str, Any], for_training: bool):
    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc

    name = model_cfg["name"]
    quantization = _quantization_config(model_cfg, torch, BitsAndBytesConfig)
    is_multimodal = model_cfg.get("model_class") == "multimodal"
    tokenizer_class = AutoProcessor if is_multimodal else AutoTokenizer
    model_class = _multimodal_model_class(transformers) if is_multimodal else AutoModelForCausalLM
    tokenizer = tokenizer_class.from_pretrained(name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(
        name,
        quantization_config=quantization,
        device_map="auto" if for_training else model_cfg.get("device_map", "auto"),
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=False,
    )
    return _attach_adapter(model, model_cfg, for_training), tokenizer


def load_model_and_tokenizer(config: dict[str, Any], *, for_training: bool = True):
    """Load the configured model, tokenizer, quantization, and optional adapter."""
    model_cfg = config.get("model", config)
    if config.get("training", {}).get("backend") == "unsloth":
        return _load_unsloth(model_cfg)
    return _load_transformers(model_cfg, for_training)
