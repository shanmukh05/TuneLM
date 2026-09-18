"""Load causal language models and optional PEFT adapters."""

from __future__ import annotations

from typing import Any

from tunelm.config import project_path


def _quantization_config(model_cfg: dict[str, Any], torch, config_class):
    if not model_cfg.get("load_in_4bit", False):
        return None
    return config_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.get("compute_dtype", "bfloat16")),
    )


def _attach_adapter(model, model_cfg: dict[str, Any], for_training: bool):
    adapter = model_cfg.get("adapter")
    if not adapter:
        return model
    try:
        from peft import PeftModel, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
    if for_training and getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    adapter_path = project_path(adapter)
    adapter_source = str(adapter_path) if adapter_path.exists() else adapter
    model = PeftModel.from_pretrained(model, adapter_source, is_trainable=for_training)
    model._tunelm_adapter_loaded = True
    return model


def _load_transformers(model_cfg: dict[str, Any], for_training: bool):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc

    name = model_cfg["name"]
    quantization = _quantization_config(model_cfg, torch, BitsAndBytesConfig)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quantization,
        device_map="auto" if for_training else model_cfg.get("device_map", "auto"),
        dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=False,
    )
    return _attach_adapter(model, model_cfg, for_training), tokenizer


def load_model_and_tokenizer(config: dict[str, Any], *, for_training: bool = True):
    """Load the configured model, tokenizer, quantization, and optional adapter."""
    model_cfg = config.get("model", config)
    return _load_transformers(model_cfg, for_training)
