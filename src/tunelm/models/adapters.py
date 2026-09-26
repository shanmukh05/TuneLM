from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def attach_lora(model, config: dict[str, Any]):
    lora = config.get("lora", {})
    if not lora.get("enabled", True) or getattr(model, "_tunelm_adapter_loaded", False):
        return model
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    lora_kwargs: dict[str, Any] = {
        "r": int(lora.get("rank", 32)),
        "lora_alpha": int(lora.get("alpha", 64)),
        "lora_dropout": float(lora.get("dropout", 0.05)),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    # Gemma 4 wraps some towers in Gemma4ClippableLinear; omit target_modules so
    # PEFT>=0.19 scopes LoRA to the language-model layers automatically.
    if lora.get("target_modules"):
        lora_kwargs["target_modules"] = lora["target_modules"]
    return get_peft_model(model, LoraConfig(**lora_kwargs))
