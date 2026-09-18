from __future__ import annotations

from typing import Any


def attach_lora(model, config: dict[str, Any]):
    lora = config.get("lora", {})
    if not lora.get("enabled", True) or getattr(model, "_tunelm_adapter_loaded", False):
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=int(lora.get("rank", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    return get_peft_model(model, peft_config)
