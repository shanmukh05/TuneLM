from __future__ import annotations

from typing import Any


def attach_lora(model, config: dict[str, Any]):
    lora = config.get("lora", {})
    if not lora.get("enabled", True) or getattr(model, "_tunelm_adapter_loaded", False):
        return model
    if config.get("training", {}).get("backend") == "unsloth":
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError("Install the optional backend: pip install -e '.[unsloth]'") from exc
        return FastLanguageModel.get_peft_model(
            model,
            r=int(lora.get("rank", 32)),
            lora_alpha=int(lora.get("alpha", 64)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            bias="none",
            target_modules=lora.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            use_gradient_checkpointing="unsloth",
            random_state=int(config.get("training", {}).get("seed", 42)),
        )
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Install TuneLM training extras: pip install -e '.[train]'") from exc
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
