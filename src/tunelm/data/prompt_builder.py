from __future__ import annotations

import random
import re
from typing import Any

from tunelm.data.vocab import VocabStore, load_vocab


def _fill_template(template: str, slots: dict[str, Any]) -> str:
    result = template
    for key, value in slots.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def build_sft_prompt(
    rng: random.Random,
    *,
    mood: str,
    instruments: list[str],
    tempo: tuple[int, int],
    structure: str,
    rhythm: str,
    texture: str,
    genre: str | None = None,
    vocab: VocabStore | None = None,
) -> str:
    vocab = vocab or load_vocab()
    templates = vocab.compose_templates.get("templates", [])
    template = rng.choice(templates)
    return _fill_template(
        template,
        {
            "mood": mood,
            "instruments": ", ".join(instruments),
            "tempo_min": tempo[0],
            "tempo_max": tempo[1],
            "structure": structure,
            "rhythm": rhythm,
            "texture": texture,
            "genre": genre or rng.choice(vocab.genres),
        },
    )


def build_rl_compose_prompt(
    rng: random.Random,
    *,
    tempo_phrase: str,
    instruments: list[str],
    layers: int,
    forbidden_phrase: str,
    vocab: VocabStore | None = None,
) -> str:
    vocab = vocab or load_vocab()
    templates = vocab.compose_templates.get("rl_templates", [])
    template = rng.choice(templates)
    layer_suffix = "" if layers == 1 else "s"
    return _fill_template(
        template,
        {
            "tempo_phrase": tempo_phrase,
            "instruments": ", ".join(instruments),
            "layers": layers,
            "layer_suffix": layer_suffix,
            "forbidden_phrase": forbidden_phrase,
        },
    )


def build_repair_prompt(rng: random.Random, vocab: VocabStore | None = None) -> str:
    vocab = vocab or load_vocab()
    return rng.choice(vocab.repair_templates["prompt_templates"])


def normalize_prompt(text: str) -> str:
    lowered = text.casefold()
    lowered = re.sub(r"\d+", "N", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered
