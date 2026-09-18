"""Author natural-language SFT compose prompts from procedural briefs.

Each compose task samples a 1–4 complexity level, builds a verifier-friendly
brief, and asks a configured provider to paraphrase it. Constraints always come
from the brief, not from the prompt author.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from tunelm.data.vocab import VocabStore, load_vocab
from tunelm.providers.base import ProviderRouter
from tunelm.providers.provenance import provider_provenance
from tunelm.schemas import ModelProvenance, TaskConstraints, TempoRange
from tunelm.models.prompt_author_templates import compose_level_system_prompt
from tunelm.strudel.parser import ResponseParseError, parse_json_object

MELODIC_INSTRUMENTS = (
    "piano",
    "guitar",
    "flute",
    "violin",
    "strings",
    "synth",
    "pad",
    "pluck",
    "organ",
    "harp",
    "cello",
    "saxophone",
    "trumpet",
    "choir",
    "marimba",
    "vibraphone",
    "accordion",
    "nylon-guitar",
    "banjo",
)
LIGHT_PERCUSSION = ("hh", "shaker", "rim", "tambourine", "cp", "oh")

PROMPT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "additionalProperties": False,
}

LEVEL_4_HINTS = (
    "Open with a textural percussion bed, introduce a melody, brighten the mood, "
    "then return to the opening motif.",
    "Begin minimally, build layers across sections, peak in the middle, "
    "and resolve back to the first theme.",
    "Start with rain-like percussion, add a melancholic melody, shift to an "
    "uplifting passage, then revisit the opening motif.",
)


@dataclass(slots=True)
class ComposeBrief:
    """Structured compose spec used for constraints and prompt authoring."""

    level: int
    mood: str
    instruments: list[str]
    tempo: tuple[int, int]
    genre: str | None = None
    rhythm: str | None = None
    structure: str | None = None
    texture: str | None = None
    hints: list[str] = field(default_factory=list)

    def to_author_payload(self) -> dict[str, Any]:
        """Scene hints for the prompt author; tempo/instruments stay in constraints only."""
        payload: dict[str, Any] = {"level": self.level, "mood": self.mood}
        if self.genre:
            payload["setting"] = self.genre
        if self.texture:
            payload["atmosphere"] = self.texture
        if self.hints:
            payload["creative_hints"] = self.hints
        return payload


def sample_compose_level(rng: random.Random, config: dict[str, Any] | None = None) -> int:
    """Sample a compose prompt complexity level, biased toward vaguer scene prompts."""
    config = config or {}
    weights = config.get("prompt_level_weights", {1: 0.30, 2: 0.30, 3: 0.25, 4: 0.15})
    levels = (1, 2, 3, 4)
    values = [float(weights.get(level, weights.get(str(level), 0.25))) for level in levels]
    return rng.choices(levels, weights=values, k=1)[0]


def prompt_author_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the provider config slice used for compose prompt authoring."""
    author = config.get("prompt_author")
    if author:
        return {**config, **author}
    return config


def sample_compose_brief(
    rng: random.Random, level: int, vocab: VocabStore | None = None
) -> ComposeBrief:
    """Sample a level-scaled brief from repository vocabularies."""
    vocab = vocab or load_vocab()
    mood = rng.choice(vocab.moods)
    tempo = rng.choice(vocab.tempo_ranges)
    if level == 1:
        instruments = [rng.choice(MELODIC_INSTRUMENTS)]
        return ComposeBrief(level=level, mood=mood, instruments=instruments, tempo=tempo)
    if level == 2:
        instruments = [rng.choice(MELODIC_INSTRUMENTS)]
        if rng.random() < 0.8:
            percussion = rng.choice(LIGHT_PERCUSSION)
            if percussion not in instruments:
                instruments.append(percussion)
        return ComposeBrief(level=level, mood=mood, instruments=instruments, tempo=tempo)
    if level == 3:
        count = rng.randint(2, 3)
        pool = list(dict.fromkeys((*MELODIC_INSTRUMENTS, *LIGHT_PERCUSSION)))
        instruments = rng.sample(pool, k=count)
        return ComposeBrief(
            level=level,
            mood=mood,
            instruments=instruments,
            tempo=tempo,
            rhythm=rng.choice(vocab.rhythms),
            structure=rng.choice(vocab.structures),
            texture=rng.choice(vocab.textures),
            hints=[
                rng.choice(
                    (
                        "introduce sparse drums after a short opening",
                        "gradually increase drum density",
                        "add a supporting layer as the piece develops",
                    )
                )
            ],
        )
    count = rng.randint(2, min(4, len(vocab.instruments)))
    instruments = rng.sample(vocab.instruments, k=count)
    return ComposeBrief(
        level=level,
        mood=mood,
        instruments=instruments,
        tempo=tempo,
        genre=rng.choice(vocab.genres),
        rhythm=rng.choice(vocab.rhythms),
        structure=rng.choice(vocab.structures),
        texture=rng.choice(vocab.textures),
        hints=[rng.choice(LEVEL_4_HINTS)],
    )


def brief_to_constraints(brief: ComposeBrief) -> TaskConstraints:
    """Map a brief to explicit constraints used by validation and rewards."""
    min_layers = 1 if brief.level <= 2 else min(2, len(brief.instruments))
    return TaskConstraints(
        tempo=TempoRange(min=brief.tempo[0], max=brief.tempo[1]),
        required_instruments=list(brief.instruments),
        mood=[brief.mood],
        min_layers=min_layers,
    )


def author_compose_prompt(
    brief: ComposeBrief,
    router: ProviderRouter,
    *,
    temperature: float = 0.9,
    max_tokens: int = 256,
    max_attempts: int = 3,
) -> tuple[str, ModelProvenance]:
    """Ask a configured provider to paraphrase one brief into a musician prompt."""
    messages = [
        {"role": "system", "content": compose_level_system_prompt(brief.level)},
        {
            "role": "user",
            "content": json.dumps(brief.to_author_payload(), ensure_ascii=False),
        },
    ]
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            provider = router.choose()
            text = provider.generate(
                messages,
                response_schema=PROMPT_RESPONSE_SCHEMA,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not text.strip():
                raise ValueError("prompt author returned empty response")
            payload = parse_json_object(text)
            prompt = str(payload.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("prompt author returned empty prompt")
            return prompt, provider_provenance(provider)
        except (ResponseParseError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"prompt author failed after {max_attempts} attempts: {last_error}")
