from __future__ import annotations

import random
from collections.abc import Iterator

from tunelm.data.prompt_builder import (
    build_rl_compose_prompt,
    build_sft_prompt,
)
from tunelm.data.vocab import load_vocab
from tunelm.schemas import ComposeTask, TaskConstraints, TempoRange


def composition_tasks(count: int, seed: int = 42) -> Iterator[ComposeTask]:
    rng = random.Random(seed)
    vocab = load_vocab()
    instruments = vocab.instruments
    seen: set[tuple] = set()
    index = 0
    while index < count:
        selected = sorted(rng.sample(instruments, rng.randint(2, 5)))
        bpm = rng.randrange(60, 151, 5)
        tolerance = 0 if index % 2 == 0 else 5
        tempo = bpm if not tolerance else TempoRange(min=bpm - tolerance, max=bpm + tolerance)
        layers = rng.randint(1, min(4, len(selected)))
        remaining = [name for name in instruments if name not in selected]
        forbidden = [rng.choice(remaining)] if index % 4 == 0 and remaining else []
        signature = (tuple(selected), bpm, tolerance, layers, tuple(forbidden))
        if signature in seen:
            continue
        seen.add(signature)

        if rng.random() < 0.65:
            mood = rng.choice(vocab.moods)
            structure = rng.choice(vocab.structures)
            rhythm = rng.choice(vocab.rhythms)
            texture = rng.choice(vocab.textures)
            genre = rng.choice(vocab.genres)
            tempo_range = (
                (tempo.min, tempo.max) if isinstance(tempo, TempoRange) else (bpm - 5, bpm + 5)
            )
            prompt = build_sft_prompt(
                rng,
                mood=mood,
                instruments=selected,
                tempo=tempo_range,
                structure=structure,
                rhythm=rhythm,
                texture=texture,
                genre=genre,
            )
            constraints = TaskConstraints(
                tempo=tempo,
                required_instruments=selected,
                forbidden_instruments=forbidden,
                mood=[mood],
                min_layers=layers,
            )
            tags = [mood, rhythm, texture, genre, "rich-compose"]
        else:
            forbidden_phrase = f" Do not use {forbidden[0]}." if forbidden else ""
            tempo_phrase = f"about {bpm} BPM" if tolerance else f"exactly {bpm} BPM"
            prompt = build_rl_compose_prompt(
                rng,
                tempo_phrase=tempo_phrase,
                instruments=selected,
                layers=layers,
                forbidden_phrase=forbidden_phrase,
            )
            constraints = TaskConstraints(
                tempo=tempo,
                required_instruments=selected,
                forbidden_instruments=forbidden,
                min_layers=layers,
            )
            tags = ["constraint-compose"]

        yield ComposeTask(
            id=f"rl-compose-{seed}-{index:07d}",
            prompt=prompt,
            constraints=constraints,
            difficulty=min(5, 1 + len(selected)),
            tags=tags,
        )
        index += 1
