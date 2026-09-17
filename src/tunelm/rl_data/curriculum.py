from __future__ import annotations

from collections.abc import Iterable, Iterator

from tunelm.schemas import RLTask


def curriculum_stage(tasks: Iterable[RLTask], stage: int) -> Iterator[RLTask]:
    """Yield tasks unlocked by a 1–5 curriculum stage."""
    if not 1 <= stage <= 5:
        raise ValueError("curriculum stage must be between 1 and 5")
    for task in tasks:
        if task.difficulty <= stage:
            yield task
