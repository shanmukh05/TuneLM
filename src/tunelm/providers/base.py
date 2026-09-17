from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


Message = Mapping[str, str]


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate one response as text."""


@dataclass(slots=True)
class ProviderChoice:
    provider: LLMProvider
    weight: float = 1.0


class ProviderRouter:
    def __init__(
        self,
        choices: list[ProviderChoice],
        *,
        strategy: str = "weighted",
        seed: int = 42,
    ) -> None:
        if not choices:
            raise ValueError("At least one provider must be enabled")
        if strategy not in {"weighted", "round_robin", "random", "single_provider"}:
            raise ValueError(f"Unknown provider strategy: {strategy}")
        self.choices = choices
        self.strategy = strategy
        self.random = random.Random(seed)
        self.index = 0

    def choose(self) -> LLMProvider:
        if self.strategy == "single_provider":
            return self.choices[0].provider
        if self.strategy == "round_robin":
            provider = self.choices[self.index % len(self.choices)].provider
            self.index += 1
            return provider
        if self.strategy == "random":
            return self.random.choice(self.choices).provider
        return self.random.choices(
            [choice.provider for choice in self.choices],
            weights=[choice.weight for choice in self.choices],
            k=1,
        )[0]
