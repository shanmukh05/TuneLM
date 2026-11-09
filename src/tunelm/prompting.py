"""Schedule end-user request styles independently of composition task types.

Private teacher briefs come from YAML and are frozen alongside generation jobs.
Only the style and scenario labels accompany exported SFT and RL examples.
"""

import re

from pydantic import Field, model_validator

from tunelm.io import load_config
from tunelm.schemas import PromptStyle, Record


class Scenario(Record):
    situation: str = Field(min_length=1)
    mood: str = Field(min_length=1)
    genres: list[str] = Field(min_length=1)


class PromptConfig(Record):
    """A repeatable style schedule and palette of musical situations, not user prompts."""

    schedule: list[PromptStyle] = Field(min_length=1)
    styles: dict[PromptStyle, str]
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def defined_styles(self):
        if set(self.schedule) - self.styles.keys():
            raise ValueError("Every scheduled prompt style needs instructions")
        if any(not instruction.strip() for instruction in self.styles.values()):
            raise ValueError("Prompt style instructions cannot be blank")
        return self

    def brief(self, index: int) -> dict:
        """Pair a style with a scene; unspecified implementation choices stay private."""
        style = self.schedule[index % len(self.schedule)]
        # Rotate after each schedule so a full run explores different style/scene pairings.
        cycle = index // len(self.schedule)
        scenario = self.scenarios[(index + cycle) % len(self.scenarios)]
        return {
            "style": style,
            "instructions": self.styles[style],
            "scenario": scenario.model_dump(exclude={"schema_version"}),
            "grounding": (
                "Use the scenario as musical direction, not a checklist to copy into the prompt. "
                "For recombine, adapt the composition to suit it within the plan. "
                "For backtranslate, describe only compatible audible qualities; "
                "do not claim an unsupported genre. "
                "For co-creation, make a natural request to develop the supplied existing music. "
                "Do not add BPM, key, instruments, or implementation requirements unless the "
                "assigned style calls for them. Put your own musical choices in decisions and code."
            ),
        }


def load_prompt_config(path: str | None) -> PromptConfig | None:
    """Opt into explicit style coverage without requiring it on legacy records."""
    return PromptConfig.model_validate(load_config(path)) if path else None


def validate_prompt_style(prompt: str, brief: dict | None) -> None:
    """Reject obvious technical leakage; genre/mood fidelity still needs listening."""
    if not brief or brief["style"] == "technical":
        return
    if re.search(r"\b(?:setcpm|stack|arrange|lpf|gain|scale|note|s|n)\s*\(", prompt, re.I):
        raise ValueError("This request style must not prescribe Strudel API calls")
    if brief["style"] == "musical":
        return
    precise = (
        r"\b\d+(?:\.\d+)?\s*(?:bpm|cpm|hz)\b|\b[A-G][#b]?[ :]+"
        r"(?:major|minor|dorian|mixolydian|phrygian|lydian|locrian)\b"
    )
    if re.search(precise, prompt, re.I):
        raise ValueError("This request style must leave exact tempo/key/effect settings open")
