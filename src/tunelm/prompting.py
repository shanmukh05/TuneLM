"""Schedule end-user request styles independently of composition task types.

Input-author briefs come from YAML and are frozen before any music is composed.
The author invents the situation. No scene list is supplied.
"""

import re

from pydantic import Field, model_validator

from tunelm.schemas import PromptStyle, Record


class PromptConfig(Record):
    """A repeatable style schedule for request wording, not a list of musical situations."""

    schedule: list[PromptStyle] = Field(min_length=1)
    styles: dict[PromptStyle, str]
    writing_guidance: str = ""
    openings: list[str] = Field(default_factory=list)
    focuses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def defined_styles(self):
        if set(self.schedule) - self.styles.keys():
            raise ValueError("Every scheduled prompt style needs instructions")
        if any(not instruction.strip() for instruction in self.styles.values()):
            raise ValueError("Prompt style instructions cannot be blank")
        if any(not item.strip() for item in [*self.openings, *self.focuses]):
            raise ValueError("Openings and focuses cannot be blank")
        return self

    def brief(self, index: int) -> dict:
        """Give an input author a writing style, without a scene or an answer plan."""
        style = self.schedule[index % len(self.schedule)]
        brief = {
            "style": style,
            "instructions": self.styles[style],
            "writing_guidance": self.writing_guidance,
        }
        if self.openings:
            brief["opening"] = self.openings[index % len(self.openings)]
        if self.focuses:
            brief["focus"] = self.focuses[(index * 3 + 1) % len(self.focuses)]
        return brief


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
