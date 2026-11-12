"""Schedule end-user request styles and define the SFT input-author stage.

Input-author briefs come from YAML and are frozen before any music is composed.
The author invents the situation. No scene list is supplied. `prompt_generation`
runs one `AuthorStage` per bank: `SFT_STAGE` here, `rl_prompts.RL_STAGE` for RL.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from tunelm.schemas import MusicRequest, PromptStyle, Record

PROMPT_SYSTEM = """Write ONE realistic end-user request for instrumental music.
Return JSON with schema_version=2, prompt, and original_code (null normally).
This is the input-authoring stage. Do NOT write the answer, an implementation plan,
musical decisions for the composer, or a finished composition. You have no source
composition to describe. The brief assigns a writing style, a category, and, when present, an opening and a
focus. Follow that opening and that focus. They are the variation for this request.
Invent the situation yourself. No situation list is provided. Choose a fresh reason
for the music on every request. The reason can be a few words.
When already_written is present, those requests are already accepted. Do not reuse
their situation or their musical ask.
Some requests are only a few words. Others are a few sentences. Do not make them
all long. Vary the emotion too: joy, tenderness, tension, play, grief, calm,
unease, anger, warmth, excitement. Do not make the bank mostly cozy or nostalgic.
Do not recycle a scene you have used before, and do not fall back on a stock setup.
The listener may be vague or specify several preferences. Leave genuine choices
open; do not add a BPM/key/instrument inventory just to aid validation.
Follow the assigned request style and category. Arrangement requests should ask
for development in natural language. Atomic requests ask for one part. Only for
co-creation, provide a short original Strudel starting pattern as original_code,
and ask to develop it; that starting pattern is input material, not the answer.
No real artist imitation, promised vocals, marketing copy, or fabricated API names.
"""


class PromptReply(Record):
    prompt: str = Field(min_length=1)
    original_code: str | None = None


@dataclass(frozen=True)
class AuthorStage:
    """Everything that differs between input banks; the authoring engine is shared.

    `payload` adds fields to every author request. `check` raises ValueError with
    feedback the author sees on its next attempt.
    """

    name: Literal["sft", "rl"]
    system: str
    reply: type[PromptReply]
    allow_tempo_key: bool = False
    payload: Callable[[], dict] = dict
    check: Callable[[MusicRequest], None] = lambda row: None


SFT_STAGE = AuthorStage(name="sft", system=PROMPT_SYSTEM, reply=PromptReply)


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


def validate_prompt_style(
    prompt: str, brief: dict | None, *, allow_tempo_key: bool = False
) -> None:
    """Reject obvious technical leakage; genre/mood fidelity still needs listening.

    RL requests set `allow_tempo_key` because every remaining request must state both.
    SFT styles keep the old rule and leave exact tempo and key to the technical style.
    """
    if not brief or brief["style"] == "technical":
        return
    if re.search(r"\b(?:setcpm|stack|arrange|lpf|gain|scale|note|s|n)\s*\(", prompt, re.I):
        raise ValueError("This request style must not prescribe Strudel API calls")
    if brief["style"] == "musical" or allow_tempo_key:
        return
    precise = (
        r"\b\d+(?:\.\d+)?\s*(?:bpm|cpm|hz)\b|\b[A-G][#b]?[ :]+"
        r"(?:major|minor|dorian|mixolydian|phrygian|lydian|locrian)\b"
    )
    if re.search(precise, prompt, re.I):
        raise ValueError("This request style must leave exact tempo/key/effect settings open")
