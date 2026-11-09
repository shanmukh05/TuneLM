"""Conservative prompt-to-plan rules and tokenizer-bounded memory formatting.

Explicit JSON plans are supported when prose needs detail the basic rules miss.
"""

from __future__ import annotations

import re
from html import escape

from tunelm.io import load_config
from tunelm.schemas import Plan, Voice

ROLE_WORDS = {
    "drums_core": ("drum", "kick", "snare", "beat"),
    "perc_tops": ("hat", "shaker", "percussion", "tops"),
    "bass": ("bass", "bassline", "low end", "low-end"),
    "chords_pad": ("chord", "pad", "harmony"),
    "arp_lead": ("arp", "lead", "melody", "hook"),
    "fx_texture": ("texture", "noise", "sweep", "drone"),
}


def make_plan(prompt: str) -> Plan:
    """Infer common cues; store which choices were explicit instead of guessing silently."""
    text = prompt.lower()
    bpm = re.search(r"\b(\d+(?:\.\d+)?)\s*bpm\b", text)
    key = re.search(
        r"\b([a-gA-G][#b]?)[ :]+(major|minor|dorian|mixolydian|phrygian|lydian|locrian)\b",
        prompt,
        re.I,
    )
    roles = [role for role, words in ROLE_WORDS.items() if any(_word(text, w) for w in words)]
    roles = roles or ["drums_core", "bass", "chords_pad"]
    atomic = any(word in text for word in ["only", "single voice", "atomic", "bassline alone"])
    if atomic:
        roles = roles[:1]
    category = "atomic" if atomic else "stack"
    if any(word in text for word in ["intro", "breakdown", "section", "arrangement"]):
        category = "arrange"
    short = any(word in text for word in ["one bar", "one-bar", "static", "short loop"])
    explicit = [
        name
        for name, found in [
            ("bpm", bpm),
            ("key", key),
            ("roles", any(_word(text, w) for words in ROLE_WORDS.values() for w in words)),
        ]
        if found
    ]
    # "Coffee", "slow music", and "fast workout" do not request API calls.
    techniques = [
        name
        for name in ["euclid", "jux", "off", "lpf", "slow", "fast"]
        if re.search(rf"\b{name}\s*\(|\b(?:use|using|apply)\s+{name}\b", text)
    ]
    pitched = any(r in roles for r in ["bass", "chords_pad", "arp_lead"])
    return Plan(
        bpm=float(bpm[1]) if bpm else 120,
        key=(key[1][0].upper() + key[1][1:] + ":" + key[2].lower())
        if key
        else ("C:minor" if pitched else None),
        category=category,
        voices=[Voice(role=r, query=prompt, techniques=techniques) for r in roles],
        explicit=explicit,
        short_loop=short,
        cycles=32 if category == "arrange" else 16,
    )


def _word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}(?:s)?\b", text) is not None


def _stem_block(match: dict) -> str:
    stem = match["stem"]
    code = stem["setup"] + "\n" + stem["code"]
    if "]]>" in code:
        raise ValueError("Stem contains an unsupported CDATA terminator")
    return f'<stem role="{match["role"]}" id="{stem["id"]}"><![CDATA[{code.strip()}]]></stem>'


def _mix_notes(matches: list[dict]) -> list[tuple[str, str]]:
    by_role = {m["role"]: m["stem"] for m in matches}
    if not {"bass", "chords_pad"} <= set(by_role):
        return []
    filtered = "lpf" in by_role["bass"]["methods"] or "hpf" in by_role["chords_pad"]["methods"]
    if filtered:
        return []
    return [
        (
            "mix",
            "<mix>Consider separating bass and pad registers or filtering the pad; "
            "balance gains and verify by listening.</mix>",
        )
    ]


def pack_context(plan: Plan, retrieval: dict, tokenizer, budget: int = 550) -> dict:
    """Keep complete XML items, enforcing the actual tokenizer's token budget."""
    registry = load_config("configs/registry/api.yaml")["methods"]
    opening = f'<memory_context><defaults cpm="{plan.cpm}" key="{escape(plan.key or "none")}"/>'
    items, omitted = [], []
    techniques = {"setcpm", "stack", "s", "gain"}
    for voice in plan.voices:
        techniques.update(voice.techniques)
    for match in retrieval["matches"]:
        techniques.update(match["stem"]["methods"])
    references = [
        f"<api>{escape(registry[m]['signature'])}</api>"
        for m in sorted(techniques)
        if m in registry
    ]
    candidates = [("api", item) for item in references]
    candidates.extend(_mix_notes(retrieval["matches"]))
    candidates.extend((match["stem"]["id"], _stem_block(match)) for match in retrieval["matches"])
    candidates.extend(
        ("missing", f'<missing role="{role}"/>') for role in retrieval["missing_roles"]
    )
    for name, item in candidates:
        candidate = opening + "".join(items + [item]) + "</memory_context>"
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            items.append(item)
        else:
            omitted.append(name)
    text = opening + "".join(items) + "</memory_context>"
    count = len(tokenizer.encode(text, add_special_tokens=False))
    if count > budget:
        raise ValueError("Context budget cannot fit even the global defaults")
    return {
        "context": text,
        "tokens": count,
        "omitted": omitted,
        "missing_roles": retrieval["missing_roles"],
        "memory_version": retrieval["version"],
    }
