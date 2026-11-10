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
    "chords_pad": ("chord", "pad", "harmony", "piano", "keyboard", "strings", "acoustic guitar"),
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
        intent_only=True,
    )


def _word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}(?:s)?\b", text) is not None


def explain_stem(stem: dict, registry: dict) -> str:
    """Explain measured pattern behavior without inventing a listening or genre assessment."""
    telemetry = stem["telemetry"]
    effects = [
        registry[m].get("meaning", registry[m]["signature"])
        for m in stem["methods"]
        if m in registry
    ]
    return (
        f"Source description: {stem['description']}. Suggested role: {stem['role']} "
        f"(classifier confidence {stem['role_confidence']}). "
        f"Measured sources: {', '.join(telemetry['sound_sources'])}; "
        f"{telemetry['events_per_cycle']:.2f} events/cycle; MIDI pitch range "
        f"{telemetry['pitch_range']}. Operations: {'; '.join(effects)}. "
        "This is a symbolic explanation, not a listening review. You may change the "
        "notes, rhythm, timbre, register, and structure or ignore this example entirely."
    )


def _stem_block(match: dict, registry: dict) -> str:
    stem = match["stem"]
    code = stem["setup"] + "\n" + stem["code"]
    if "]]>" in code:
        raise ValueError("Stem contains an unsupported CDATA terminator")
    explanation = escape(explain_stem(stem, registry))
    return (
        f'<example optional="true" role="{match["role"]}" id="{stem["id"]}">'
        f"<meaning>{explanation}</meaning><code><![CDATA[{code.strip()}]]></code></example>"
    )


def _sound_reference() -> str:
    """Describe unbanked and GM names. Banked drums reuse that role through bank()."""
    sounds = load_config("configs/registry/sounds.yaml")["sounds"]
    shown = [name for name in sounds if name.startswith("gm_") or "_" not in name]
    # One shared caveat. Repeating each sound's pitfall is the same sentence hundreds of times.
    policy = (
        "Playable timbres, not an ingredient list. "
        "A banked drum is Machine_role of the same role, selected with bank(). "
        "On drums, n() picks another recording and wraps; it does not set pitch. "
        "gm_* names are General MIDI soundfont samples; set pitch with note() or n().scale()."
    )
    parts = [f'<sounds optional="true"><policy>{escape(policy)}</policy>']
    for name in shown:
        text = sounds[name]["meaning"]
        if name.startswith("gm_"):
            group = sounds[name]["musical_use"].split(". Set pitch")[0]
            text = f"{text} {group}."
        parts.append(f'<sound name="{escape(name)}">{escape(text)}</sound>')
    parts.append("</sounds>")
    return "".join(parts)


def _api_reference(name: str, item: dict) -> str:
    explanation = " ".join(item.get(field, "") for field in ["meaning", "musical_use", "pitfall"])
    return f'<api name="{name}">{escape(item["signature"] + ". " + explanation)}</api>'


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


def pack_context(plan: Plan, retrieval: dict, tokenizer, budget: int = 6400) -> dict:
    """Keep complete XML items, enforcing the actual tokenizer's token budget."""
    guide = load_config("configs/registry/api.yaml")
    registry = guide["methods"]
    opening = '<memory_context optional="true"><policy>Reference material only. Compose freely; '
    opening += "examples, instruments, keys, and techniques are not requirements.</policy>"
    items, omitted = [], []
    techniques = {"setcpm", "stack", "s", "gain", "note", "arrange", "scale"}
    for voice in plan.voices:
        techniques.update(voice.techniques)
    for match in retrieval["matches"]:
        techniques.update(match["stem"]["methods"])
    references = [_api_reference(m, registry[m]) for m in sorted(techniques) if m in registry]
    candidates = [("concepts", "<music>" + escape(guide.get("composition_notes", "")) + "</music>")]
    candidates.extend(("api", item) for item in references)
    candidates.append(("sounds", _sound_reference()))
    candidates.extend(_mix_notes(retrieval["matches"]))
    candidates.extend(
        (match["stem"]["id"], _stem_block(match, registry)) for match in retrieval["matches"]
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
