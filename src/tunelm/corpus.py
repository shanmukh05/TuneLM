"""Ingest licensed compositions, preserve family splits, and verify every stem.

JSONL source and stem records are canonical; retrieval indexes can be rebuilt.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tunelm.io import content_hash, project_path, read_json, read_jsonl, write_json, write_jsonl
from tunelm.schemas import Source, Stem, Telemetry
from tunelm.verifier import Verifier

LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "CC0-1.0",
    "CC-BY-4.0",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
}


def family_split(family: str) -> str:
    bucket = int(content_hash(family)[:8], 16) % 100
    if bucket < 80:
        return "train"
    return "val" if bucket < 90 else "benchmark"


def infer_role(telemetry: Telemetry) -> tuple[str, float]:
    names = {sound.split("_")[-1] for sound in telemetry.sound_sources}
    if names & {"bd", "sd", "cp"} and not telemetry.pitches:
        return "drums_core", 0.95
    if names & {"hh", "oh", "sh", "rim", "rd", "cr"} and not telemetry.pitches:
        return "perc_tops", 0.9
    if telemetry.pitch_range and telemetry.pitch_range[1] <= 55:
        return "bass", 0.8
    if telemetry.pitch_range and telemetry.events_per_cycle >= 8:
        return "arp_lead", 0.65
    if telemetry.pitch_range:
        return "chords_pad", 0.55
    return "fx_texture", 0.5


def _verified_stems(source: Source, result: dict, split: str, verifier: Verifier) -> list[Stem]:
    decomposition = verifier.request(source.code, operation="decompose")
    if not decomposition["valid"] or not decomposition["decomposable"]:
        return []
    stems = []
    for piece in decomposition["stems"]:
        code = f"setcpm({result['cpm'] or 30});\n{piece['setup']}\n{piece['code']}"
        verified = verifier.request(code, cycles=16)
        if not verified["valid"]:
            continue
        telemetry = Telemetry.model_validate(verified["telemetry"])
        role, confidence = infer_role(telemetry)
        if source.role:
            role, confidence = source.role, 1.0
        identity = content_hash({"code": piece, "family": source.family})[:20]
        stems.append(
            Stem(
                id=identity,
                source_id=source.id,
                family=source.family,
                split=split,
                source_hash=content_hash(source.model_dump()),
                license=source.license,
                code=piece["code"],
                setup=piece["setup"],
                role=role,
                role_confidence=confidence,
                description=source.description,
                tags=source.tags,
                methods=verified["methods"],
                telemetry=telemetry,
                verifier_version=verified["runtime_version"],
            )
        )
    return stems


def ingest(sources_file: str, *, cycles: int = 32, verifier: Verifier | None = None) -> dict:
    """Rebuild a corpus from an explicit manifest; never harvest arbitrary repos."""
    verifier = verifier or Verifier()
    sources = [Source.model_validate(row) for row in read_jsonl(sources_file)]
    if len({s.id for s in sources}) != len(sources):
        raise ValueError("Duplicate source IDs")
    splits_path = project_path("datasets/manifests/splits.json")
    splits = read_json(splits_path).get("families", {}) if splits_path.exists() else {}
    seeds, stems, rejected, seen = [], {}, [], {}
    for source in sources:
        if source.license not in LICENSES:
            raise ValueError(f"Unapproved or unknown license on {source.id}: {source.license}")
        split = source.split or splits.get(source.family) or family_split(source.family)
        if source.family in splits and splits[source.family] != split:
            raise ValueError(f"Source-family split conflict: {source.family}")
        splits[source.family] = split
        fingerprint = content_hash(source.code)
        if fingerprint in seen:
            raise ValueError(f"Duplicate composition: {source.id} and {seen[fingerprint]}")
        seen[fingerprint] = source.id
        result = verifier.request(source.code, cycles=cycles)
        if not result["valid"]:
            rejected.append({"source_id": source.id, "diagnostic": result})
            continue
        extracted = _verified_stems(source, result, split, verifier)
        facts = verifier.request(source.code, operation="decompose")
        keys = facts.get("keys", [])
        category = (
            "arrange"
            if "arrange" in result["methods"]
            else ("stack" if len(extracted) > 1 else "atomic")
        )
        seeds.append(
            {
                **source.model_dump(),
                "split": split,
                "content_hash": fingerprint,
                "telemetry": result["telemetry"],
                "verifier_version": result["runtime_version"],
                "cpm": result["cpm"],
                "key": keys[0] if len(keys) == 1 else None,
                "category": category,
                "roles": sorted({s.role for s in extracted}),
            }
        )
        for stem in extracted:
            stems.setdefault(stem.id, stem)
    write_jsonl("datasets/seeds/compositions.jsonl", seeds)
    write_jsonl("datasets/stems/verified.jsonl", [s.model_dump() for s in stems.values()])
    write_jsonl("results/ingestion/rejected.jsonl", rejected)
    write_json("datasets/manifests/splits.json", {"schema_version": 2, "families": splits})
    report = {
        "schema_version": 2,
        "source_manifest": sources_file,
        "source_manifest_hash": content_hash([s.model_dump() for s in sources]),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "accepted": len(seeds),
        "stems": len(stems),
        "rejected": len(rejected),
    }
    write_json("datasets/manifests/corpus.json", report)
    return report
