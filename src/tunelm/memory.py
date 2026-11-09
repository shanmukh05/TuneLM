"""Build and query a reproducible stem index from canonical JSONL records.

SQLite and NumPy are derived artifacts. Semantic and explicit offline lexical
embeddings use the same metadata filters and deterministic ranking.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from tunelm.io import content_hash, output_path, project_path, read_json, read_jsonl, write_json
from tunelm.schemas import Plan, Stem


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_]+", text.lower())


def _lexical(texts: list[str], vocabulary: list[str]) -> np.ndarray:
    lookup = {word: i for i, word in enumerate(vocabulary)}
    values = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in _words(text):
            if word in lookup:
                values[row, lookup[word]] += 1
    return _normalize(values)


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def embed(texts: list[str], settings: dict) -> np.ndarray:
    """Encode consistently; lexical mode is explicit and never called semantic."""
    if settings["mode"] == "lexical":
        return _lexical(texts, settings["vocabulary"])
    if settings["mode"] != "semantic" or not re.fullmatch(
        r"[a-f0-9]{40}", settings.get("revision", "")
    ):
        raise ValueError("Semantic embeddings require an explicit model and pinned revision")
    model = _encoder(settings["model"], settings["revision"])
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


@lru_cache(maxsize=2)
def _encoder(model: str, revision: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model, revision=revision)


def build_memory(stems_file: str, directory: str, settings: dict) -> dict:
    """Index train stems only; held-out source families never enter retrieval."""
    stems = [Stem.model_validate(row) for row in read_jsonl(stems_file)]
    stems = sorted((s for s in stems if s.split == "train"), key=lambda s: s.id)
    if not stems or len({s.id for s in stems}) != len(stems):
        raise ValueError("Memory needs nonempty, uniquely identified training stems")
    texts = [s.description + " " + " ".join(s.tags) + " " + s.role for s in stems]
    settings = dict(settings)
    if settings["mode"] == "lexical":
        settings["vocabulary"] = sorted({word for text in texts for word in _words(text)})
    vectors = _normalize(embed(texts, settings))
    records = [s.model_dump() for s in stems]
    version = content_hash({"stems": records, "embedding": settings})
    dest = output_path(Path(directory) / "index.sqlite")
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute(
                "CREATE TABLE stems (id TEXT PRIMARY KEY, role TEXT, density REAL, "
                "rowid_vector INTEGER, record TEXT)"
            )
            import json

            db.executemany(
                "INSERT INTO stems VALUES (?, ?, ?, ?, ?)",
                [
                    (s.id, s.role, s.telemetry.events_per_cycle, i, json.dumps(s.model_dump()))
                    for i, s in enumerate(stems)
                ],
            )
            db.execute("CREATE TABLE version (value TEXT)")
            db.execute("INSERT INTO version VALUES (?)", (version,))
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    with output_path(Path(directory) / "embeddings.npy").open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    metadata = {
        "schema_version": 2,
        "version": version,
        "embedding": settings,
        "dimension": vectors.shape[1],
        "ids": [s.id for s in stems],
        "source_hash": content_hash(records),
        "split": "train",
    }
    write_json(Path(directory) / "metadata.json", metadata)
    return metadata


class Memory:
    """Read a frozen corpus snapshot; no mutations occur during generation or RL."""

    def __init__(self, directory: str = "datasets/memory"):
        self.directory = project_path(directory)
        self.metadata = read_json(self.directory / "metadata.json")
        self.vectors = np.load(self.directory / "embeddings.npy", allow_pickle=False)
        if self.vectors.shape != (len(self.metadata["ids"]), self.metadata["dimension"]):
            raise ValueError("Embedding shape does not match index metadata")
        with self._database() as db:
            if db.execute("SELECT value FROM version").fetchone()[0] != self.version:
                raise ValueError("Mixed memory index versions; rebuild memory")

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @contextmanager
    def _database(self):
        uri = (self.directory / "index.sqlite").as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            yield db

    def retrieve(self, plan: Plan, *, exclude_families: list[str] | None = None) -> dict:
        import json

        matches, missing = [], []
        queries = embed([v.query for v in plan.voices], self.metadata["embedding"])
        for voice, query in zip(plan.voices, queries, strict=True):
            with self._database() as db:
                rows = db.execute(
                    "SELECT rowid_vector, record FROM stems "
                    "WHERE role=? AND density BETWEEN ? AND ? ORDER BY id",
                    (voice.role, *voice.density),
                ).fetchall()
            ranked = []
            for index, raw in rows:
                stem = Stem.model_validate(json.loads(raw))
                if stem.family in (exclude_families or []):
                    continue
                overlap = len(set(voice.techniques) & set(stem.methods)) / max(
                    1, len(voice.techniques)
                )
                score = (
                    0.65 * float(self.vectors[index] @ query)
                    + 0.25 * overlap
                    + 0.10 * stem.telemetry.structural_variation
                )
                ranked.append((score, stem))
            ranked.sort(key=lambda item: (-item[0], item[1].id))
            if not ranked:
                missing.append(voice.role)
                continue
            score, stem = ranked[0]
            matches.append(
                {"role": voice.role, "score": round(score, 5), "stem": stem.model_dump()}
            )
        return {"version": self.version, "matches": matches, "missing_roles": missing}
