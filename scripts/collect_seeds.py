"""Collect pinned, explicitly licensed Strudel sources into a JSONL manifest.

Downloads are temporary; parsed JSONL and license notices support offline reruns.
Use --ingest to verify compositions and extract stems through tunelm.corpus;
this script never calls teacher APIs, builds memory, or changes archive/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote

from pydantic import Field, field_validator

from tunelm.corpus import LICENSES, ingest
from tunelm.io import (
    ROOT,
    content_hash,
    load_config,
    output_path,
    project_path,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tunelm.schemas import Record, Source

LOG = logging.getLogger("collect_seeds")
MAX_FILE_BYTES = 2 * 1024 * 1024
HOSTS = {
    "github": (
        "https://raw.githubusercontent.com/{repo}/{revision}/{path}",
        "https://github.com/{repo}/blob/{revision}/{path}",
    ),
    "codeberg": (
        "https://codeberg.org/{repo}/raw/commit/{revision}/{path}",
        "https://codeberg.org/{repo}/src/commit/{revision}/{path}",
    ),
}


class RemoteFile(Record):
    """One immutable, bounded UTF-8 file and its reviewed content checksum."""

    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    format: str = Field(
        default="strudel",
        pattern=r"^(strudel|programs|markdown|module_objects|module_templates|json)$",
    )
    expected_candidates: int = Field(default=1, ge=1)
    retain_notice: bool = True
    url: str | None = Field(
        default=None,
        pattern=r"^https://(?:raw\.githubusercontent\.com|codeberg\.org)/[A-Za-z0-9_./%+-]+$",
    )

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
            raise ValueError("Source paths must stay within the configured repository")
        return value


class Repository(Record):
    """Reviewed source scope; a repository's software license is not inferred."""

    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    host: Literal["github", "codeberg"] = "github"
    family: str | None = None
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    license: str
    evidence: list[RemoteFile] = Field(min_length=1)
    files: list[RemoteFile] = Field(min_length=1)
    notes: str = Field(min_length=1)

    @field_validator("license")
    @classmethod
    def known_license(cls, value):
        if value not in LICENSES:
            raise ValueError("Use a source license supported by corpus ingestion")
        return value


class CollectionConfig(Record):
    """Explicit source allowlist and output settings for one corpus snapshot."""

    output_dir: str = "datasets/sources"
    local_sources: list[str] = Field(default_factory=list)
    cycles: int = Field(default=32, ge=8, le=64)
    repositories: list[Repository] = Field(min_length=1)


def source_url(repo: Repository, spec: RemoteFile, *, download=False) -> str:
    if spec.url:
        return spec.url
    template = HOSTS[repo.host][0 if download else 1]
    return template.format(repo=repo.repo, revision=repo.revision, path=quote(spec.path))


def fetch_file(repo: Repository, spec: RemoteFile, workspace: Path) -> Path:
    """Verify temporary downloads; never execute source or accept changed bytes."""
    path = output_path(workspace / repo.repo / repo.revision / spec.path)
    if path.exists():
        raw = path.read_bytes()
    else:
        url = source_url(repo, spec, download=True)
        request = urllib.request.Request(url, headers={"User-Agent": "TuneLM-seed-collector"})
        LOG.info("Downloading %s", url)
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES or hashlib.sha256(raw).hexdigest() != spec.sha256:
        raise ValueError(f"Source size/checksum mismatch: {repo.repo}/{spec.path}")
    text = raw.decode("utf-8")
    if not path.exists():
        # Reuse the atomic writer, preserving the reviewed source bytes exactly.
        from tunelm.io import _atomic_text

        _atomic_text(path, text)
    return path


def license_evidence(repo: Repository, directory: Path, workspace: Path) -> list[dict]:
    evidence = []
    for spec in repo.evidence:
        path = fetch_file(repo, spec, workspace)
        item = {
            "url": source_url(repo, spec),
            "sha256": spec.sha256,
        }
        if spec.retain_notice:
            from tunelm.io import _atomic_text

            notice = directory / "licenses" / repo.repo / repo.revision / spec.path
            _atomic_text(notice, path.read_text(encoding="utf-8"))
            item["notice_path"] = str(notice.relative_to(ROOT))
        evidence.append(item)
    return evidence


def extract_file(path: Path, format_name: str) -> list[dict]:
    """Extract standalone code, Markdown fences, or literal example records."""
    text = path.read_text(encoding="utf-8")
    if format_name in {"module_objects", "module_templates", "programs"}:
        result = subprocess.run(
            ["node", str(ROOT / "scripts/extract_seed_objects.mjs"), str(path), format_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            cwd=ROOT,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        return json.loads(result.stdout)
    if format_name == "json":
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
        records = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON example objects: {path}")
            code = row.get("code") or row.get("pattern")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"Expected a nonempty JSON code/pattern string: {path}")
            records.append({**row, "code": code, "line": 1})
        return records
    if format_name == "markdown":
        fences = re.finditer(
            r"^[ \t]{0,3}```(?:js|javascript|strudel)?[ \t]*\r?\n"
            r"(.*?)^[ \t]{0,3}```[ \t]*\r?$",
            text,
            re.M | re.S,
        )
        return [
            {"code": match[1], "line": text.count("\n", 0, match.start()) + 2} for match in fences
        ]
    return [{"code": text, "line": 1}]


def source_record(repo: Repository, spec: RemoteFile, item: dict, ordinal: int) -> Source:
    code = item["code"].strip()
    if not re.search(r"\bset(?:cpm|cps|Cpm|Cps)\s*\(", code):
        code = "setcpm(30);\n" + code
    item_id = item.get("id") or str(ordinal)
    return Source(
        id="collected-" + content_hash([repo.repo, spec.path, item_id])[:24],
        # Keep each upstream collection together: variants cannot cross splits.
        family=repo.family or f"{repo.host}:{repo.repo}",
        source=f"{source_url(repo, spec)}#L{item['line']}",
        license=repo.license,
        code=code,
        description=item.get("description") or item.get("name") or Path(spec.path).stem,
        tags=["collected", *item.get("tags", [])],
    )


def collect_repository(repo, directory, workspace):
    evidence = license_evidence(repo, directory, workspace)
    rows, provenance = [], []
    for spec in repo.files:
        path = fetch_file(repo, spec, workspace)
        candidates = extract_file(path, spec.format)
        if len(candidates) != spec.expected_candidates:
            raise ValueError(f"Unexpected candidate count: {repo.repo}/{spec.path}")
        for ordinal, item in enumerate(candidates, 1):
            row = source_record(repo, spec, item, ordinal)
            rows.append(row)
            provenance.append(
                {
                    "source_id": row.id,
                    "url": row.source,
                    "repo": repo.repo,
                    "revision": repo.revision,
                    "source_path": spec.path,
                    "source_sha256": spec.sha256,
                    "license": repo.license,
                    "license_evidence": evidence,
                    "license_scope": repo.notes,
                    "normalization": (
                        item.get(
                            "normalization",
                            "independent program extraction"
                            if spec.format == "programs"
                            else "literal extraction",
                        )
                        + "; trim; default 30 CPM only when no tempo call is present"
                    ),
                }
            )
    LOG.info("Extracted %d candidates from %s", len(rows), repo.repo)
    return rows, provenance


def unique_sources(rows):
    accepted, duplicates, seen = [], [], {}
    for row in rows:
        digest = content_hash(row.code.strip())
        if digest in seen:
            duplicates.append({"source_id": row.id, "duplicate_of": seen[digest]})
            continue
        seen[digest] = row.id
        accepted.append(row)
    if len({row.id for row in accepted}) != len(accepted):
        raise ValueError("Conflicting source IDs in the configured collections")
    return accepted, duplicates


def local_sources(config: CollectionConfig) -> list[Source]:
    return [
        Source.model_validate(value) for path in config.local_sources for value in read_jsonl(path)
    ]


def load_snapshot(config, directory):
    saved = read_json(directory / "collection.json")
    if saved.get("collector_version") != 2 or saved["config_hash"] != content_hash(
        config.model_dump()
    ):
        raise ValueError("No matching parsed snapshot; run collection without --offline")
    if saved["local_source_hash"] != content_hash([s.model_dump() for s in local_sources(config)]):
        raise ValueError("Local source inputs changed; refresh collection without --offline")
    names = {
        "seeds": "source_manifest_hash",
        "provenance": "provenance_hash",
        "duplicates": "duplicates_hash",
    }
    values = {}
    for name, hash_key in names.items():
        rows = read_jsonl(directory / f"{name}.jsonl")
        if content_hash(rows) != saved[hash_key]:
            raise ValueError(f"Parsed snapshot was modified: {name}.jsonl")
        values[name] = rows
    return (
        [Source.model_validate(r) for r in values["seeds"]],
        values["provenance"],
        values["duplicates"],
    )


def collect_inputs(config, directory, offline):
    if offline:
        return load_snapshot(config, directory)
    rows = local_sources(config)
    provenance = []
    with tempfile.TemporaryDirectory(prefix="tunelm-source-downloads-") as temporary:
        for repo in config.repositories:
            candidates, origins = collect_repository(repo, directory, Path(temporary))
            rows.extend(candidates)
            provenance.extend(origins)
    rows, duplicates = unique_sources(rows)
    return rows, provenance, duplicates


def collect(config: CollectionConfig, *, offline=False, ingest_sources=False) -> dict:
    """Save candidate/provenance manifests, optionally finish Step 1 ingestion."""
    directory = project_path(config.output_dir)
    rows, provenance, duplicates = collect_inputs(config, directory, offline)
    if not rows:
        raise ValueError("No source candidates were collected")
    sources_path = directory / "seeds.jsonl"
    write_jsonl(sources_path, [row.model_dump() for row in rows])
    write_jsonl(directory / "provenance.jsonl", provenance)
    write_jsonl(directory / "duplicates.jsonl", duplicates)
    catalog = load_config("configs/registry/sounds.yaml")
    report = {
        "schema_version": 2,
        "collector_version": 2,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(),
        "config_hash": content_hash(config.model_dump()),
        "source_manifest": str(sources_path.relative_to(ROOT)),
        "source_manifest_hash": content_hash([row.model_dump() for row in rows]),
        "provenance_hash": content_hash(provenance),
        "duplicates_hash": content_hash(duplicates),
        "local_source_hash": content_hash([s.model_dump() for s in local_sources(config)]),
        "raw_sources_retained": False,
        "sound_catalog": {"version": catalog["version"], "hash": content_hash(catalog)},
        "candidates": len(rows),
        "duplicates": len(duplicates),
        "status": "collected",
    }
    write_json(directory / "collection.json", report)
    if ingest_sources:
        report["ingestion"] = ingest(str(sources_path), cycles=config.cycles)
        report["status"] = "ingested"
        write_json(directory / "collection.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/registry/sources.yaml")
    parser.add_argument(
        "--offline", action="store_true", help="Reuse a matching parsed JSONL snapshot"
    )
    parser.add_argument(
        "--ingest", action="store_true", help="Also verify sources and extract stems"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = CollectionConfig.model_validate(load_config(args.config))
        report = collect(config, offline=args.offline, ingest_sources=args.ingest)
    except (ValueError, OSError, subprocess.SubprocessError, urllib.error.URLError) as error:
        LOG.error("Collection failed: %s", error)
        sys.exit(1)
    print(json.dumps({key: value for key, value in report.items() if key != "config"}, indent=2))


if __name__ == "__main__":
    main()
