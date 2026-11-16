"""Render verified Strudel compositions as benchmark WAVs."""

from __future__ import annotations

import json
import math
import os
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tunelm.core.io import ROOT, output_path
from tunelm.strudel.verifier import Verifier

RENDER_SECONDS = 30


def render_audio(
    code: str,
    target: Path,
    cache: Path,
    seconds: float = RENDER_SECONDS,
    *,
    scoring_seconds: float = 0,
) -> None:
    verifier = Verifier(timeout=12)
    result = verifier.request(code, cycles=32)
    if not result.get("valid"):
        raise ValueError(f"Strudel verification failed: {result.get('error')}")
    cpm = result["cpm"] or 30
    if result["cycles"] * 60 / cpm < seconds:
        cycles = math.ceil(seconds * cpm / 60)
        if cycles > 64:
            raise ValueError("Render window exceeds the verifier's 64-cycle limit")
        result = verifier.request(code, cycles=cycles)
        if not result.get("valid"):
            raise ValueError(f"Strudel verification failed: {result.get('error')}")
    request = {"cpm": cpm, "seconds": seconds, "events": result["events"]}
    target = output_path(target)
    with TemporaryDirectory(prefix="tunelm-render-", dir=target.parent) as scratch:
        temporary = Path(scratch) / "audio.wav"
        proc = subprocess.run(
            [
                "node",
                str(ROOT / "scripts" / "render_benchmark_audio.mjs"),
                str(temporary),
                str(cache),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=150,
            cwd=scratch,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            check=False,
        )
        if proc.returncode or not temporary.is_file():
            raise RuntimeError(proc.stderr.strip() or "Strudel renderer produced no WAV")
        if scoring_seconds:
            with wave.open(str(temporary), "rb") as wav:
                frames = wav.readframes(round(scoring_seconds * wav.getframerate()))
            if not np.any(np.frombuffer(frames, dtype="<i2").reshape(-1, 2).sum(axis=1)):
                raise ValueError("Strudel rendered silence in the scoring window")
        temporary.replace(target)
